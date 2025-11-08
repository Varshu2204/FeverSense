import streamlit as st
import requests
import pandas as pd
import plotly.express as px

# -------------------- CONFIG --------------------
BACKEND_URL = "http://127.0.0.1:5000"
st.set_page_config(page_title="FeverSense", page_icon="🩺", layout="wide")

# -------------------- HEADER --------------------
st.markdown("""
    <h1 style="text-align:center; color:#00B4D8;">🤖 FeverSense: AI-Assisted Diagnostics & Triage</h1>
    <p style="text-align:center; color:#AAAAAA;">Smart clinical triage prediction with personalized medication recommendation</p>
""", unsafe_allow_html=True)


# -------------------- PATIENT FORM --------------------
st.subheader("🧍 Patient Input Form")

with st.form("triage_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        name = st.text_input("Name", placeholder="e.g. Varsha")
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        lifestyle = st.selectbox("Lifestyle", ["Sedentary", "Moderate", "Active"])

    with col2:
        # Temperature + Unit
        st.write("**Temperature**")
        temp_col1, temp_col2 = st.columns([3, 1])
        with temp_col1:
            temperature = st.number_input("Value", min_value=30.0, max_value=110.0, value=37.0, step=0.1, label_visibility="collapsed")
        with temp_col2:
            unit = st.selectbox("Unit", ["°C", "°F"], label_visibility="collapsed")

        heart_rate = st.number_input("Heart Rate (bpm)", 30, 200, 85)
        blood_pressure = st.selectbox("Blood Pressure", ["Normal", "High", "Low"])

    with col3:
        days_fever = st.number_input("Days of Fever", 0, 20, 2)
        cough = st.selectbox("Cough", ["No", "Yes"])
        rash = st.selectbox("Rash", ["No", "Yes"])
        headache = st.selectbox("Headache", ["No", "Yes"])

    st.markdown("---")
    col4, col5, col6 = st.columns(3)
    with col4:
        fatigue = st.selectbox("Fatigue", ["No", "Yes"])
        chronic = st.selectbox("Chronic Conditions", ["No", "Yes"])
    with col5:
        allergies = st.selectbox("Allergies", ["No", "Yes"])
        smoking = st.selectbox("Smoking History", ["No", "Yes"])
    with col6:
        alcohol = st.selectbox("Alcohol Consumption", ["No", "Yes"])
        diet = st.selectbox("Diet Type", ["Vegetarian", "Non-Vegetarian", "Vegan"])

    previous_med = st.selectbox("Previous Medication", ["None", "Paracetamol", "Ibuprofen"])

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("🔍 Analyze & Predict", use_container_width=True)

# -------------------- PREDICTION SECTION --------------------
if submitted:
    # Convert Fahrenheit → Celsius if needed
    temperature_c = round((temperature - 32) * 5/9, 2) if unit == "°F" else temperature

    payload = {
        "Name": name, "Age": age, "Gender": gender,
        "Temperature": temperature_c, "Heart_Rate": heart_rate,
        "Blood_Pressure": blood_pressure, "Days_Fever": days_fever,
        "Cough": cough, "Rash": rash, "Headache": headache,
        "Fatigue": fatigue, "Chronic_Conditions": chronic,
        "Allergies": allergies, "Smoking_History": smoking,
        "Alcohol_Consumption": alcohol, "Lifestyle": lifestyle,
        "Diet_Type": diet, "Previous_Medication": previous_med,
        "unit": unit
    }

    with st.spinner("🤖 AI analyzing patient data..."):
        try:
            res = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=15)
            if res.status_code == 200:
                result = res.json()
                triage = result.get("predicted_triage", "Unknown")
                conf = result.get("triage_confidence", 0)
                med = result.get("recommended_medication", "N/A")

                # Color mapping
                color_map = {"Normal": "#2E8B57", "Mild Fever": "#FFD700", "High Fever": "#C0392B"}
                bg_color = color_map.get(triage, "#3A3A3A")

                st.markdown(f"""
                    <div style="background-color:{bg_color};padding:20px;border-radius:10px;text-align:center;color:white;">
                        <h3>🩺 Predicted Triage: {triage}</h3>
                        <p><b>Confidence:</b> {round(conf*100,2)}%</p>
                    </div>
                """, unsafe_allow_html=True)

                if "No medication" in med:
                    st.info("💧 Rest well, stay hydrated, and monitor your temperature regularly.")
                else:
                    st.success(f"💊 Recommended Medication: **{med}**")

                if med.lower().startswith("para"):
                    st.info("💡 Dosage Suggestion: Paracetamol 500 mg every 6 hours as needed.")
                elif med.lower().startswith("ibu"):
                    st.info("💡 Dosage Suggestion: Ibuprofen 400 mg every 8 hours (post meals).")

                # Feature importance chart
                if "feature_importances" in result:
                    imp = pd.DataFrame(list(result["feature_importances"].items()), columns=["Feature", "Importance"])
                    imp = imp.sort_values(by="Importance", ascending=False).head(8)
                    st.subheader("🔬 Key Influencing Features")
                    fig = px.bar(imp, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Tealgrn")
                    st.plotly_chart(fig, use_container_width=True)

                # Input summary
                with st.expander("📋 Patient Input Summary"):
                    df_input = pd.DataFrame(payload.items(), columns=["Parameter", "Value"])
                    st.dataframe(df_input, use_container_width=True)

            else:
                st.error(f"Backend Error: {res.text}")
        except Exception as e:
            st.error(f"Connection Error: {e}")

# -------------------- HISTORY SECTION --------------------
st.markdown("---")
st.subheader("📜 Previous Patient History")

colh1, colh2 = st.columns([3, 1])
with colh1:
    view_btn = st.button("📂 View History", use_container_width=True)
with colh2:
    delete_btn = st.button("🗑️ Delete All Records", use_container_width=True)

if view_btn:
    try:
        res = requests.get(f"{BACKEND_URL}/history", timeout=10)
        if res.status_code == 200:
            history = res.json()
            if history:
                df = pd.DataFrame(history)
                st.dataframe(df[['id', 'name', 'age', 'triage', 'medication', 'confidence', 'timestamp']], use_container_width=True)
                st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode('utf-8'), "patient_history.csv", "text/csv", use_container_width=True)
            else:
                st.info("No patient records found yet.")
        else:
            st.error("Failed to fetch history.")
    except Exception as e:
        st.error(f"Error fetching history: {e}")

if delete_btn:
    with st.expander("⚠️ Confirm Deletion", expanded=True):
        st.warning("Are you sure you want to delete **ALL patient records**? This action cannot be undone.")
        coldel1, coldel2 = st.columns(2)
        with coldel1:
            confirm_delete = st.button("✅ Yes, Delete All", use_container_width=True)
        with coldel2:
            cancel_delete = st.button("❌ Cancel", use_container_width=True)

        if confirm_delete:
            try:
                res = requests.delete(f"{BACKEND_URL}/delete_history", timeout=10)
                if res.status_code == 200:
                    st.success("🧹 All patient data cleared successfully!")
                else:
                    st.error("Failed to delete data.")
            except Exception as e:
                st.error(f"Error deleting data: {e}")
        elif cancel_delete:
            st.info("Deletion cancelled. Your records are safe ✅")
