FeverSense: AI-Assisted Diagnostics and Triage

FeverSense is an AI-driven diagnostic system designed to assist healthcare professionals and patients in identifying the severity of fever conditions and recommending appropriate medications and triage levels. The platform integrates machine learning models, real-time patient data input, and an intuitive user interface to provide intelligent fever management and diagnostic support.

Project Overview

FeverSense leverages Artificial Intelligence to analyze symptoms, vital parameters, and patient history to determine the triage level — Normal, Mild, Moderate, or Severe — along with suitable medical recommendations. The system aims to minimize manual diagnostic errors and streamline early intervention in fever-related cases.

Key Features

AI-based Fever Severity Prediction using trained ML models.

Dynamic Triage Classification — Normal, Mild, Moderate, Severe.

Smart Medical Recommendation System — suggests medications and dosages.

Temperature Unit Flexibility — supports both Celsius and Fahrenheit inputs.

Real-time Visualization of patient data and confidence levels.

Patient History Management — stores and retrieves past records.

Database Integration with SQLite for persistent data storage.

Secure “Delete All Records” option with confirmation alerts.

Clean, professional Streamlit-based UI with form input and instant prediction.

Flask Backend API integration with Swagger documentation.

Tech Stack

Backend: Flask
Frontend: Streamlit
Database: SQLite
Machine Learning: Scikit-learn, Pandas, NumPy
Visualization: Plotly
Language: Python
Version Control: Git, GitHub

System Architecture

The system consists of two major components:

Backend (Flask API) – Handles model training, prediction, and database management.

Frontend (Streamlit UI) – Provides an interactive interface for user input, displays predictions, and visualizes triage confidence and medical recommendations.

How It Works

User enters patient details such as temperature, heart rate, blood pressure, days of fever, and symptoms.

The trained machine learning model processes the input data.

FeverSense predicts the triage level and confidence percentage.

The system recommends suitable medications and dosage suggestions based on the severity.

The results are stored in the database and can be visualized for future reference.

Challenges Faced

Ensuring accurate classification across multiple fever severity levels.

Integrating frontend and backend seamlessly using RESTful APIs.

Managing data persistence and ensuring database synchronization.

Designing an intuitive yet professional user interface suitable for medical contexts.

Handling inconsistent input formats (e.g., Celsius/Fahrenheit conversions).

Optimizing model performance with limited dataset diversity.

Achievements and Highlights

Fully functional AI-assisted triage system deployed locally.

Developed an explainable and interactive healthcare diagnostic tool.

Improved user experience through dynamic UI and visualization.

Achieved stable backend integration with database and frontend modules.

Simplified medical diagnostics through real-time prediction and data-driven insights.

What We Learned

Flask-Streamlit integration and REST API design.

End-to-end machine learning deployment with real-world datasets.

Effective use of SQLite for patient data management.

Practical knowledge of debugging and model optimization.

Importance of user interface design in medical applications.

Future Enhancements

Integration of cloud-based storage and APIs for multi-user support.

Expanding dataset with hospital-grade patient data for higher accuracy.

Incorporating live IoT-based temperature and heart rate sensors.

Deployment on cloud (AWS/Heroku) for real-time accessibility.

Enhancing explainability through AI interpretability dashboards.

How to Run

Clone the repository:
git clone https://github.com/Varshu2204/FeverSense.git

Install dependencies:
pip install -r requirements.txt

Run the backend:
python -m backend.app

Run the frontend:
streamlit run frontend/streamlit_app.py

Access the app locally and start diagnosing fever severity intelligently.

Contributors

Team: FeverSense
Developed by: Varshu H R, Kushaal V, Monika G, Denzil Hevin Patil
Institution: BMS College of Engineering
Domain: AI-Assisted Diagnostics & Healthcare
