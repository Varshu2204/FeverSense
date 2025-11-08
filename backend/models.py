from datetime import datetime
from .db import db

class Patient(db.Model):
    __tablename__ = "patients"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=True)
    age = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_snapshot = db.Column(db.JSON, nullable=True)

class PredictionRecord(db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=True)
    input_features = db.Column(db.JSON, nullable=False)
    predicted_class = db.Column(db.Integer, nullable=False)
    predicted_label = db.Column(db.String(32))
    probability = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    patient = db.relationship("Patient", backref=db.backref("predictions", lazy=True))
