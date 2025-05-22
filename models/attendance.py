from src.extensions import db
from datetime import datetime

class Attendance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    image_filename = db.Column(db.String(200), nullable=True) # Path to the captured image

    def __repr__(self):
        return f"<Attendance {self.name} @ {self.timestamp}>"

