# src/routes/attendance_routes.py
from flask import Blueprint, request, jsonify, current_app, url_for
from src.extensions import db
from src.models.attendance import Attendance
from src.ml_models.recognition_service import recognize_face_from_image, load_recognition_models
import base64
import numpy as np
import cv2 as cv
from datetime import datetime
import os

attendance_bp = Blueprint("attendance_bp", __name__)

# Ensure models are loaded when the blueprint is initialized or before the first request
# Depending on Flask app structure, this might be better in main.py or using app.before_first_request
# For simplicity here, we can try to ensure it's loaded.
load_recognition_models() # Call it here to ensure models are loaded when this module is imported.

PROFILE_PICS_STATIC_PATH = "profile_pics"
CAPTURED_IMAGES_STATIC_PATH = "captured_images"

@attendance_bp.route("/recognize", methods=["POST"])
def recognize_and_log_attendance():
    CAPTURED_IMAGES_DIR = os.path.join(current_app.static_folder, CAPTURED_IMAGES_STATIC_PATH)
    PROFILE_PICS_DIR = os.path.join(current_app.static_folder, PROFILE_PICS_STATIC_PATH)

    for p_dir in [CAPTURED_IMAGES_DIR, PROFILE_PICS_DIR]:
        if not os.path.exists(p_dir):
            try:
                os.makedirs(p_dir)
                current_app.logger.info(f"Created directory: {p_dir}")
            except OSError as e:
                current_app.logger.error(f"Error creating directory {p_dir}: {e}")
                # If directory creation fails, it might be critical

    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided", "name": None, "user_image_url": None}), 400

    image_data_url = data["image"]
    
    try:
        header, encoded_data = image_data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_np = cv.imdecode(nparr, cv.IMREAD_COLOR)
        
        if img_np is None:
            current_app.logger.error("Failed to decode image from base64 string.")
            return jsonify({"error": "Invalid image data", "name": None, "user_image_url": None}), 400

    except Exception as e:
        current_app.logger.error(f"Error processing image data: {e}")
        return jsonify({"error": f"Error processing image: {str(e)}", "name": None, "user_image_url": None}), 500

    recognition_result_status = recognize_face_from_image(img_np)
    current_time = datetime.utcnow()
    user_image_url = None

    if recognition_result_status == "NO_FACE_DETECTED":
        current_app.logger.info("No face detected in the image.")
        return jsonify({
            "name": "NO_FACE_DETECTED", 
            "message": "No face detected.",
            "user_image_url": None
        }), 200
    
    elif recognition_result_status == "Unknown":
        current_app.logger.info("Face detected but not recognized (Unknown).")
        return jsonify({
            "name": "Unknown", 
            "message": "Face recognized as Unknown. Attendance not recorded.",
            "user_image_url": None
        }), 200

    elif recognition_result_status and recognition_result_status not in ["NO_FACE_DETECTED", "Unknown", None]:
        # This means a known person was recognized
        recognized_name = recognition_result_status
        try:
            profile_pic_basename = recognized_name.replace(" ", "_")
            possible_extensions = [".jpg", ".jpeg", ".png"]
            profile_pic_filename = None
            for ext in possible_extensions:
                temp_filename = profile_pic_basename + ext
                if os.path.exists(os.path.join(PROFILE_PICS_DIR, temp_filename)):
                    profile_pic_filename = temp_filename
                    break
            
            if profile_pic_filename:
                user_image_url = url_for("static", filename=f"{PROFILE_PICS_STATIC_PATH}/{profile_pic_filename}", _external=False)
                current_app.logger.info(f"Found profile picture: {user_image_url}")
            else:
                current_app.logger.info(f"No profile picture found for {recognized_name} in {PROFILE_PICS_DIR}")

            timestamp_str = current_time.strftime("%Y%m%d_%H%M%S_%f")
            snapshot_filename = f"{profile_pic_basename}_{timestamp_str}.jpg"
            # Use CAPTURED_IMAGES_DIR which is an absolute path for saving
            snapshot_path = os.path.join(CAPTURED_IMAGES_DIR, snapshot_filename)
            cv.imwrite(snapshot_path, img_np)

            new_attendance = Attendance(
                name=recognized_name,
                timestamp=current_time,
                image_filename=os.path.join(CAPTURED_IMAGES_STATIC_PATH, snapshot_filename) # Store relative path for static serving
            )
            db.session.add(new_attendance)
            db.session.commit()
            current_app.logger.info(f"Attendance logged for {recognized_name} at {current_time}")
            return jsonify({
                "name": recognized_name,
                "time": current_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "message": "Attendance recorded successfully",
                "user_image_url": user_image_url
            }), 200
        except Exception as e:
            db.session.rollback()
            current_app.logger.error(f"Database or file saving error for {recognized_name}: {e}")
            return jsonify({"error": f"Failed to record attendance: {str(e)}", "name": recognized_name, "user_image_url": user_image_url}), 500
    else:
        # Handles None from recognition_service or other unexpected values
        current_app.logger.error(f"Recognition service returned an unexpected value: {recognition_result_status}")
        return jsonify({"error": "Recognition failed or returned unexpected data", "name": None, "user_image_url": None}), 500

