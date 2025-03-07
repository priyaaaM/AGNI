import os
from dotenv import load_dotenv

from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import atexit

from flask import Flask, render_template, request, jsonify

from src.satellite_functions import satellite_cnn_predict
from src.camera_functions import camera_cnn_predict
from src.meteorological_functions import weather_data_predict


# Load environment variables from .env file
load_dotenv()
MAPBOX_TOKEN = os.getenv("MAPBOX_TOKEN")

# Function that runs email_alert.py script
def run_alert_script():
    script_path = os.path.join(os.path.dirname(__file__), "email_alert.py")
    try:
        subprocess.run(["python", script_path], check=True)
        print("Processing script executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error running processing script: {e}")


# Initialize the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(func=run_alert_script, trigger="interval", hours=1)
scheduler.start()

# Ensure the scheduler is shut down properly on exit
atexit.register(lambda: scheduler.shutdown())


app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/detect/camera")
def detect_camera():
    return render_template("detect_camera.html")


@app.route("/detect/satellite")
def detect_satellite():
    return render_template("detect_satellite.html", MAPBOX_TOKEN=MAPBOX_TOKEN)


# The route for predicting wildfire using satellite data
@app.route("/satellite_predict", methods=["POST"])
def satellite_predict():
    data = request.json
    latitude = data["location"][1]
    longitude = data["location"][0]
    zoom = data["zoom"]

    output_size = (350, 350)
    crop_amount = 35
    save_path = "satellite_image.png"

    prediction_sattelite = satellite_cnn_predict(
        latitude,
        longitude,
        output_size=output_size,
        zoom_level=zoom,
        crop_amount=crop_amount,
        save_path=save_path,
    )

    satellite_confidence = round(
        (
            prediction_sattelite
            if prediction_sattelite > 0.5
            else 1 - prediction_sattelite
        )
        * 100
    )  # float to percentage

    satellite_status = 1 if prediction_sattelite > 0.5 else 0

    prediction_weather = weather_data_predict(latitude, longitude)

    weather_confidence = round(
        (prediction_weather if prediction_weather > 0.5 else 1 - prediction_weather)
        * 100
    )  # float to percentage

    weather_status = 1 if prediction_weather > 0.5 else 0

    # Calculate average probability and its corresponding binary status
    prediction_average = (prediction_sattelite + prediction_weather) / 2

    average_confidence = round(
        (prediction_average if prediction_average > 0.5 else 1 - prediction_average)
        * 100
    )  # float to percentage

    average_status = 1 if prediction_average > 0.5 else 0

    response = {
        "satellite_probability": satellite_confidence,
        "satellite_status": satellite_status,
        "weather_probability": weather_confidence,
        "weather_status": weather_status,
        "average_probability": average_confidence,
        "average_status": average_status,
    }

    return jsonify(response), 200


# The route for predicting wildfire using camera images
@app.route("/camera_predict", methods=["POST"])
def camera_predict():
    image_file = request.files["image"]
    prediction = camera_cnn_predict(image_file)

    confidence = round(
        (prediction if prediction > 0.5 else 1 - prediction) * 100)

    # Alphabetically *fire* (0) comes before *no fire* (1)
    wildfire_prediction = 1 if prediction < 0.5 else 0

    response_data = {
        "wildfire_prediction": wildfire_prediction,
        "confidence": confidence,
    }

    return jsonify(response_data), 200


if __name__ == "__main__":
   
    app.run(debug=True)
