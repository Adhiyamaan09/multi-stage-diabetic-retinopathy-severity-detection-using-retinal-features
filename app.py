import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image

from python_scripts.feature_extraction import BloodVessel, Exudates
from python_scripts.get_prediction import ImagePrediction


app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process_image", methods=["POST"])
def process_image():
    if request.method == "POST":
        if 'image' not in request.files:
            return jsonify({"error": "No image file found"}), 400
        
        filename = request.files["image"]
        image = Image.open(filename)

        try:
            BV = BloodVessel(image)
            blood_vessels_image = BV.extractBloodVessels()

            EX = Exudates(image)
            exudates_image = EX.extractExudates()
            
            return jsonify({"vesselimage": blood_vessels_image, "exudateimage": exudates_image}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route("/get_prediction", methods=["POST"])
def get_prediction():
    if request.method == "POST":
        if 'image' not in request.files:
            return jsonify({"error": "No image file found"}), 400
        
        filename = request.files["image"]
        image = Image.open(filename)

        try:
            IP = ImagePrediction()
            pred_class, pred_probability, pred_graph_image = IP.getPrediction(image)
            
            return jsonify({"predclass": pred_class, "predprob": pred_probability, "predgraphimage": pred_graph_image}), 200
        except Exception as e:
            print(e)
            return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(debug = True)