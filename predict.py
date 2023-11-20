from flask import Flask, request, jsonify
import joblib
from PIL import Image
import numpy as np
from io import BytesIO

# Load the model
# model = joblib.load('/digits/API/best_model.pkl')
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/predict_digit', methods=['POST'])
def predict_digit_endpoint():
    if 'image' not in request.files:
        return jsonify(error='Please provide an image.'), 400

    image_bytes = request.files['image'].read()
    image = Image.open(BytesIO(image_bytes)).convert('L')
    resized_image = image.resize((8, 8), Image.LANCZOS)
    
    image_array = np.array(resized_image).reshape(1, -1)
    prediction = model.predict(image_array)

    return jsonify(predicted_digit=int(prediction[0]))



if __name__ == '__main__':
    app.run(debug=True)
