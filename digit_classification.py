import base64
from flask import Flask, request, jsonify
from PIL import Image
import io
import numpy as np
from sklearn import datasets, svm
from utils import prepare_data_splits

app = Flask(__name__)

# Load the digits dataset
digits_data = datasets.load_digits()
X_data = digits_data.data
y_data = digits_data.target

# Train an SVM model for digit classification
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(X_data, y_data)

def preprocess_base64_image(base64_data):
    # Decode base64 data to binary
    image_data = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_data))
    # Resize the image to match the dataset's image size
    image = image.resize((8, 8))
    image = np.array(image)
    return image

@app.route('/predict', methods=['POST'])
def compare_images():
    try:
        data = request.get_json()
        base64_image1 = data.get('image1')
        base64_image2 = data.get('image2')

        if not base64_image1 or not base64_image2:
            return jsonify({"result": "Invalid input"}), 400

        # Preprocess the images
        image1 = preprocess_base64_image(base64_image1)
        image2 = preprocess_base64_image(base64_image2)

        # Predict the digit labels for both images
        digit1 = clf.predict(image1.reshape(1, -1))[0]
        digit2 = clf.predict(image2.reshape(1, -1))[0]

        # Compare the predicted digits
        result = digit1 == digit2

        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
