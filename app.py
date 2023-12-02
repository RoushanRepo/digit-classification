from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn import svm, tree, linear_model

app = Flask(__name__)

# Define paths for each model
svm_model_path = 'svm_model.pkl'
lr_model_path = 'm22aie243_lr_liblinear.joblib'
dt_model_path = 'decision_tree_model.pkl'

# Initialize variables to store loaded models
svm_model = None
lr_model = None
dt_model = None

# Function to load models
def load_models():
    global svm_model, lr_model, dt_model
    try:
        svm_model = joblib.load(svm_model_path)
        print("SVM Model loaded successfully.")
    except Exception as e:
        print(f"Error loading SVM model: {str(e)}")

    try:
        lr_model = joblib.load(lr_model_path)
        print("LR Model loaded successfully.")
    except Exception as e:
        print(f"Error loading LR model: {str(e)}")

    try:
        dt_model = joblib.load(dt_model_path)
        print("Decision Tree Model loaded successfully.")
    except Exception as e:
        print(f"Error loading Decision Tree model: {str(e)}")

# Load models when the application starts
load_models()

def compare_images(model, image1, image2):
    # Convert input strings to NumPy arrays
    image1_array = np.array(image1, dtype=float)
    image2_array = np.array(image2, dtype=float)

    # Flatten the arrays to 1D
    image1_array = image1_array.flatten()
    image2_array = image2_array.flatten()

    # Use the provided model to predict if the images are the same or different
    prediction = model.predict([image1_array, image2_array])

    return prediction[0]

@app.route('/predict/<model_type>', methods=['POST'])
def compare_images_endpoint(model_type):
    try:
        print("Request Received")
        data = request.get_json()

        image1 = data['image1']
        image2 = data['image2']

        # Determine which model to use based on the route
        if model_type == 'svm':
            model = svm_model
        elif model_type == 'lr':
            model = lr_model
        elif model_type == 'tree':
            model = dt_model
        else:
            return jsonify({'error': 'Invalid model type'})

        result = compare_images(model, image1, image2)

        response = {'result': 'Same' if result == 1 else 'Different'}
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/')
def home():
    return 'Hello, ML Model App!'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)
