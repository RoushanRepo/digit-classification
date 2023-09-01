# Digit Classification with Support Vector Machines (SVM)

This Python code performs digit classification using Support Vector Machines (SVM) . It includes functions for splitting the data into training, development, and test sets, training the SVM model, and evaluating its performance.

## Prerequisites

Before running this code, make sure you have the required Python packages installed. You can install them using the `requirements.txt` file:


conda create --name ml-env python=3.10  # Create a virtual environment (if not already created)

conda activate ml-env  # Activate the virtual environment

pip install -r requirements.txt  # Install required packages


Usage
To use this code, follow these steps:

Activate your virtual environment:

conda activate ml-env

python digit_classification.py


The script will load the MNIST digits dataset, split it into training, development, and test sets, train an SVM classifier, and evaluate its performance. The results, including a classification report and confusion matrix, will be displayed in the terminal.

Additionally, the script will display images of the first four test samples along with their predicted labels.

Functions
split_train_dev_test(X, y, test_size, dev_size)
This function splits the data into training, development, and test sets.

X: Input data (features).
y: Target labels.
test_size: Proportion of data for the test set.
dev_size: Proportion of data for the development set.
predict_and_eval(model, X_test, y_test)
This function predicts labels using a trained model and evaluates its performance.

model: The trained machine learning model.
X_test: Input data for testing.
y_test: True labels for testing


Contact
For any questions or feedback, please contact m22aie243@iitj.ac.in