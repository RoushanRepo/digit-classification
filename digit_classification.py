# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers, and performance metrics
from sklearn import datasets, metrics, svm
from sklearn.model_selection import train_test_split


# Function to split data into training, development, and test sets
def split_train_dev_test(X, y, test_size, dev_size):
    # First, split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Next, split the test set into development and test sets
    X_dev, X_test, y_dev, y_test = train_test_split(X_test, y_test, test_size=dev_size, random_state=42)

    return X_train, y_train, X_dev, y_dev, X_test, y_test


# Function to predict and evaluate a model
def predict_and_eval(model, X_test, y_test):
    # Predict the value of the digit on the test subset
    predicted = model.predict(X_test)

    # Print classification report
    print(f"Classification report for classifier {model}:\n{metrics.classification_report(y_test, predicted)}\n")

    # Display confusion matrix
    disp = metrics.ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}\n")

    # Rebuild classification report from confusion matrix
    y_true = []
    y_pred = []
    cm = disp.confusion_matrix
    for gt in range(len(cm)):
        for pred in range(len(cm)):
            y_true += [gt] * cm[gt][pred]
            y_pred += [pred] * cm[gt][pred]

    print("Classification report rebuilt from confusion matrix:\n")
    print(metrics.classification_report(y_true, y_pred))


# Load the digits dataset
digits = datasets.load_digits()

# Visualize a subset of the data
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title("Training: %i" % label)

# Flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.001)

# Split data into training, development, and test sets
X_train, y_train, X_dev, y_dev, X_test, y_test = split_train_dev_test(data, digits.target, test_size=0.2, dev_size=0.1)

# Learn the digits on the training subset
clf.fit(X_train, y_train)

# Predict and evaluate the model on the test subset
predict_and_eval(clf, X_test, y_test)

# Visualize the first 4 test samples and their predicted values
_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
predicted = clf.predict(X_test)  # Move this line here to define 'predicted'
for ax, image, prediction in zip(axes, X_test[:4], predicted[:4]):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction}")

plt.show()
