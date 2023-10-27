import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # Load the digits dataset
    digits_data = datasets.load_digits()

    # Split the data into features (X) and labels (y)
    X = digits_data.data
    y = digits_data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Decision Tree classifier
    clf = DecisionTreeClassifier()

    # Train the model on the training data
    clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Generate a classification report
    report = classification_report(y_test, y_pred)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()
