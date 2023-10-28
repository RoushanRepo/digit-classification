import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, roc_curve, roc_auc_score

def main():
    # Load the digits dataset
    digits_data = datasets.load_digits()
    print("Digits dataset loaded.")

    # Split the data into features (X) and labels (y)
    X = digits_data.data
    y = digits_data.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a Decision Tree classifier
    decision_tree = DecisionTreeClassifier()
    print("Training Decision Tree model...")
    # Train the Decision Tree model
    decision_tree.fit(X_train, y_train)
    print("Decision Tree model trained.")
    # Make predictions
    y_dt_pred = decision_tree.predict(X_test)
    print("Decision Tree predictions made.")

    # Create an SVM classifier
    svm_classifier = SVC()
    print("Training SVM model...")
    # Train the SVM model
    svm_classifier.fit(X_train, y_train)
    print("SVM model trained.")
    # Make predictions
    y_svm_pred = svm_classifier.predict(X_test)
    print("SVM predictions made.")

    # Compare Decision Tree and SVM models
    print("Comparing models...")
    compare_models(y_test, y_dt_pred, "Decision Tree")
    compare_models(y_test, y_svm_pred, "SVM")

    print("Comparison completed.")

def compare_models(y_true, y_pred, model_name):
    results = []

    results.append(f"Results for {model_name}:")
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    results.append(f"Accuracy: {accuracy:.2f}")
    results.append(f"Precision: {precision:.2f}")
    results.append(f"Recall: {recall:.2f}")
    results.append(f"F1 Score: {f1:.2f}")

    # Classification report
    results.append("Classification Report:")
    results.append(classification_report(y_true, y_pred))

    # ROC Curve and AUC (for binary classification tasks)
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        results.append(f"ROC AUC: {auc:.2f}")

    results.append("===================================")

    # Print results to the console
    for line in results:
        print(line)

if __name__ == '__main__':
    main()
