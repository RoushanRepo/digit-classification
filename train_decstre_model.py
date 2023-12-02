from sklearn import datasets, tree
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def train_and_save_decision_tree(X, y, model_path='decision_tree_model.pkl'):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create a Decision Tree model
    decision_tree_model = tree.DecisionTreeClassifier()

    # Train the model
    decision_tree_model.fit(X_train, y_train)

    # Evaluate the model on the test set (optional)
    accuracy = decision_tree_model.score(X_test, y_test)
    print(f"Decision Tree Model Accuracy: {accuracy}")

    # Save the trained model to a file
    joblib.dump(decision_tree_model, model_path)

if __name__ == "__main__":
    # Load or prepare your dataset
    # For example, using the digits dataset:
    digits_data = datasets.load_digits()
    X_data = digits_data.data
    y_data = digits_data.target

    # Train and save the Decision Tree model
    train_and_save_decision_tree(X_data, y_data, model_path='decision_tree_model.pkl')
