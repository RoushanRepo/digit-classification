# Import necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load the MNIST dataset (example dataset)
digits = datasets.load_digits()
X = digits.data
y = digits.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the production model (SVM)
prod_model = SVC()
prod_model.fit(X_train, y_train)

# Train the candidate model (Decision Tree)
candidate_model = DecisionTreeClassifier()
candidate_model.fit(X_train, y_train)

# Make predictions using both models
prod_predictions = prod_model.predict(X_test)
candidate_predictions = candidate_model.predict(X_test)

# Calculate accuracies
prod_accuracy = accuracy_score(y_test, prod_predictions)
candidate_accuracy = accuracy_score(y_test, candidate_predictions)

# Calculate confusion matrices
confusion_matrix_all = confusion_matrix(y_test, prod_predictions, labels=range(10))
confusion_matrix_diff = confusion_matrix(y_test, [int(p1 != p2) for p1, p2 in zip(prod_predictions, candidate_predictions)])

# Calculate F1 scores
f1_macro_prod = f1_score(y_test, prod_predictions, average='macro')
f1_macro_candidate = f1_score(y_test, candidate_predictions, average='macro')

# Output the results
print("Production model's accuracy:", prod_accuracy)
print("Candidate model's accuracy:", candidate_accuracy)
print("Confusion matrix between production and candidate models:")
print(confusion_matrix_all)
print("Confusion matrix for samples predicted correctly in production but not in candidate:")
print(confusion_matrix_diff)
print("Production model's macro-average F1 score:", f1_macro_prod)
print("Candidate model's macro-average F1 score:", f1_macro_candidate)
