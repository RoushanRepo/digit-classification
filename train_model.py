import itertools
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from utils import hyperparameter_tuning, prepare_data_splits

def main():
    # Load the digits dataset
    digits_data = datasets.load_digits()

    # Flatten the images
    n_samples = len(digits_data.images)
    flattened_data = digits_data.images.reshape((n_samples, -1))
    X_data = flattened_data
    y_data = digits_data.target

    # Apply unit normalization to the input data
    scaler = StandardScaler()
    X_data_normalized = scaler.fit_transform(X_data)

    # Define parameter ranges for Logistic Regression
    solver_values = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    C_values_lr = [0.1, 1, 2, 5, 10]
    all_lr_param_combinations = list(itertools.product(solver_values, C_values_lr))

    # Define test and dev set sizes
    test_sizes = [0.1, 0.2, 0.3]
    dev_sizes = [0.1, 0.2, 0.3]
    size_combinations = list(itertools.product(test_sizes, dev_sizes))

    rollno = "m22aie243"

    for test_frac, dev_frac in size_combinations:
        # Split the normalized data into train, dev, and test sets
        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = prepare_data_splits(X_data_normalized, y_data, test_frac, dev_frac)

        for solver, C in all_lr_param_combinations:
            model = linear_model.LogisticRegression(C=C, solver=solver)

            # Train the model
            model.fit(X_train, Y_train)

            # Evaluate the model performance
            accuracy = model.score(X_test, Y_test)
            print(f"Model performance for Logistic Regression with solver {solver} and C={C}: {accuracy}")

            # Save the model with the specified name format
            model_name = f'm22aie243_lr_{solver}.joblib'
            model_path = model_name
            joblib.dump(model, model_path)


            # Print model training logs to the console
            print(f"Model training complete for test_frac={test_frac}, dev_frac={dev_frac}, solver={solver}, C={C}")

if __name__ == "__main__":
    main()
