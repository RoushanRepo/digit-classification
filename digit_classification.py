import itertools
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from utils import hyperparameter_tuning, prepare_data_splits

def main():

    # Load the digits dataset
    digits_data = datasets.load_digits()

    # Flatten the images
    n_samples = len(digits_data.images)
    flattened_data = digits_data.images.reshape((n_samples, -1))
    X_data = flattened_data
    y_data = digits_data.target

    # Define parameter ranges
    gamma_values = [0.001, 0.01, 0.1, 1, 10, 100]
    C_values = [0.1, 1, 2, 5, 10, 20]
    all_param_combinations = list(itertools.product(gamma_values, C_values))

    # Define test and dev set sizes
    test_sizes = [0.1, 0.2, 0.3]
    dev_sizes = [0.1, 0.2, 0.3]
    size_combinations = list(itertools.product(test_sizes, dev_sizes))

    for test_frac, dev_frac in size_combinations:
        # Split the data into train, dev, and test sets
        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = prepare_data_splits(X_data, y_data, test_frac, dev_frac)

        # Tune hyperparameters
        trained_model, best_gamma, best_C, validation_metric, train_metric = hyperparameter_tuning(X_train, Y_train, X_dev, Y_dev, all_param_combinations)

        # Predict on the test set
        test_predictions = trained_model.predict(X_test)
        test_accuracy = metrics.accuracy_score(y_pred=test_predictions, y_true=Y_test)

        # Print results
        print(
            f'Training Size:{1 - (test_frac + dev_frac)} Test Size:{test_frac} Dev Size:{dev_frac} Train Acc:{train_metric} Test Acc:{test_accuracy} Val Acc:{validation_metric}')
        print(f'SVM model Metrics with gamma:{best_gamma} and C:{best_C}')


if __name__ == '__main__':
    main()
