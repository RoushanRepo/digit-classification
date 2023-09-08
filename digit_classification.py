import itertools
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from utils import hyperparameter_tuning, prepare_data_splits

def main():


    # Load the digits dataset
    digits = datasets.load_digits()

    # Flatten the images
    n_samples = len(digits.images)
    data = digits.images.reshape((n_samples, -1))
    X = data
    y = digits.target

    # Define parameter ranges
    gamma_ranges = [0.001, 0.01, 0.1, 1, 100]
    C_ranges = [0.1, 1, 2, 5, 10]
    list_of_all_param_combination = list(itertools.product(gamma_ranges, C_ranges))

    # Define test and dev set sizes
    test_size = [0.1, 0.2, 0.3]
    dev_size = [0.1, 0.2, 0.3]
    list_of_size = list(itertools.product(test_size, dev_size))

    for test_frac, dev_frac in list_of_size:
        # Split the data into train, dev, and test sets
        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = prepare_data_splits(X, y, test_frac, dev_frac)

        # Tune hyperparameters
        model, gamma, C, cur_metric, train_metric = hyperparameter_tuning(X_train, Y_train, X_dev, Y_dev, list_of_all_param_combination)

        # Predict on the test set
        pred_test_result = model.predict(X_test)
        test_metrix = metrics.accuracy_score(y_pred=pred_test_result, y_true=Y_test)

        # Print results
        print(
        f'Training Size:{1 - (test_frac + dev_frac)} Test Size:{test_frac} Dev Size:{dev_frac} Train Acc:{train_metric} Test Acc:{test_metrix} Val Acc:{cur_metric}')
        print(f' SVM model Metrix with gamma:{gamma} and C:{C}')


if __name__ == '__main__':
    main()
