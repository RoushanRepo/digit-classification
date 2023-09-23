import itertools
import matplotlib.pyplot as plt
from sklearn import datasets, metrics, svm
from utils import hyperparameter_tuning, prepare_data_splits
import numpy as np  # Import NumPy

def main():
    # Load the digits dataset
    digits_data = datasets.load_digits()

    # Define test and dev set sizes
    test_size = 0.2
    dev_size = 0.1

    # Iterate over different image sizes
    for image_size in [4, 6, 8]:
        # Resize the images
        n_samples = len(digits_data.images)
        resized_data = np.array([np.resize(image, (image_size, image_size)) for image in digits_data.images])
        X_data = resized_data.reshape((n_samples, -1))
        y_data = digits_data.target

        # Define parameter ranges
        gamma_values = [0.001, 0.01, 0.1, 1, 100]
        C_values = [0.1, 1, 2, 5, 10]
        all_param_combinations = list(itertools.product(gamma_values, C_values))

        # Split the data into train, dev, and test sets
        X_train, Y_train, X_dev, Y_dev, X_test, Y_test = prepare_data_splits(X_data, y_data, test_size, dev_size)

        # Tune hyperparameters
        trained_model, best_gamma, best_C, validation_metric, train_metric = hyperparameter_tuning(X_train, Y_train, X_dev, Y_dev, all_param_combinations)

        # Predict on the test set
        test_predictions = trained_model.predict(X_test)
        test_accuracy = metrics.accuracy_score(y_pred=test_predictions, y_true=Y_test)

        # Print results
        print(f'image size: {image_size}x{image_size} train_size: {1 - (test_size + dev_size):.1f} dev_size: {dev_size} test_size: {test_size} '
              f'train_acc: {train_metric:.4f} dev_acc: {validation_metric:.4f} test_acc: {test_accuracy:.4f}')

if __name__ == '__main__':
    main()
