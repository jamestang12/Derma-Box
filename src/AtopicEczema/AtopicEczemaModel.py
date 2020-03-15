from __future__ import print_function, absolute_import, division,unicode_literals

import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

class AtopicEczemaModel:
    """

    """

    def __init__(self, file: str, total_images: int):
        self.sep = ","
        self.eyes_dataframe = pd.read_csv(file, self.sep)
        self.eyes_dataframe = self.eyes_dataframe.reindex(
            np.random.permutation(self.eyes_dataframe.index))
        self.total_images = total_images
        self.export_directory="./export_dir"
        self.learning_rate = 0.05
        self.step_size = 150
        self.batch_size = 20

    def setup(self):
        # Training data
        self.training_examples = self.preprocess_features(
            self.eyes_dataframe.head(int(self.total_images * 0.7)))
        print(self.training_examples.describe())

        self.training_targets = self.preprocess_targets(
            self.eyes_dataframe.head(int(self.total_images * 0.7)))
        print(self.training_targets.describe())

        # Validation data
        self.validation_examples = self.preprocess_features(
            self.eyes_dataframe.tail(int(self.total_images * 0.3)))
        print(self.validation_examples.describe())

        self.validation_targets = self.preprocess_targets(
            self.eyes_dataframe.tail(int(self.total_images * 0.3)))
        print(self.validation_targets.describe())

    def plot(self):
        # Plot latitude and longitude
        """plt.figure(figsize=(13, 8))

        ax = plt.subplot(1, 2, 1)
        ax.set_title("Validation Data")

        ax.set_autoscaley_on(False)
        ax.set_ylim([0, 1])
        ax.set_autoscalex_on(False)
        ax.set_xlim([0, 1])
        print("-------------------------------------")
        print(self.validation_examples)
        print("-------------------------------------")
        plt.scatter(self.validation_examples["Is_Sick"],
                    self.validation_examples["Percent_Red"],
                    cmap="coolwarm",
                    c=self.validation_targets["Is_Sick"] / self.validation_targets["Is_Sick"].max())

        ax = plt.subplot(1,2,2)
        ax.set_title("Training Data")

        ax.set_autoscaley_on(False)
        ax.set_ylim([0, 1])
        ax.set_autoscalex_on(False)
        ax.set_xlim([0, 1])
        plt.scatter(self.training_examples["Is_Sick"],
                    self.training_examples["Percent_Red"],
                    cmap="coolwarm",
                    c=self.training_targets["Is_Sick"] / self.training_targets["Is_Sick"].max())
        _ = plt.plot()
        plt.show()"""

    def load_from_disk(self):
        # Create a linear regressor object.
        my_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)
        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,
                                                                   5.0)
        self.linear_regressor = tf.estimator.LinearRegressor(
            feature_columns=self.construct_feature_columns(self.training_examples),
            optimizer=my_optimizer,
            model_dir=self.export_directory
        )

    def train_new_model(self):
        self.linear_regressor = self.train_model(
            learning_rate=self.learning_rate,
            steps=self.step_size,
            batch_size=self.batch_size,
            training_examples=self.training_examples,
            training_targets=self.training_targets,
            validation_examples=self.validation_examples,
            validation_targets=self.validation_targets,
            showPlot=False
        )

    def train_current_model(self):
        self.linear_regressor = self.train_model(
            learning_rate=self.learning_rate,
            steps=self.step_size,
            batch_size=self.batch_size,
            training_examples=self.training_examples,
            training_targets=self.training_targets,
            validation_examples=self.validation_examples,
            validation_targets=self.validation_targets,
            showPlot=False,
            load_from_file=True
        )

    def runTests(self, file: str):
        test_data = pd.read_csv(file, sep=",")

        test_examples = self.preprocess_features(test_data)
        test_targets = self.preprocess_targets(test_data)

        predict_test_input_fn = lambda: self.my_input_fn(
            test_examples,
            test_targets["Is_Sick"],
            num_epochs=1,
            shuffle=False)

        test_predictions = self.linear_regressor.predict(input_fn=predict_test_input_fn)
        test_predictions = np.array(
            [item['predictions'][0] for item in test_predictions])

        root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(test_predictions, test_targets))

        print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)

    def runTest(self, Percent_Red: float) -> float:
        test_data = pd.DataFrame({'Percent_Red': [Percent_Red], 'Is_Sick': [0]})
        test_example = self.preprocess_features(test_data)
        test_targets = self.preprocess_targets(test_data)

        predict_test_input_fn = lambda: self.my_input_fn(
            test_example,
            test_targets["Is_Sick"],
            num_epochs=1,
            shuffle=False)

        test_predictions = self.linear_regressor.predict(input_fn=predict_test_input_fn)
        test_predictions = np.array(
            [item['predictions'][0] for item in test_predictions])
        return test_predictions[0]

    def runTestAndGetMessage(self, Percent_Red: float) -> str:
        value = self.runTest(Percent_Red)

        if value < 0.2:
            return "You do not have high blood pressure."
        elif value < 0.5:
            return "You have mild high blood pressure."

        return "You have very high blood pressure."

    def export_current_model(self):
        # export model and weights
        export_dir = self.linear_regressor.export_saved_model(export_dir_base="../export_dir",
                                                              serving_input_receiver_fn=self.serving_input_receiver_fn)

    def serving_input_receiver_fn(self):
        """
        input placeholder
        """
        inputs = {"Percent_Red": tf.placeholder(shape=[self.total_images], dtype=tf.float32)}
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    def preprocess_features(self, input_dataframe):
        """Prepares input features from California housing data set.

        Args:
          input_dataframe: A Pandas DataFrame expected to contain data
            from the data set.
        Returns:
          A DataFrame that contains the features to be used for the model, including
          synthetic features.
        """
        selected_features = input_dataframe[
            ["Percent_Red"]]
        processed_features = selected_features.copy()
        # Create a synthetic feature.
        # processed_features["rooms_per_person"] = (
        #   input_dataframe["total_rooms"] /
        #   input_dataframe["population"])
        return processed_features

    def preprocess_targets(self, input_dataframe, target="Is_Sick"):
        """Prepares target features (i.e., labels) from California housing data set.

        Args:
          input_dataframe: A Pandas DataFrame expected to contain data
            from the data set.
        Returns:
          A DataFrame that contains the target feature.
        """
        output_targets = pd.DataFrame()
        # Scale the target to be in units of thousands of dollars.
        output_targets[target] = (
            input_dataframe[target])
        return output_targets
        # return eye_dataframe

    def construct_feature_columns(self, input_features):
        """Construct the TensorFlow Feature Columns.

        Args:
          input_features: The names of the numerical input features to use.
        Returns:
          A set of feature columns
        """
        return set([tf.feature_column.numeric_column(my_feature)
                    for my_feature in input_features])

    def my_input_fn(self, features, targets, batch_size=1, shuffle=True,
                    num_epochs=None):
        """Trains a linear regression model of multiple features.

        Args:
          features: pandas DataFrame of features
          targets: pandas DataFrame of targets
          batch_size: Size of batches to be passed to the model
          shuffle: True or False. Whether to shuffle the data.
          num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
        Returns:
          Tuple of (features, labels) for next data batch
        """

        # Convert pandas data into a dict of np arrays.
        features = {key: np.array(value) for key, value in
                    dict(features).items()}

        # Construct a dataset, and configure batching/repeating.
        ds = Dataset.from_tensor_slices(
            (features, targets))  # warning: 2GB limit
        ds = ds.batch(batch_size).repeat(num_epochs)

        # Shuffle the data, if specified.
        if shuffle:
            ds = ds.shuffle(10000)

        # Return the next batch of data.
        features, labels = ds.make_one_shot_iterator().get_next()
        return features, labels

    def train_model(self,
            learning_rate,
            steps,
            batch_size,
            training_examples,
            training_targets,
            validation_examples,
            validation_targets,
            showPlot,
            loadFromFile=False):
        """Trains a linear regression model of multiple features.

        In addition to training, this function also prints training progress information,
        as well as a plot of the training and validation loss over time.

        Args:
          learning_rate: A `float`, the learning rate.
          steps: A non-zero `int`, the total number of training steps. A training step
            consists of a forward and backward pass using a single batch.
          batch_size: A non-zero `int`, the batch size.
          training_examples: A `DataFrame` containing one or more columns from
            `eye_dataframe` to use as input features for training.
          training_targets: A `DataFrame` containing exactly one column from
            `eye_dataframe` to use as target for training.
          validation_examples: A `DataFrame` containing one or more columns from
            `eye_dataframe` to use as input features for validation.
          validation_targets: A `DataFrame` containing exactly one column from
            `eye_dataframe` to use as target for validation.

        Returns:
          A `LinearRegressor` object trained on the training data.
        """

        periods = 3
        steps_per_period = steps / periods

        # Create a linear regressor object.
        my_optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
        my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,
                                                                   5.0)

        if loadFromFile:
            linear_regressor = tf.estimator.LinearRegressor(
                feature_columns=self.construct_feature_columns(training_examples),
                optimizer=my_optimizer,
                model_dir=self.export_directory
            )
        else:
            linear_regressor = tf.estimator.LinearRegressor(
                feature_columns=self.construct_feature_columns(training_examples),
                optimizer=my_optimizer
            )

        # 1. Create input functions.
        training_input_fn = lambda: self.my_input_fn(
            training_examples,
            training_targets["Is_Sick"],
            batch_size=batch_size)
        predict_training_input_fn = lambda: self.my_input_fn(
            training_examples,
            training_targets["Is_Sick"],
            num_epochs=1,
            shuffle=False)
        predict_validation_input_fn = lambda: self.my_input_fn(
            validation_examples, validation_targets["Is_Sick"],
            num_epochs=1,
            shuffle=False)

        # Train the model, but do so inside a loop so that we can periodically assess
        # loss metrics.
        print("Training model...")
        print("RMSE (on training data):")
        training_rmse = []
        validation_rmse = []
        for period in range(0, periods):
            # Train the model, starting from the prior state.
            linear_regressor.train(
                input_fn=training_input_fn,
                steps=steps_per_period,
            )
            # 2. Take a break and compute predictions.
            training_predictions = linear_regressor.predict(
                input_fn=predict_training_input_fn)
            training_predictions = np.array(
                [item['predictions'][0] for item in training_predictions])

            validation_predictions = linear_regressor.predict(
                input_fn=predict_validation_input_fn)
            validation_predictions = np.array(
                [item['predictions'][0] for item in validation_predictions])

            # Compute training and validation loss.
            training_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(training_predictions,
                                           training_targets))
            validation_root_mean_squared_error = math.sqrt(
                metrics.mean_squared_error(validation_predictions,
                                           validation_targets))
            # Occasionally print the current loss.
            print("  period %02d : %0.2f" % (
                period, training_root_mean_squared_error))
            # Add the loss metrics from this period to our list.
            training_rmse.append(training_root_mean_squared_error)
            validation_rmse.append(validation_root_mean_squared_error)
        print("Model training finished.")

        # Output a graph of loss metrics over periods.
        plt.subplot()
        plt.ylabel("RMSE")
        plt.xlabel("Periods")
        plt.title("Root Mean Squared Error vs. Periods")
        plt.tight_layout()
        plt.plot(training_rmse, label="training")
        plt.plot(validation_rmse, label="validation")
        plt.legend()

        if showPlot:
            plt.show()

        return linear_regressor