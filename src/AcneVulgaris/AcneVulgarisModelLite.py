from __future__ import print_function, absolute_import, division,unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

class AcneVulgarisModel:
    """

    """

    def __init__(self, file: str, total_images: int):
        self.sep = ","
        self.eyes_dataframe = pd.read_csv(file, self.sep)
        self.eyes_dataframe = self.eyes_dataframe.reindex(
            np.random.permutation(self.eyes_dataframe.index))
        self.total_images = total_images
        self.export_directory="./export_dir"
        self.learning_rate = 0.03
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
            showPlot=True
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
            showPlot=True,
            load_from_file=True
        )

    def runTests(self, file: str):
        test_data = pd.read_csv(file, sep=",")

        test_examples = self.preprocess_features(test_data)
        test_targets = self.preprocess_targets(test_data)

        predict_test_input_fn = lambda: self.my_input_fn(
            test_examples,
            test_targets["Has_High_Blood_Pressure"],
            num_epochs=1,
            shuffle=False)

        test_predictions = self.linear_regressor.predict(input_fn=predict_test_input_fn)
        test_predictions = np.array(
            [item['predictions'][0] for item in test_predictions])


    def runTest(self, Percent_Red: float) -> float:
        test_data = pd.DataFrame({'Percent_Red': [Percent_Red], 'Has_High_Blood_Pressure': [0]})
        test_example = self.preprocess_features(test_data)
        test_targets = self.preprocess_targets(test_data)

        predict_test_input_fn = lambda: self.my_input_fn(
            test_example,
            test_targets["Has_High_Blood_Pressure"],
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

    def preprocess_targets(self, input_dataframe, target="Has_High_Blood_Pressure"):
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