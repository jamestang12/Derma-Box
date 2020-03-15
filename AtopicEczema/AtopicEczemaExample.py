from __future__ import print_function, absolute_import, division,unicode_literals

import pandas as pd
import tensorflow as tf

from AtopicEczema.AtopicEczemaModel import AtopicEczemaModel

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

# Total images in dataset
total_images = 101
file = "filtered_image_data.csv"

# Do these calls in THIS order!!!!!!!!!!!!!!!
# Creating our model:
model = AtopicEczemaModel(file, total_images)
model.setup()

# -- TRAIN THE MODEL FROM SCRATCH --
model.train_new_model()
model.runTests(file)
print(model.runTest(0.2))
model.export_current_model()

# -- RUN A SINGLE TEST ON THE MODEL --
# model.load_from_disk()
# print(model.runTest(0.2))
# or: print(model.runTest(getPercentRed('camera_output.jpg')))

# -- MORE RANDOM THINGS --
# Broken:
# model.plot()

# You can either train the model from the saved one on disk or from scratch...
# model.train_model()
# model.train_new_model()
# ... or you can simply load the model from disk.
# model.load_from_disk()

# Then you can run tests from the data set...
# model.runTests(file)
# ... or you can ask for a prediction like so.
# You must input the amount of the image that has ulcers on it (0 - 1).
# print(model.runTest(0.2))
# model.export_current_model()
# model.createTFLiteModel(0.2)