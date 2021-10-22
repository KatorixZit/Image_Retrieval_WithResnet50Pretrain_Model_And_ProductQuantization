from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np

model = VGG16(include_top=False)


print(model.summary())