import os
import keras_cv
import tensorflow as tf
from tensorflow import keras as tfk

overlays = [f for f in os.listdir("images")]

overlay_augmentation = keras_cv.layers.Augmenter(
    [
        keras_cv.layers.RandomFlip(),
        keras_cv.layers.RandAugment(value_range=(0, 255)),
    ],
)

for i, over in enumerate(overlays):
    if i % 100 == 0:
        print(i)
    image = tfk.preprocessing.image.load_img("images/"+over)
    image = tfk.preprocessing.image.img_to_array(image).astype('uint8')
    image = overlay_augmentation(image)
    tfk.preprocessing.image.save_img("images_augmented/"+over, image)