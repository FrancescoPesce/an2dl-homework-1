import os
import keras_cv
from tensorflow import keras as tfk

overlays = [f for f in os.listdir("keras_overlays")]

overlay_augmentation = keras_cv.layers.RandAugment(value_range=(0,255))

for i, over in enumerate(overlays):
    image = tfk.preprocessing.image.load_img("keras_overlays/"+over)
    image = tfk.preprocessing.image.img_to_array(image).astype('uint8')
    image = overlay_augmentation(image)
    tfk.preprocessing.image.save_img("keras_overlays_augmented/"+over, image)
