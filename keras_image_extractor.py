from tensorflow import keras as tfk
import random
import matplotlib.pyplot as plt
random.seed(11037)

(x_train, y_train), (x_test, y_test) = tfk.datasets.cifar100.load_data(label_mode="fine")

num_images = 11959
images = x_train[:num_images]

images = tfk.layers.Resizing(96,96)(images)

"""
fig, axs = plt.subplots(1, num_images, figsize=(50, 50))

for i in range(num_images):
    axs[i].imshow(images[i])
    axs[i].axis('off')

plt.show()
"""

for i, image in enumerate(images):
    tfk.preprocessing.image.save_img("keras_overlays/"+str(i)+".png", image)