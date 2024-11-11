import numpy as np
from PIL import Image

data = np.load('training_set.npz', allow_pickle=True)
X = data['images']
y = data['labels']

for i in range(len(X)):
	im = Image.fromarray(X[i])
	im.save(str(i)+".jpg")
