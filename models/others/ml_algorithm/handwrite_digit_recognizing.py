# -*- coding: utf-8 -*-


# Standard scientific Python imports
import matplotlib.pyplot as plot

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

# The digits dataset
digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]): # no.3
    plot.subplot(2, 4, index + 1)
    plot.axis('off')
    plot.imshow(image, cmap=plot.cm.gray_r, interpolation='nearest')
    plot.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
