''' COMP6223 Computer Vision Coursework 1
Image Filtering and Hybrid Images
due: 15 Nov
author: Haoze Zhang
This module contains necessary tool to create a hybrid image by merging high
and low frequency components from different images.
Usage: $ python hybrid-image.py [low image] [high image] [sigma low] [sigma high]
e.g. $ python submarine.bmp fish.bmp 4.5 4.5
'''

import sys
import cv2 as cv
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt

def main():
	# testing
	if len(sys.argv) is 1:
		sys.argv += ("../data/fish.bmp", "../data/submarine.bmp", 4.5, 4.5)
	# wrong usage, usage prompt
	elif len(sys.argv) is not 5:
		print("Wrong argument number.")
		print("Usage: [low image] [high image] [sigma low] [sigma high]")
		print("e.g. $ python hybrid-image.py submarine.bmp fish.bmp 4.5 4.5")
		exit(1)

	lowImageDir = sys.argv[1]
	highImageDir = sys.argv[2]
	# not number, usage prompt
	try:
		sigmaLow = float(sys.argv[3])
		sigmaHigh = float(sys.argv[4])
	except Exception as e:
		print("Sigmas are not correct.")
		print("Usage: [low image] [high image] [sigma low] [sigma high]")
		print("e.g. $ python hybrid-image.py submarine.bmp fish.bmp 4.5 4.5")
		exit(1)
	low = cv.imread(lowImageDir)
	high = cv.imread(highImageDir)
	if any(img is None for img in (low, high)):
		print("Cannot find the specified image")
		exit(1)
	kernel1dl, kernel2dl = generateGaussianKernel(sigmaLow)
	kernel1dh, kernel2dh = generateGaussianKernel(sigmaHigh)
	low, high, new = hybrid(low, kernel2dl, high, kernel2dh)
	pyramid = createPyramid(new)

	showImage(new, "$\sigma_L={0},\ \sigma_H={1}$".format(sigmaHigh, sigmaLow))
	showImage(low, "Low frequency components $\sigma_L={}$".format(sigmaLow))
	showImage(high, "High frequency components $\sigma_H={}$".format(sigmaHigh))
	showImage(pyramid, "Pyramid")
	cv.imwrite("../result/L={},H={}.png".format(sigmaLow, sigmaHigh),new)

def convolve(image, kernel):
	''' Convolve the image with a kernel
	The dimension of the new image will be truncated by kernel/2 on each border.
	It works with a kernel of any odd dimension. Not necessary to be square.
	opencv alternative:
	newImage = cv.filter2D(image, -1, kernel)
	@param:
	image: numpy.array<y,x[,c]>
		gray scale or 3-channel BGR image
	kernel: numpy.array<y,x>
		convolution template matrix
	@return: numpy.array<y,x,c> uint8
		convoluted image
	'''
	kernelDimension = kernel.shape
	# treat gray and colour image equally by making the image a 3D array
	oldDimension = image.shape+(1,) if len(image.shape)==2 else image.shape
	image = image.reshape(image.shape+(1,)) if len(image.shape)==2 else image
	newDimension = (np.array(oldDimension)[:2]
	              - np.array(kernelDimension) + [1,1]).astype(int)
	newDimension = tuple(np.append(newDimension, oldDimension[2]))

	if any(k > o for (o, k) in zip(oldDimension, kernelDimension)):
		print("Kernel cannot be larger than the image")
		exit(1)
	if any(d%2 == 0 for d in kernelDimension):
		print("Kernel cannot have even dimensions: {}".format(kernelDimension))
		exit(1)

	# normalise the kernel
	kernel = kernel/kernel.sum()
	# create a blank new image
	newImage = np.ones(newDimension, dtype=np.uint8)
	# iterate over the old image
	for y in range(newDimension[0]):
		for x in range(newDimension[1]):
			# convolve over the kernel for each channel
			newImage[y,x] = [
				(image[y:y+kernelDimension[0],x:x+kernelDimension[1],i]
				* kernel).sum() for i in range(newDimension[2])
			]

			# piecewise alternative (unacceptably slow)
			'''
			partialSum = 0
			for yt in range(kernelDimension[0]):
				for xt in range(kernelDimension[1]):
					partialSum += kernel[yt,xt]*image[y+yt,x+xt]
			newImage[y,x] = partialSum
			'''
	return (newImage if newImage.shape[2]==3 else
	        newImage.reshape(newImage.shape[:2]))

def hybrid(lowImage, lowKernel, highImage, highKernel):
	''' Create the hybrid image from two images filtered by their kernels
	Images to be merged can be both coloured or gray. Images of different sizes
	will be truncated according to the smallest dimension. Kernel should be
	low-pass kernels.
	@param:
	image: numpy.array<y,x,c>
		images to be merged
	kernel: tuple<numpy.array<y,x>
		kernel used to filter the images
	@return: tuple<<numpy.array<y,x,c>>
		low frequency, high frequency, hybrid image
	'''
	# convolute the images
	lowImage = convolve(lowImage, lowKernel)
	minusLowImage = convolve(highImage, highKernel)
	offsety = int(highKernel.shape[0]/2)
	offsetx = int(highKernel.shape[1]/2)
	highImage = (highImage[offsety:-offsety,offsetx:-offsety].astype(np.int16)
	           - minusLowImage.astype(np.int16))
	# truncate images to the same size, aligned to the middle
	def centre(img, height, width):
		xs = (img.shape[0]-height) // 2
		xe = xs + height
		ys = (img.shape[1]-width) // 2
		ye = ys + width
		return img[xs:xe,ys:ye]
	height = min(lowImage.shape[0], highImage.shape[0])
	width = min(lowImage.shape[1], highImage.shape[1])
	lowImage = centre(lowImage, height, width)
	highImage = centre(highImage, height, width)

	# merge the images
	hybridImage = lowImage+highImage

	# normalise the images so they do not overflow and suitable to display
	# shift the values so there are no more negative
	if hybridImage.min() < 0:
		hybridImage -= hybridImage.min()
	if highImage.min() < 0:
		highImage -= highImage.min()
	# cast to opencv supported unsigned data type
	hybridImage = hybridImage.astype(np.uint16)
	highImage = highImage.astype(np.uint16)
	# normalize to 0-255 value range
	cv.normalize(highImage, highImage, 0, 255, cv.NORM_MINMAX, cv.CV_16UC1)
	cv.normalize(hybridImage, hybridImage, 0, 255, cv.NORM_MINMAX, cv.CV_16UC1)
	# cast to uint8 for display
	hybridImage = hybridImage.astype(np.uint8)
	highImage = highImage.astype(np.uint8)
	return (lowImage, highImage, hybridImage)

def generateGaussianKernel(sigma, size=None):
	''' Generate Gaussian kernel based on sigma
	#opencv alternative
	gaussian1d = cv.getGaussianKernel(size,sigma)
	gaussian2d = gaussian1d*gaussian1d.T
	'''
	if size is None:
		size = int(8.0*sigma + 1.0)
		size = size+1 if size%2==0 else size
	gaussian1d = signal.gaussian(size, std=sigma)[:,None]
	return (gaussian1d, gaussian1d*gaussian1d.T)

def showImage(img, title):
	''' show gray and coloured images '''
	if len(img.shape)==3:
		plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
	else:
		plt.imshow(img.reshape(img.shape[0:2]), cmap="gray")
	plt.title(title), plt.axis("off"), plt.show()

def createPyramid(img, number=4):
	''' Create n horizontal pyramid of the target image '''
	img = img.reshape((img.shape)+(1,)) if len(img.shape) == 2 else img
	pyramid = img
	height = img.shape[0]
	channel = img.shape[2]
	for i in range(number):
		img = cv.pyrDown(img)
		block = np.concatenate(
			(255*np.ones((height, 5, channel)), # horizontal padding
				np.concatenate( # vertical filling the blank
					(255*np.ones((height-img.shape[0], img.shape[1], channel)),
					img), 0 # vertical axis
				)
			), 1 # horizontal axis
		)
		# horizontally concatenate pyramid
		pyramid = np.concatenate((pyramid, block), 1)
	return pyramid.astype(np.uint8)



# low = cv.imread('../data/fish.bmp')
# high = cv.imread('../data/submarine.bmp')
# x = cv.getGaussianKernel(20,5)
# print(x)
# gaussianL = x*x.T
# x = cv.getGaussianKernel(20,5)
# gaussianH = x*x.T
# lowFiltered = cv.filter2D(low,-1,gaussianL)
# minusLowFiltered = cv.filter2D(high,-1,gaussianH)
# highFiltered = high-minusLowFiltered
# plt.imshow(cv.cvtColor(0.3*highFiltered+lowFiltered, cv.COLOR_BGR2RGB))
# plt.show()





if __name__ == '__main__':
	main()