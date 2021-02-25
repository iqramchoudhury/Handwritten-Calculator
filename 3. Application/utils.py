import numpy as np
from PIL import Image
import keras
model = keras.models.load_model('Trained Model.h5')  

#Functions for the GUI

def find_image_center(img):
	"""Return tuple, coordinate center of img."""
	m = img / np.sum(np.sum(img))
	dx = np.sum(m, 0)
	dy = np.sum(m, 1)
	cx = int(round(np.sum(dx * np.arange(len(dx))), 0))
	cy = int(round(np.sum(dy * np.arange(len(dy))), 0))
	return cx, cy

def clean_equivalency_dict(d):
	"""Clean equivalency dict from connected component algorithm."""
	for key, val in d.items():
		d[key] = d[val]
	return d

def sort_by_other_list(this_list, by_this_list):
	"""Sort list based on another list."""
	result_list = []
	for cx, x in sorted(zip(by_this_list, this_list)):
		result_list.append(x)
	return result_list

# Functions for Translator

M_SIZE = 28     # is size of each sample on MNIST dataset 28x28
PADDING = 8     # number of empty pixel row / column (the smallest one)
MAP_SYMBOLS = {10: '/', 11: ')', 12: '-', 13: '*', 14: '(', 15: '+'}   

def thresholding(image, width, height):
	"""Convert image object to numpy array
	then thresholding each pixel to binary (grayscale)."""
	# image = Image.open(image)
	greyscale_image = image.convert('L')
	pixel_intensities = np.array(greyscale_image.getdata())
	pixel_array = pixel_intensities.reshape(height, width)
	thresh = np.vectorize(lambda x: 1 if x == 0 else 0)
	img = thresh(pixel_array)
	return img

def detect(image):
	"""Detect and seperate every single image of symbol 
	from image using row by row connected components algorithm.
	Return list of image and label.
	"""
	d = {0:0}
	idx = 0
	img = np.array(image)
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if img[i,j] != 0:
				top = img[i-1,j]
				left = img[i,j-1]
				if (left != 0 and (top == 0 or top == left)):
					img[i,j] = left
				elif ((left == 0 or left == top) and top != 0):
					img[i,j] = top
				elif left != 0 and top != 0 and left != top:
					img[i,j] = min(left, top)
					d[max(left,top)] = min(left,top)
				else:
					idx += 1
					img[i,j] = idx
					d[idx] = idx
	d = clean_equivalency_dict(d)
	map_d = np.vectorize(d.get)
	imgs = map_d(img)
	return imgs, d

def crop(img, label):
	"""Crop image array based on label.
	Return cropped image and it's horizontal center
	relative to original image."""
	thresh = np.vectorize(lambda x: 1 if x == label else 0)
	img = thresh(img)
	cx, cy = find_image_center(img)
	v, h = np.nonzero(img)
	img = img[min(v):max(v)+1, min(h):max(h)+1]
	return img, cx

def image_resize(img):
	"""Resize image keep aspect ratio to MNIST sample scale."""
	h, w = img.shape
	if h == max(h, w):
		h1 = M_SIZE - PADDING
		w1 = round((h1 / h) * w)		
	else:
		w1 = M_SIZE - PADDING
		h1 = round((w1 / w) * h)		
	# we use image for Image object, img for array object
	image = Image.fromarray(np.uint8(img) , 'L')
	image = image.resize((w1, h1))
	img = np.array(image)
	return img

def image_centering(img):
	"""Return M_SIZE x M_SIZE array of img on center with padding."""
	cx, cy = find_image_center(img)
	p = int(M_SIZE / 2)
	img = np.pad(img, ((p,p),(p,p)), 'constant')
	cx += p
	cy += p
	img = img[cy-p:cy+p, cx-p:cx+p]
	return img

def image_pad(img):
	padded_img = np.pad(img, ((2,2), (2,2)))
	return padded_img

def save_image_as_file(img):
	"""Save array as a .png for testing"""
	thresh = np.vectorize(lambda x: 0 if x == 1 else 255)
	wh_image = thresh(img)
	image = Image.fromarray(wh_image)
	image.save('Output Image.png')
	return wh_image

def predict(img):
	"""Run model.predict from keras model, and interpret it's symbol."""
	img = img.reshape(28,28,1)
	img = np.pad(img, ((2,2),(2,2),(0,0)))
	num = model.predict(img)
	if num >= 10:
		num = MAP_SYMBOLS[num]
	num = str(num)
	return num

def calculate(calculation):
	"""Return text contain calculation and answer for displaying to user."""
	try:
		result = eval(calculation)
		if calculation == str(result):
			result = ''
		else:
			if int(result) == result:
				result = int(result)
			else:
				result = np.round(result, 2)
			result = ' = ' + str(result)
	except Exception as e:
		result = ''
	result = ' ' + calculation + result
	return result