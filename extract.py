import numpy as np
import gzip

def extract_data(filename, num_images, IMAGE_WIDTH):
	"""Extract the images into a 4D tensor [image index, y, x, channels].
	Values are rescaled from [0, 255] down to [-0.5, 0.5].
	"""
	print('Extracting', filename)
	with gzip.open(filename) as bytestream:
		bytestream.read(16)
		buf = bytestream.read(IMAGE_WIDTH * IMAGE_WIDTH * num_images)
		data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
		data = data.reshape(num_images, IMAGE_WIDTH*IMAGE_WIDTH)
		return data

def extract_labels(filename, num_images):
	"""Extract the labels into a vector of int64 label IDs."""
	print('Extracting', filename)
	with gzip.open(filename) as bytestream:
		bytestream.read(8)
		buf = bytestream.read(1 * num_images)
		labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
	return labels
