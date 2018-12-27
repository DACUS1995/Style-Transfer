import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image as kp_image
import tensorflow as tf
from matplotlib import pyplot as plt

def load_image(image_path: str, dimension=512) -> np.ndarray:
	image = Image.open(image_path)
	print(f'Loaded image {image_path} with size: [{image.size}]')
	
	max_dim = max(image.size)
	scale = dimension / max_dim
	
	image = image.resize((round(image.size[0] * scale), round(image.size[1] * scale)), Image.ANTIALIAS)
	image = kp_image.img_to_array(image)

	image = np.expand_dims(image, axis=0)
	return image

def display_image(image: np.ndarray) -> None:
	image = np.squeeze(image, axis=0)
	image = image.astype("uint8")
	plt.imshow(image)
	plt.show()

def preprocess_image(image: np.ndarray) -> np.ndarray:
	new_img = tf.keras.applications.vgg19.preprocess_input(image)
	return new_img

def prepare_image_visualization(image: np.ndarray) -> np.ndarray:
	if len(image.shape) == 4:
		image = np.squeeze(image, 0)
	image_copy = image.copy()
	image_copy[:, :, 0] += 103.939
	image_copy[:, :, 1] += 1116.779
	image_copy[:, :, 2] += 123.68
	image_copy = image_copy[:, :, ::-1] # VGG preprocess step tranforms RGB -> BGR
	image_copy = np.clip(image_copy, 0, 255)
	return image_copy

def compute_gram_matrix(feature_map: np.ndarray) -> np.ndarray:
	num_channels = int(feature_map.shape[-1])
	feature_map = tf.reshape(feature_map, (-1, num_channels))
	length = tf.shape(feature_map)[0]
	gram_matrix = tf.matmul(tf.transpose(feature_map), feature_map)
	gram_matrix = gram_matrix / tf.cast(length, tf.float32)
	return gram_matrix