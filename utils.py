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
	image_copy = image.copy()
	image_copy[:, :, 0] += 103.939
	image_copy[:, :, 1] += 1116.779
	image_copy[:, :, 2] += 123.68
	image_copy = image_copy[:, :, ::-1] # VGG preprocess step tranforms RGB -> BGR
	image_copy = np.clip(image_copy, 0, 255)
	return image_copy
