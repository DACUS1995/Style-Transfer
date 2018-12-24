import numpy as np 
import argparse
from typing import Dict, List, Tuple
import  tensorflow as tf
from tensorflow.keras import models

import utils

tf.enable_eager_execution()


def get_intermediate_layers(layers: Dict) -> models.Model:
	vgg_model = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
	vgg_model.trainable = False # Freeze the weights

	style_layer_outputs = [vgg_model.get_layer(layer_name).output for layer_name in layers["style_layers"]]
	content_layer_outputs = [vgg_model.get_layer(layer_name).output for layer_name in layers["content_layers"]]
	all_outputs = style_layer_outputs + content_layer_outputs

	return models.Model(vgg_model.input, all_outputs)


def compute_style_loss():
	pass


def compute_content_loss():
	pass


def transfer_style(content_image, style_image):
	pass



def main(args):
	content_image = utils.load_image(args.content_image_path)
	style_image = utils.load_image(args.style_image_path)
	transfer_style(content_image, style_image)


if __name__ == "__main__":
	parser = argparse.ArgumentParser("Style Transfer")
	parser.add_argument("-l", "--learning-rate", type=int, default=0.05, help="learning rate")
	parser.add_argument("-s", "--style-image-path", type=str, default="./images/style/style.jpg", help="path to the style images")
	parser.add_argument("-c", "--content-image-path", type=str, default=".
	
	as/images/content/content.jpg", help="path to the content images")
	args = parser.parse_args()

	main(args)