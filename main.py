import numpy as np 
import argparse
from typing import Dict, List, Tuple
import  tensorflow as tf
from tensorflow.keras import models

import utils

tf.enable_eager_execution()

STYLE_LAYERS = [
	"block1_conv1",
	"block2_conv1",
	"block3_conv1", 
	"block4_conv1", 
	"block5_conv1"
]

CONTENT_LAYERS = [
	"block5_conv2"
]


def get_intermediate_layers(layers: Dict) -> models.Model:
	vgg_model = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
	vgg_model.trainable = False # Freeze the weights

	style_layer_outputs = [vgg_model.get_layer(layer_name).output for layer_name in layers["style_layers"]]
	content_layer_outputs = [vgg_model.get_layer(layer_name).output for layer_name in layers["content_layers"]]
	all_outputs = style_layer_outputs + content_layer_outputs

	return models.Model(vgg_model.input, all_outputs)


def compute_style_loss(tranformed_image, target_content_image):
	return tf.reduce_mean(tf.square(tranformed_image - target_content_image))


def compute_content_loss(computed_style, targeted_style_gram):
	computed_style_gram = utils.compute_gram_matrix(computed_style)
	return tf.reduce_mean(tf.square(computed_style_gram - targeted_style_gram))


def transfer_style(content_image, style_image):
	model = get_intermediate_layers({
		"style_layers": STYLE_LAYERS,
		"content_layers": CONTENT_LAYERS
	})



def main(args):
	content_image = utils.load_image(args.content_image_path)
	style_image = utils.load_image(args.style_image_path)
	transfer_style(content_image, style_image)


if __name__ == "__main__":
	parser = argparse.ArgumentParser("Style Transfer")
	parser.add_argument("-l", "--learning-rate", type=int, default=0.05, help="learning rate")
	parser.add_argument("-s", "--style-image-path", type=str, default="./images/style/style.jpg", help="path to the style images")
	parser.add_argument("-c", "--content-image-path", type=str, default="./images/content/content.jpg", help="path to the content images")
	args = parser.parse_args()

	main(args)