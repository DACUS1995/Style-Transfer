import numpy as np 
import argparse

import  tensorflow as tf
from tensorflow.keras import models
import tensorflow.contrib.eager as tfe

from PIL import Image
import IPython.display
from matplotlib import pyplot as plt
from typing import Dict, Tuple, List
import time

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


def get_intermediate_layers_model(layers: Dict) -> models.Model:
	vgg_model = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
	vgg_model.trainable = False # Freeze the weights

	style_layer_outputs = [vgg_model.get_layer(layer_name).output for layer_name in layers["style_layers"]]
	content_layer_outputs = [vgg_model.get_layer(layer_name).output for layer_name in layers["content_layers"]]
	all_outputs = style_layer_outputs + content_layer_outputs

	new_model = models.Model(vgg_model.input, all_outputs)
	for layer in new_model.layers:
		layer.trainable = False
	return new_model


def compute_content_loss(generated_layer, target_content_layer):
	return tf.reduce_mean(tf.square(generated_layer - target_content_layer))


def compute_style_loss(generated_layer_style, targeted_style_gram):
	generated_layer_style_gram = utils.compute_gram_matrix(generated_layer_style)
	return tf.reduce_mean(tf.square(generated_layer_style_gram - targeted_style_gram))

def compute_total_loss(model, style_representations_gram_matrix, content_representations, generated_image, loss_weights) -> Tuple:
	generated_representations = model(generated_image)
	generated_content_representations = generated_representations[len(STYLE_LAYERS):]
	generated_style_representations = generated_representations[:len(STYLE_LAYERS)]

	style_loss = 0
	content_loss = 0
	style_weight, content_weight = loss_weights

	weight_style_layer = 1.0 / float(len(STYLE_LAYERS))
	for target_style, gen_style in zip(style_representations_gram_matrix, generated_style_representations):
		style_loss += weight_style_layer * compute_style_loss(gen_style[0], target_style)

	weight_content_layer = 1.0 / float(len(CONTENT_LAYERS))
	for target_content, gen_content in zip(content_representations, generated_content_representations):
		content_loss += weight_content_layer * compute_content_loss(gen_content[0], target_content)

	style_loss *= style_weight
	content_loss *= content_weight
	total_loss = style_loss + content_loss
	return total_loss, style_loss, content_loss

def compute_grads(model, style_representations_gram_matrix, content_representations, generated_image, loss_weights):
	with tf.GradientTape() as tape: 
		all_loss = compute_total_loss(
			model, 
			style_representations_gram_matrix, 
			content_representations, 
			generated_image, 
			loss_weights
		)

	total_loss = all_loss[0]
	return tape.gradient(total_loss, generated_image), all_loss


def transfer_style(content_image, style_image, learning_rate=5, content_weight=1e3, style_weight=1e-2, num_iterations=100):
	model = get_intermediate_layers_model({
		"style_layers": STYLE_LAYERS,
		"content_layers": CONTENT_LAYERS
	})

	# Get intermediate representations for the content image and the style image
	content_image = utils.preprocess_image(content_image)
	style_image = utils.preprocess_image(style_image)
	
	content_representations = model(content_image)
	style_representations = model(style_image)

	content_representations = [layer[0] for layer in content_representations[len(STYLE_LAYERS):]]
	style_representations_gram_matrix = [utils.compute_gram_matrix(layer[0]) for layer in style_representations[:len(STYLE_LAYERS)]]

	# Create the generated image
	generated_image = np.copy(content_image)
	generated_image = tfe.Variable(generated_image, dtype=tf.float32)
	
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.99, epsilon=1e-1)
	loss_weights = (style_weight, content_weight)
	best_loss, best_img = float('inf'), None

	# For displaying grid
	num_rows = 2
	num_cols = 5

	display_interval = num_iterations/(num_rows * num_cols)
	start_time = time.time()
	global_start = time.time()

	norm_means = np.array([103.939, 116.779, 123.68])
	min_vals = -norm_means
	max_vals = 255 - norm_means 

	imgs = []
	for i in range(num_iterations):
		grads, all_loss = compute_grads(
			model, 
			style_representations_gram_matrix, 
			content_representations, 
			generated_image, 
			loss_weights
		)

		loss, style_score, content_score = all_loss
		optimizer.apply_gradients([(grads, generated_image)])
		clipped = tf.clip_by_value(generated_image, min_vals, max_vals)
		generated_image.assign(clipped)

		# Save the image with the best score(smallest total loss)
		if loss < best_loss:
			best_loss = loss
			best_img = utils.prepare_image_visualization(generated_image.numpy())

		if i % display_interval == 0:
			start_time = time.time()

			plot_img = generated_image.numpy()
			plot_img = utils.prepare_image_visualization(plot_img)
			imgs.append(plot_img)
			IPython.display.clear_output(wait=True)
			IPython.display.display_png(Image.fromarray(plot_img))
			print(f'Iteration: {i}')      
			print(f'Total loss: {loss}, style loss: {style_score}, content loss: {content_score}, time: {time.time() - start_time}s')
	print(f'Total time: {time.time() - global_start}s')

	IPython.display.clear_output(wait=True)
	plt.figure(figsize=(14,4))
	for i,img in enumerate(imgs):
		plt.subplot(num_rows,num_cols,i+1)
		plt.imshow(img)
		plt.xticks([])
		plt.yticks([])

	return best_img, best_loss  


def main(args):
	content_image = utils.load_image(args.content_image_path)
	style_image = utils.load_image(args.style_image_path)
	best_img, best_loss = transfer_style(
		content_image, 
		style_image, 
		learning_rate=args.learning_rate, 
		num_iterations=args.iterations
	)

	img = Image.fromarray(best_img)
	img.save('my.png')
	IPython.display.display_png(img)


if __name__ == "__main__":
	parser = argparse.ArgumentParser("Style Transfer")
	parser.add_argument("-l", "--learning-rate", type=int, default=5, help="learning rate")
	parser.add_argument("-i", "--iterations", type=int, default=1000, help="number of iterations")
	parser.add_argument("-s", "--style-image-path", type=str, default="./images/style/style.jpg", help="path to the style images")
	parser.add_argument("-c", "--content-image-path", type=str, default="./images/content/content.jpg", help="path to the content images")
	args = parser.parse_args()

	main(args)