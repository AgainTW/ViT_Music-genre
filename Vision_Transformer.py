import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras import layers, datasets, optimizers

class Config:
	# Origin image size
	input_size = [512, 128, 1]
	# Input shape of image
	input_shape = [input_size[0], input_size[1], 1]
	# Learning rate of the Model
	learning_rate = 0.001
	# Batch size of the Model
	batch_size = 32
	num_classes = 10
	num_epochs = 30
	image_size = [128, 32]
	patch_size = [16, 4]
	num_patches = (image_size[0]//patch_size[0])*(image_size[1]//patch_size[1])
	projection_dim = 128
	num_heads = 4
	transformer_units = [
		projection_dim * 2,
		projection_dim
	]
	transformer_layers = 8
	mlp_head_units = [2048, 1024]

def sample_images(images, row_count, column_count):
	fig, axs = plt.subplots(row_count, column_count, figsize=(10,10))
	for i in range(row_count):
		for j in range(column_count):
			axs[i,j].imshow(images[i * column_count + j])
			axs[i,j].axis('off')
	plt.show()

def mlp(x, hidden_units, dropout_rate):
	for units in hidden_units:
		x = layers.Dense(units, activation=tf.nn.gelu)(x)
		x = layers.Dropout(dropout_rate)(x)
	return x

# This Layer can convert image to N x N Grid.
class Patches(layers.Layer):
	def __init__(self, patch_size):
		super(Patches, self).__init__()
		self.patch_size = patch_size

	def call(self, images):
		batch_size = tf.shape(images)[0]
		patches = tf.image.extract_patches(
			images = images,
			sizes=[1, self.patch_size[0], self.patch_size[1], 1],
			strides=[1, self.patch_size[0], self.patch_size[1], 1],
			rates=[1, 1, 1, 1],
			padding="VALID",
			)
		patch_dims = patches.shape[-1]
		patches = tf.reshape(patches, [batch_size, -1, patch_dims])
		return patches

class PatchEncoder(layers.Layer):   
	def __init__(self, num_patches, projection_dim):
		super(PatchEncoder, self).__init__()
		self.num_patches = num_patches
		self.projection = layers.Dense(projection_dim)
		self.position_embedding = layers.Embedding(
			input_dim=num_patches, output_dim=projection_dim
			)
	def call(self, patch):
		positions = tf.range(start=0, limit=self.num_patches, delta=1)
		encoded = self.projection(patch) + self.position_embedding(positions)
		return encoded

augmentation_layer = tf.keras.Sequential([
    keras.layers.Input(Config.input_shape),
    keras.layers.experimental.preprocessing.Normalization(),
    keras.layers.experimental.preprocessing.Resizing(Config.image_size[0], Config.image_size[1]),
])

def create_vision_transformer():
	# Inputs
	inputs = layers.Input(shape=Config.input_shape)
	# Data norm
	norm = augmentation_layer(inputs)
	# Patches
	patches = Patches(Config.patch_size)(norm)
	encoder_patches = PatchEncoder(Config.num_patches, Config.projection_dim)(patches)

	for _ in range(Config.transformer_layers):
		# Layer Normalization 1
		x1 = layers.LayerNormalization(epsilon=1e-6)(encoder_patches)
		# Multi-Head Attention Layer
		attention_output = layers.MultiHeadAttention(
			num_heads=Config.num_heads, 
			key_dim=Config.projection_dim,
			dropout=0.1
			)(x1, x1)
		# Skip Connnection 1
		x2 = attention_output + encoder_patches

		# Layer Normalization 2
		x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

		# MLP
		x3 = mlp(x3, hidden_units=Config.transformer_units, dropout_rate=0.1)

		# Skip Connnection 2
		encoder_patches = x3 + x2

	representation = layers.LayerNormalization(epsilon=1e-6)(encoder_patches)
	representation = layers.Flatten()(representation)
	representation = layers.Dropout(0.5)(representation)

	features = mlp(representation, hidden_units=Config.mlp_head_units, dropout_rate=0.5)

	outputs = layers.Dense(Config.num_classes)(features)

	model = keras.Model(inputs=inputs, outputs=outputs)
	return model		

#if __name__ == '__main__':