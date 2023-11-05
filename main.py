import util
import pickle
import Vision_Transformer as VT
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


loc = "C:/AG/course notes/111_2/多媒體內容分析/HW5/data/"
style_type = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
"""
# audio to train data
"""
if(0):
	for i in range(10):
		y = []
		first_x_flag = 0
		for j in range(50):
			read_loc = loc+style_type[i]+"/"+style_type[i]+"."+str(j).zfill(5)+".wav"
			rate, w = util.read_audio(read_loc)
			spec, sf = util.spectrogram(w, rate)

			batch = util.s2batch(spec)
			if(first_x_flag==0):
				first_x_flag = 1
				x = batch
			else:
				x = np.vstack((x, batch))

			for k in range(batch.shape[0]):
				y.append(i)
		y = np.array(y)

		print(style_type[i])
		print(y.shape)
		print(x.shape)

		with open('train_x_'+str(i)+'.npy', 'wb') as f:
			np.save(f, x)
		with open('train_y_'+str(i)+'.npy', 'wb') as f:
			np.save(f, y)

"""
# train
"""
if(1):
	# load train data
	first_flag = 0
	for i in range(10):
		with open('train_x_'+str(i)+'.npy', 'rb') as f:
			x = np.load(f)
		x = np.array(x, dtype="int32")	# 這個超重要
		with open('train_y_'+str(i)+'.npy', 'rb') as f:
			y = np.load(f)

		x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

		if(first_flag==0):
			first_flag = 1
			full_x_train = x_train
			full_x_test = x_test
			full_y_train = y_train
			full_y_test = y_test
		else:
			full_x_train = np.r_[full_x_train, x_train]
			full_x_test = np.r_[full_x_test, x_test]
			full_y_train = np.r_[full_y_train, y_train]
			full_y_test = np.r_[full_y_test, y_test]
	
	full_x_train, full_y_train = shuffle(full_x_train, full_y_train, random_state=42)
	

	print(full_x_train.shape)
	# reshape data,通常都啟用
	if(1):
		train = full_x_train.reshape((-1, VT.Config.input_size[0], VT.Config.input_size[1], 1))
		test = full_x_test.reshape((-1, VT.Config.input_size[0], VT.Config.input_size[1], 1))
		train_labels = full_y_train
		test_labels = full_y_test

	print(train.shape)
	# sample image and show,需要秀圖再開
	if(0):
		indices = np.random.choice(train.shape[0], 100)
		VT.sample_images(train[indices].squeeze(), 10, 10)

	# Patches 測試
	if(0):
		plt.figure(figsize=(4, 4))
		plt.imshow(np.squeeze(train[0]).astype("uint8"))
		plt.axis("off")
		plt.show()

		print(train.shape)
		patches = VT.Patches(VT.Config.patch_size)(train)
		print(f"Image size: {VT.Config.image_size[0]} X {VT.Config.image_size[1]}")
		print(f"Patch size: {VT.Config.patch_size[0]} X {VT.Config.patch_size[1]}")
		print(f"Patches per image: {patches.shape[1]}")
		print(f"Elements per patch: {patches.shape[-1]}")
		print(patches.shape)
		for i, patch in enumerate(patches[0]):
			ax = plt.subplot(VT.Config.image_size[1]//VT.Config.patch_size[1], 
				VT.Config.image_size[0]//VT.Config.patch_size[0], i+1)
			patch_img = tf.reshape(patch, (VT.Config.patch_size[0], VT.Config.patch_size[1]))
			plt.imshow(patch_img.numpy().astype("uint8"))
			plt.axis("off")
		plt.show()

	if(1):
		keras.backend.clear_session()
		vit_classifier = VT.create_vision_transformer()
		vit_classifier.summary()
		tf.keras.utils.plot_model(vit_classifier, to_file='model.png')

	if(1):
		vit_classifier.compile(
			loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			optimizer=tf.keras.optimizers.Adam(learning_rate=VT.Config.learning_rate),
			metrics=["accuracy"]
		)
		checkpoint_path = "model.h5"
		checkpoint = keras.callbacks.ModelCheckpoint(
		    checkpoint_path,
		    monitor="val_accuracy",
		    save_best_only=True,
		    save_weights_only=True
		)

		history = vit_classifier.fit(train, train_labels, epochs=VT.Config.num_epochs, 
			batch_size=VT.Config.batch_size, validation_data=(test, test_labels))

		# save
		vit_classifier.save("transformer_model.h5")

		with open('/trainHistoryDict', 'wb') as file_pi:
			pickle.dump(history.history, file_pi)

		# plot
		## Loss
		fig, ax = plt.subplots(figsize=(8,4))
		plt.title('loss')
		plt.plot(history.history['loss'], label='loss')
		plt.plot(history.history['val_loss'], label='val_loss', linestyle='--')
		plt.legend()
		plt.show()

		## Accuracy
		fig, ax = plt.subplots(figsize=(8,4))
		plt.title('accuracy')
		plt.plot(history.history['accuracy'], label='accuracy')
		plt.plot(history.history['val_accuracy'], label='val_accuracy', linestyle='--')
		plt.legend()
		plt.show()