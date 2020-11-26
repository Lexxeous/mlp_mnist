# coding: utf-8

# ––––––––––––––––––––––––––––––––––––––––––––––––– IMPORT LIBRARIES –––––––––––––––––––––––––––––––––––––––––––––––– #

import tensorflow as tf
import numpy as np
import os
import sys
import struct
import keras

# ––––––––––––––––––––––––––––––––––––––––––––––––– GLOBAL VARIABLES –––––––––––––––––––––––––––––––––––––––––––––––– #

AUTO_LOG = int(sys.argv[1])
LR = float(sys.argv[2])
DECAY = float(sys.argv[3])
MOMENTUM = float(sys.argv[4])
NESTEROV = sys.argv[5]
EPOCHS = int(sys.argv[6])
BATCH_SZ = int(sys.argv[7])
HIDDEN_DIM = int(sys.argv[8])
NUM_LAYERS = int(sys.argv[9])

MIN_LAYERS = 3 # don't change this unless you have to

# ––––––––––––––––––––––––––––––––––––––––––––––––– DEFINE FUNCTIONS –––––––––––––––––––––––––––––––––––––––––––––––– #


def load_mnist(path, kind='train'):
	"""
	Load MNIST data from `path`

	Run decompression command before this script:
		"cd datasets"
		"gzip -d *ubyte.gz"
	"""

	labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte')
	images_path = os.path.join(path, f'{kind}-images-idx3-ubyte')

	with open(labels_path, 'rb') as lbpath:
		magic, n = struct.unpack('>II', lbpath.read(8))
		labels = np.fromfile(lbpath, dtype=np.uint8)

	with open(images_path, 'rb') as imgpath:
		magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
		images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
		images = ((images / 255.) - .5) * 2

	return images, labels

# –––––––––––––––––––––––––––––––––––––––––––––––––– MAIN FUNCTION –––––––––––––––––––––––––––––––––––––––––––––––––– #

def main():
	tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # suppress tensorflow warnings about depreciations

	# Loading the datasets
	X_train, y_train = load_mnist('./datasets/', kind='train')
	print(f'Rows: {X_train.shape[0]},  Columns: {X_train.shape[1]}')

	X_test, y_test = load_mnist('./datasets/', kind='test')
	print(f'Rows: {X_test.shape[0]},  Columns: {X_test.shape[1]}')

	print()

	# Mean centering and normalization:
	mean_vals = np.mean(X_train, axis=0)
	std_val = np.std(X_train)

	X_train_centered = (X_train - mean_vals)/std_val
	X_test_centered = (X_test - mean_vals)/std_val

	del X_train, X_test

	print(X_train_centered.shape, y_train.shape)
	print(X_test_centered.shape, y_test.shape)

	# –––––––––––––––––––––––––––––– OPTIONAL PLOTTING ––––––––––––––––––––––––––––––

	# Show plot of some digit data
	# fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
	# ax = ax.flatten()
	# for i in range(10):
	#     img = X_train_centered[y_train == i][0].reshape(28, 28)
	#     ax[i].imshow(img, cmap='Greys')

	# ax[0].set_yticks([])
	# ax[0].set_xticks([])
	# plt.tight_layout()
	# plt.show()

	# –––––––––––––––––––––––––––––– OPTIONAL SEEDING ––––––––––––––––––––––––––––––

	# rand_int = random.randint(0, 1e6)
	# np.random.seed(123)
	# tf.random.set_seed(123)
	# tf.set_random_seed(123)


	y_train_onehot = keras.utils.to_categorical(y_train)

	print('First 3 labels: ', y_train[:3])
	print('\nFirst 3 labels (one-hot):\n', y_train_onehot[:3])
	print()

	# Initialize model
	model = keras.models.Sequential()

	# Add input layer
	model.add(keras.layers.Dense(
		input_dim=X_train_centered.shape[1],
		units=HIDDEN_DIM, # output dimension
		kernel_initializer='glorot_uniform',
		bias_initializer='zeros',
		activation='tanh')
	)

	# add hidden layer
	model.add(
		keras.layers.Dense(
			input_dim=HIDDEN_DIM,
			units=HIDDEN_DIM, # output dimension
			kernel_initializer='glorot_uniform',
			bias_initializer='zeros',
			activation='tanh')
		)

	for layer in range(MIN_LAYERS, int(sys.argv[9])):
		# add hidden layer
		model.add(
		keras.layers.Dense(
			input_dim=HIDDEN_DIM,
			units=HIDDEN_DIM, # output dimension
			kernel_initializer='glorot_uniform',
			bias_initializer='zeros',
			activation='tanh')
		)

	# add output layer
	model.add(
		keras.layers.Dense(
			input_dim=HIDDEN_DIM,
			units=y_train_onehot.shape[1], # output dimension
			kernel_initializer='glorot_uniform',
			bias_initializer='zeros',
			activation='softmax')
		)

	print(model.summary())
	param_count = model.count_params()

	NUM_LAYERS = len(model.layers)

	# define SGD optimizer
	sgd_optimizer = keras.optimizers.SGD(lr=LR, decay=DECAY, momentum=MOMENTUM, nesterov=NESTEROV)

	# compile model
	model.compile(optimizer=sgd_optimizer,loss='categorical_crossentropy')

	# train model
	history = model.fit(
		X_train_centered, y_train_onehot,
		batch_size=BATCH_SZ, epochs=EPOCHS,
		verbose=1, validation_split=0.1
	)

	y_train_pred = model.predict_classes(X_train_centered, verbose=0)

	# calculate training accuracy
	y_train_pred = model.predict_classes(X_train_centered, verbose=0)
	correct_preds = np.sum(y_train == y_train_pred, axis=0)
	train_acc = correct_preds / y_train.shape[0]

	print(f'Training accuracy: {(train_acc * 100):.2f}')

	# calculate testing accuracy
	y_test_pred = model.predict_classes(X_test_centered, verbose=0)
	correct_preds = np.sum(y_test == y_test_pred, axis=0)
	test_acc = correct_preds / y_test.shape[0]

	print(f'Test accuracy: {(test_acc * 100):.2f}')

	if(AUTO_LOG):
		with open("results.csv", 'a') as res:
			print(f"{str(round(train_acc, 4))},{str(round(test_acc, 4))},{LR},{DECAY},{MOMENTUM},{NESTEROV},{EPOCHS},{BATCH_SZ},{HIDDEN_DIM},{NUM_LAYERS},{param_count}", file=res)

# ––––––––––––––––––––––––––––––––––––––––––––––––––– MAIN GUARD –––––––––––––––––––––––––––––––––––––––––––––––––––– #

if __name__ == "__main__":
	main()
