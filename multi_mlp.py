# coding: utf-8

# ––––––––––––––––––––––––––––––––––––––––––––––––– IMPORT LIBRARIES –––––––––––––––––––––––––––––––––––––––––––––––– #

import subprocess
import sys
import numpy as np
import gc

# ––––––––––––––––––––––––––––––––––––––––––––––––– DEFINE FUNCTIONS –––––––––––––––––––––––––––––––––––––––––––––––– #

def calc_iters(start, end, step):
	return (end - start) / step

# ––––––––––––––––––––––––––––––––––––––––––––––––– GLOBAL VARIABLES –––––––––––––––––––––––––––––––––––––––––––––––– #

AUTO_LOG = 1

LR = 0.001
LR_start = 0.010
LR_end = 1.010
LR_step = 0.050
LR_arr = [LR_start, LR_end, LR_step]
LR_iters = calc_iters(LR_start, LR_end, LR_step)

DECAY = 0.0000001
DECAY_start = 0.0000000
DECAY_end = 0.0000010
DECAY_step = 0.0000001
DECAY_arr = [DECAY_start, DECAY_end, DECAY_step]
DECAY_iters = calc_iters(DECAY_start, DECAY_end, DECAY_step)

MOMENTUM = 0.9
MOMENTUM_start = 0.010
MOMENTUM_end = 1.010
MOMENTUM_step = 0.050
MOMENTUM_arr = [MOMENTUM_start, MOMENTUM_end, MOMENTUM_step]
MOMENTUM_iters = calc_iters(MOMENTUM_start, MOMENTUM_end, MOMENTUM_step)

NESTEROV = False
NESTEROV_arr = [False, True]
NESTEROV_iters = 2

EPOCHS = 50
EPOCHS_start = 1
EPOCHS_end = 101
EPOCHS_step = 10
EPOCHS_arr = [EPOCHS_start, EPOCHS_end, EPOCHS_step]
EPOCHS_iters = calc_iters(EPOCHS_start, EPOCHS_end, EPOCHS_step)

BATCH_SZ = 50
BATCH_SZ_start = 1
BATCH_SZ_end = 101
BATCH_SZ_step = 10
BATCH_SZ_arr = [BATCH_SZ_start, BATCH_SZ_end, BATCH_SZ_step]
BATCH_SZ_iters = calc_iters(BATCH_SZ_start, BATCH_SZ_end, BATCH_SZ_step)

HIDDEN_DIM = 50
HIDDEN_DIM_start = 1
HIDDEN_DIM_end = 101
HIDDEN_DIM_step = 10
HIDDEN_DIM_arr = [HIDDEN_DIM_start, HIDDEN_DIM_end, HIDDEN_DIM_step]
BATCH_SZ_iters = calc_iters(HIDDEN_DIM_start, HIDDEN_DIM_end, HIDDEN_DIM_step)

NUM_LAYERS = 3
NUM_LAYERS_start = 3
NUM_LAYERS_end = 103
NUM_LAYERS_step = 10
NUM_LAYERS_arr = [NUM_LAYERS_start, NUM_LAYERS_end, NUM_LAYERS_step]
NUM_LAYERS_iters = calc_iters(NUM_LAYERS_start, NUM_LAYERS_end, NUM_LAYERS_step)

TOTAL_iters = LR_iters * DECAY_iters * MOMENTUM_iters * NESTEROV_iters * EPOCHS_iters * BATCH_SZ_iters * BATCH_SZ_iters * NUM_LAYERS_iters

# –––––––––––––––––––––––––––––––––––––––––––––––––– MAIN FUNCTION –––––––––––––––––––––––––––––––––––––––––––––––––– #

def main():

	# ask to input custom values for start end and step
		# form arrays like LR_arr = [lr_start, lr_stop, lr_step]
		# for loops like: for i in np.arange(lr_start, lr_stop+lr_step, lr_step)
		# capture the total number of iterations for that parameter with equation: (end+step-start)/step
		# add to total that the print about run count will say: finished run count 4 of 2000

		# could add an array or True/False to determine which str(_val) values to use otherwise use the "<default_val>"

	run_count = 1
	for lr_val in np.arange(LR_arr[0], LR_arr[1], LR_arr[2]):
		for decay_val in np.arange(DECAY_arr[0], DECAY_arr[1], DECAY_arr[2]):
			for momentum_val in np.arange(MOMENTUM_arr[0], MOMENTUM_arr[1], MOMENTUM_arr[2]):
				for nesterov_val in NESTEROV_arr:
					for epochs_val in np.arange(EPOCHS_arr[0], EPOCHS_arr[1], EPOCHS_arr[2]):
						for batch_sz_val in np.arange(BATCH_SZ_arr[0], BATCH_SZ_arr[1], BATCH_SZ_arr[2]):
							for hidden_dim_val in np.arange(HIDDEN_DIM_arr[0], HIDDEN_DIM_arr[1], HIDDEN_DIM_arr[2]):
								for num_layers_val in np.arange(NUM_LAYERS_arr[0], NUM_LAYERS_arr[1], NUM_LAYERS_arr[2]):
									subprocess.run(["python3",
																	"mlp_mnist.py",
																	'1', # AUTO_LOG
																	str(lr_val), # LR
																	str(decay_val), # DECAY
																	str(momentum_val), # MOMENTUM
																	str(nesterov_val), # NESTEROV
																	str(epochs_val), # EPOCHS
																	str(batch_sz_val), # BATCH_SZ
																	str(hidden_dim_val), # HIDDEN_DIM
																	str(num_layers_val)], # NUM_LAYERS
																	capture_output=False, stdout=subprocess.DEVNULL)
									print("\nFinished run count", str(run_count), "out of", str(TOTAL_iters) + "\n", file=sys.stdout)
									run_count = run_count + 1
									if(run_count % 100 == 0): gc.collect()


# ––––––––––––––––––––––––––––––––––––––––––––––––––– MAIN GUARD –––––––––––––––––––––––––––––––––––––––––––––––––––– #

if __name__ == "__main__":
	main()

