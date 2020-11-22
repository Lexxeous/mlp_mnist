# coding: utf-8

# ––––––––––––––––––––––––––––––––––––––––––––––––– IMPORT LIBRARIES –––––––––––––––––––––––––––––––––––––––––––––––– #

AUTO_LOG = 1
LR = 0.001
DECAY = 0.0000001
MOMENTUM = 0.9
NESTEROV = False
EPOCHS = 50
BATCH_SZ = 50
HIDDEN_DIM = 50
NUM_LAYERS = 3

import subprocess
import sys

def main():
	m = [1, 2, 3]
	run_count = 1
	for M_val in m:
		subprocess.run(["python3",
										"mlp_mnist.py",
										'1',
										"0.001",
										"0.0000001",
										str(M_val),
										"False",
										"50",
										"50",
										"50",
										'3'], capture_output=False, stdout=subprocess.DEVNULL)
		print("Finished run count", run_count, "continuing...", file=sys.stdout)
	run_count = run_count + 1


# ––––––––––––––––––––––––––––––––––––––––––––––––––– MAIN GUARD –––––––––––––––––––––––––––––––––––––––––––––––––––– #

if __name__ == "__main__":
	main()