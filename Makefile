AUTO_LOG?=1
LR?=0.001
DECAY?=0.0000001
MOMENTUM?=0.9
NESTEROV?=False
EPOCHS?=50
BATCH_SZ?=50
HIDDEN_DIM?=50
NUM_LAYERS?=3


run:
	python3 multi_mlp.py


single:
	python3 mlp_mnist.py $(AUTO_LOG) $(LR) $(DECAY) $(MOMENTUM) $(NESTEROV) $(EPOCHS) $(BATCH_SZ) $(HIDDEN_DIM) $(NUM_LAYERS)

clean:
	echo "Nothing to clean."