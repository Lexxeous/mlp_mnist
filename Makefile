# default is 1
AUTO_LOG?=1
# default is 0.001
LR?=0.001
# default is 0.0000001
DECAY?=0.0000001
# default is 0.9
MOMENTUM?=0.9
# default is False
NESTEROV?=False
# default is 50
EPOCHS?=50
# default is 50
BATCH_SZ?=50
# default is 50
HIDDEN_DIM?=50
# default is 3
NUM_LAYERS?=3


run:
	python3 multi_mlp.py

single:
	python3 mlp_mnist.py $(AUTO_LOG) $(LR) $(DECAY) $(MOMENTUM) $(NESTEROV) $(EPOCHS) $(BATCH_SZ) $(HIDDEN_DIM) $(NUM_LAYERS)

clean:
	echo "Nothing to clean."