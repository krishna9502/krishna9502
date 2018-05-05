import numpy as np

def initialize_param(f, l):
	return 0.01*np.random.rand(l, f, f)

def initialize_theta(NUM_OUTPUT, l_in):
	return 0.01*np.random.rand(NUM_OUTPUT, int(l_in)) # python3.5 revision

def initialise_param_lecun_normal(FILTER_SIZE, IMG_DEPTH, scale=1.0, distribution='normal'):
	
    if scale <= 0.:
            raise ValueError('`scale` must be a positive float. Got:', scale)

    distribution = distribution.lower()
    if distribution not in {'normal'}:
        raise ValueError('Invalid `distribution` argument: '
                             'expected one of {"normal", "uniform"} '
                             'but got', distribution)

    scale = scale
    distribution = distribution
    fan_in = FILTER_SIZE*FILTER_SIZE*IMG_DEPTH
    scale = scale
    stddev = scale * np.sqrt(1./fan_in)
    shape = (IMG_DEPTH,FILTER_SIZE,FILTER_SIZE)
    return np.random.normal(loc = 0,scale = stddev,size = shape)

############################################################################### Hyperparameters
NUM_OUTPUT = 10
LEARNING_RATE = 0.01	#learning rate
IMG_WIDTH = 28
IMG_DEPTH = 1
FILTER_SIZE=5
NUM_FILT1 = 8
NUM_FILT2 = 8
BATCH_SIZE = 20
NUM_EPOCHS = 2	 # number of iterations
MU = 0.95

PICKLE_FILE = 'output.pickle'
# PICKLE_FILE = 'trained.pickle'

############################################################################### Initializing all the parameters
filt1 = {}
filt2 = {}
bias1 = {}
bias2 = {}

for i in range(0,NUM_FILT1):
	filt1[i] = initialise_param_lecun_normal(FILTER_SIZE, IMG_DEPTH, scale=1.0, distribution='normal')
	bias1[i] = 0.0
	# v1[i] = 0
for i in range(0,NUM_FILT2):
	filt2[i] = initialise_param_lecun_normal(FILTER_SIZE, NUM_FILT1, scale=1.0, distribution='normal')
	bias2[i] = 0.0
	# v2[i] = 0
w1 = IMG_WIDTH-FILTER_SIZE+1
w2 = w1-FILTER_SIZE+1
theta3 = initialize_theta(NUM_OUTPUT, (w2/2)*(w2/2)*NUM_FILT2)

bias3 = np.zeros((NUM_OUTPUT,1))
cost = []
acc = []

