# model config
# Model Config
# Using cuda or cpu for training
MODEL_DEVICE = "cuda"
# ID number of GPU
MODEL_DEVICE_ID = '0'
# Name of backbone
MODEL_NAME = 'resnet50'
# Last stride of backbone
MODEL_LAST_STRIDE = 2
# Path to pretrained model of backbone
MODEL_PRETRAIN_PATH = 'resnet50-0676ba61.pth'
# Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
# Options: 'imagenet' or 'self'
MODEL_PRETRAIN_CHOICE = 'imagenet'
# If train with BNNeck, options: 'bnneck' or 'no'
MODEL_NECK = 'no'
# If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
MODEL_IF_WITH_CENTER = 'no'
# The loss type of metric loss
# options:['triplet'](without center loss) or ['center','triplet_center'](with center loss)
MODEL_METRIC_LOSS_TYPE = 'triplet'
# For example, if loss type is cross entropy loss + triplet loss + center loss
# the setting should be: _C.MODEL.METRIC_LOSS_TYPE = 'triplet_center' and _C.MODEL.IF_WITH_CENTER = 'yes'
# If train with label smooth, options: 'on', 'off'
MODEL_IF_LABELSMOOTH = 'off'

# input config
# Size of the image during training
INPUT_SIZE_TRAIN = [256, 128]
# Size of the image during test
INPUT_SIZE_TEST = [256, 128]
# Random probability for image horizontal flip
INPUT_PROB = 0.5
# Random probability for random erasing
INPUT_RE_PROB = 0.0
# Values to be used for image normalization
INPUT_PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
INPUT_PIXEL_STD = [0.229, 0.224, 0.225]
# Value of padding size
INPUT_PADDING = 10

# dataset
# List of the dataset names for training, as present in paths_catalog.py market1501 cuhk03 dukemtmc msmt17
DATASETS_NAMES = ('dukemtmc')
# Root directory where datasets should be used (and downloaded if not found)
DATASETS_ROOT_DIR = ('.')

# dataloader
# Number of data loading threads
DATALOADER_NUM_WORKERS = 1
# Sampler for data loading
DATALOADER_SAMPLER = 'softmax'
# Number of instance for one batch(number of classes from images)
DATALOADER_NUM_INSTANCE = 16

# solver
# Name of optimizer
SOLVER_OPTIMIZER_NAME = "Adam"
# Number of max epoches
SOLVER_MAX_EPOCHS = 50
# Base learning rate
SOLVER_BASE_LR = 0.00035
# Factor of learning bias
SOLVER_BIAS_LR_FACTOR = 2
# Momentum
SOLVER_MOMENTUM = 0.9
# Margin of triplet loss
SOLVER_MARGIN = 0.3
# Margin of cluster ;pss
SOLVER_CLUSTER_MARGIN = 0.3
# Learning rate of SGD to learn the centers of center loss
SOLVER_CENTER_LR = 0.5
# Balanced weight of center loss
SOLVER_CENTER_LOSS_WEIGHT = 0.0005
# Settings of range loss
SOLVER_RANGE_K = 2
SOLVER_RANGE_MARGIN = 0.3
SOLVER_RANGE_ALPHA = 0
SOLVER_RANGE_BETA = 1
SOLVER_RANGE_LOSS_WEIGHT = 1

# Settings of weight decay
SOLVER_WEIGHT_DECAY = 0.0005
SOLVER_WEIGHT_DECAY_BIAS = 0.0005

# decay rate of learning rate
SOLVER_GAMMA = 0.1
# decay step of learning rate
SOLVER_STEPS = [40, 70]

# warm up factor
SOLVER_WARMUP_FACTOR = .01
# iterations of warm up
SOLVER_WARMUP_ITERS = 0
# method of warm up, option: 'constant','linear'
SOLVER_WARMUP_METHOD = "linear"

# epoch number of saving checkpoints
SOLVER_CHECKPOINT_PERIOD = 1
# iteration of display training log
SOLVER_LOG_PERIOD = 10
# epoch number of validation
SOLVER_EVAL_PERIOD = 50

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
SOLVER_IMS_PER_BATCH = 64

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
# Number of images per batch during test
TEST_IMS_PER_BATCH = 128
# If test with re-ranking, options: 'yes','no'
TEST_RE_RANKING = 'no'
# Path to trained model
TEST_WEIGHT = "."
# Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
TEST_NECK_FEAT = 'after'
# Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
TEST_FEAT_NORM = 'yes'

# misc options
# Path to checkpoint and saved log of trained model
OUTPUT_DIR = "."
