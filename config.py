import torch
import face_detection
import warnings
warnings.filterwarnings("ignore")

DETECTOR = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3, clip_boxes=True)

DEVICE = 'cuda:0' if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "./pix2pix_data/train/"
VAL_DIR = "./pix2pix_data/val/"
TEST_DIR = "./toon_pix_data/test/"
LEARNING_RATE = 5e-6
BETA1 = 0.5 
BATCH_SIZE = 256
NUM_WORKERS = 4
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L2_LAMBDA = 100
CE_LAMBDA = 1
IDENTITY_LAMBDA = 0
LAMBDA_GP = 10
NUM_EPOCHS = 300
LOAD_MODEL = False
SAVE_MODEL = True
SWA_START = int(NUM_EPOCHS/3)
CHECKPOINT_DISC = "disc.pt"
CHECKPOINT_GEN = "gen.pt"
