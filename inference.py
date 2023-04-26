import cv2
import torch
import face_detection
import numpy as np
import math
import face_detection
from models import Generator
from utils import load_checkpoint
import torchvision.transforms as transforms
import skimage
from utils import face_parsing, check_cords
from torchvision.utils import save_image

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

device = "cuda" if torch.cuda.is_available() else "cpu"

transform_test_data = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

Detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3, clip_boxes=True)

gen = Generator(in_channels=3)
file_path = f"./saved_models/min_train_loss_gen.pt"
    # file_path = f"./saved_models/min_train_loss_gen.pt"
gen = load_checkpoint(
    file_path,gen
)

gen = gen.to(device)

# Loading Image
image = skimage.io.imread('test_img.jpg')
y_shape, x_shape = image.shape[0], image.shape[1]

#Detect Face
detection = Detector.detect(image)
xmin, ymin, xmax, ymax, c = detection[0]
height = ymax-ymin
width = xmax-xmin
xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1 = int(xmin - width/6), int(ymin - height/4), int(xmax + width/6), int(ymax + width/15)
xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1 = check_cords(xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1, x_shape, y_shape)
face_image = image[ymin_mod_1:ymax_mod_1, xmin_mod_1:xmax_mod_1]

# get parsed face
face_image_512 = skimage.transform.resize(face_image, (512, 512), order=3, mode="reflect")
parsed_face_o = face_parsing(face_image)
parsed_face = transform_test_data(parsed_face_o)
parsed_face = torch.unsqueeze(parsed_face, 0)
parsed_face = parsed_face.to(device)
toon_gen = gen(parsed_face)
toon_gen = toon_gen * 0.5 + 0.5
save_image(toon_gen,"Saved_tensor_gen_toon.png")
toon_gen = toon_gen.permute((0,2,3,1))
toon_gen = toon_gen.cpu().detach().numpy()
toon_gen = toon_gen[0] * 255.0
toon_gen = np.array(toon_gen, dtype=np.uint8)
# skimage.io.imsave("parsed_toon.png",toon_gen)
toon_gen_resized = skimage.transform.resize(toon_gen, (512, 512), order=3, mode="reflect")
print("Shape of parsed toon : ",toon_gen_resized.shape)
print("Shape of parsed face : ",parsed_face_o.shape)
# face_toon = np.concatenate((parsed_face_o,toon_gen_resized), axis=1)
# skimage.io.imsave("parsed_face_toon.png",face_toon)
# skimage.io.imsave("parsed_toon.png",toon_gen_resized)
# print(np.max(parsed_face_o), np.min(parsed_face_o))
output = np.where(parsed_face_o==np.array([0, 0, 0]), face_image_512, toon_gen_resized)
output = output * 255.0
output = np.array(output, dtype=np.uint8)
# skimage.io.imsave("output.png", output)
image_height = int(ymax_mod_1 - ymin_mod_1)
image_width = int(xmax_mod_1 - xmin_mod_1)
output_resized = skimage.transform.resize(output, (image_height, image_width), order=3, mode="reflect")
output_resized = cv2.normalize(output_resized, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
# output_resized = cv2.cvtColor(output_resized, cv2.COLOR_BGR2RGB)
image[ymin_mod_1:ymax_mod_1, xmin_mod_1:xmax_mod_1] = output_resized
skimage.io.imsave("final_output.png", image)

# #Blur the edges
# blurred_image = cv2.GaussianBlur(image, (21, 21), 0)
# mask = np.zeros(image.shape, np.uint8)
# x_c = int((xmin_mod_1+xmax_mod_1)/2)
# y_c = int((ymin_mod_1+ymax_mod_1)/2)
# mask = cv2.ellipse(mask, (x_c, y_c), (int(width-width/3+50), int(height - height/3+50)),0,0,360,(255,255,255),-1)
# mask = cv2.ellipse(mask, (x_c, y_c), (int(width-width/3), int(height - height/3)),0,0,360,(0,0,0),-1)

# output_blur = np.where(mask==np.array([255, 255, 255]), blurred_image, image)

# skimage.io.imsave("final_output_blur.png", output_blur)

