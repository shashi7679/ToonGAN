import cv2
import face_detection
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
import ntpath

print(face_detection.available_detectors)
detector = face_detection.build_detector(
  "DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)

image_folder = "./toon_pix_data/train/original"
toon_folder="./toon_pix_data/train/toon"
i=0
for image in glob.glob(f"{image_folder}/*.png"):

  img = cv2.imread(image)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  detections = detector.detect(img)
  xmin, ymin, xmax, ymax, c = detections[0]
  xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
  height = int(ymax-ymin)
  width = int(xmax-xmin)
  xmin_mod_1, ymin_mod_1, xmax_mod_1, ymax_mod_1 = int(xmin - width/6), int(ymin - height/4), int(xmax + width/6), int(ymax + width/15)
  # cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,0,0))
  cv2.rectangle(img, (xmin_mod_1, ymin_mod_1), (xmax_mod_1, ymax_mod_1), (255,0,0),2)

  with Image.open(image) as im:

    x=ntpath.basename(image)
    for image1 in glob.glob(f"{toon_folder}/*.png"):
      img4 = cv2.imread(image1)
      img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
      if(ntpath.basename(image1)==x):
        with Image.open(image1) as im1:
          img3=im1.crop((xmin_mod_1,ymin_mod_1,xmax_mod_1, ymax_mod_1))
          newsize = (256, 256)
          img3 = img3.resize(newsize)
          img3.save("./cropped_images/train/toon/"+x)
        break

    img2=im.crop((xmin_mod_1,ymin_mod_1,xmax_mod_1, ymax_mod_1))
    newsize = (256, 256)
    img2 = img2.resize(newsize)

    img2.save("./cropped_images/train/original/"+x)
    i=i+1
    print(i)
