import os
import cv2
import numpy as np

path = "uploads/"
files = os.listdir(path)

for file in files:
	img = cv2.imread(path+file)
	cv2.imwrite(path+file.split('.')[0]+'.jpg', img)
