import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('1.jpg')
grayscalee = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, threshold = cv2.threshold(grayscalee, 151, 255, cv2.THRESH_BINARY)

grayscalee = cv2.GaussianBlur(grayscalee, (5, 5), 0)

gaus = cv2.adaptiveThreshold(grayscalee, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 151, 6)

contours,_ = cv2.findContours(gaus, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:
	(x,y,w,h) = cv2.boundingRect(contour)
	cv2.rectangle(gaus, (x,y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('img', gaus)
cv2.waitKey(0)
cv2.destroyAllWindows()