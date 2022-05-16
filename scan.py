
from skimage.filters import threshold_local
import numpy as np
import argparse
import cv2
import imutils
from skimage.io import imsave


def edge_detection(image):
	ratio = image.shape[0] / 500.0
	original = image.copy()
	image = imutils.resize(image, height = 500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (5, 5), 0)
	edged = cv2.Canny(gray, 75, 200)
	return edged,ratio

def find_contours(image, edged):
	image = imutils.resize(image, height = 500)
	cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
	for c in cnts:
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		if len(approx) == 4:
			screenCnt = approx
			break

	cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
	return image,screenCnt

def perspective(original,screenCnt,ratio):
	from transform import four_point_transform
	warped = four_point_transform(original, screenCnt.reshape(4, 2) * ratio)
	warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
	T = threshold_local(warped, 11, offset = 10, method = "gaussian")
	warped = (warped > T).astype("uint8") * 255
	output=imutils.resize(warped, height = 650)
	return output
def download(image):
	cv2.imwrite('output.jpg',image)