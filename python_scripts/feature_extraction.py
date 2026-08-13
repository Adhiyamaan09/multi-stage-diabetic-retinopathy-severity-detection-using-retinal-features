import cv2
import numpy as np

from python_scripts.utils import convert_cv2_image_to_PIL_image, convert_PIL_image_to_base64, getContours


class BloodVessel():
	image = None

	def __init__(self, image):
		self.image = np.array(image)
	
	def extractGreenComponent(self):
		_, img_gc, _ = cv2.split(self.image)
		img_gc[img_gc <= 30] = 0
		self.image = img_gc
	
	def CLAHE(self, clipLimit = 2.0, tileGridSize = (3, 3)):
		clahe = cv2.createCLAHE(clipLimit = clipLimit, tileGridSize = tileGridSize)
		clImg = clahe.apply(self.image)

		self.image = clImg
	
	def alternateSequentialFilter(self):
		r1 = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations = 1)
		R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations = 1)
		r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations = 1)
		R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations = 1)
		r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations = 1)
		R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15 , 15)), iterations = 1)
		r4 = cv2.morphologyEx(R3, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)), iterations = 1)
		R4 = cv2.morphologyEx(r4, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11 , 11)), iterations = 1)
		
		img = cv2.subtract(R4, self.image)
		self.image = img
		self.CLAHE(clipLimit = 16.0, tileGridSize = (2, 2))
	
	def medianBlur(self):
		mdImg = cv2.medianBlur(cv2.GaussianBlur(self.image, (3,3), 0), 3)
		self.image = mdImg
	
	def applyThreshold(self):
		_, thresImg = cv2.threshold(self.image, 40, 255, cv2.THRESH_BINARY)
		self.image = thresImg
	
	def getContours1(self, img, thresh):
		mask1 = np.ones(img.shape, dtype = "uint8") * 255	
		cnts0, _ = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		for i in cnts0:
			if cv2.contourArea(i) <= thresh:
				cv2.drawContours(mask1, [i], -1, 0, 1)
		mask1 = cv2.bitwise_and(img, img, mask = mask1)
		return mask1

	def removeSmallObjects(self):
		img = getContours(self.image, 8)
		self.image = getContours(img, 1)

	def getImage(self):
		img = convert_cv2_image_to_PIL_image(self.image)
		self.image = convert_PIL_image_to_base64(img)
		return self.image
	
	def extractBloodVessels(self):
		self.extractGreenComponent()
		self.CLAHE(clipLimit = 2.0, tileGridSize = (3, 3))
		self.alternateSequentialFilter()
		self.medianBlur()
		self.applyThreshold()
		self.removeSmallObjects()

		return self.getImage()



class Exudates():
	image = None

	def __init__(self, image):
		self.image = np.array(image)
	
	def extractGreenComponent(self):
		gcImg = self.image[:, :, 1]
		self.image = gcImg
	
	def CLAHE(self):
		clahe = cv2.createCLAHE()
		clImg = clahe.apply(self.image)
		self.image = clImg
		
	def dilation(self):
		strEl = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
		dilateImg = cv2.dilate(self.image, strEl)
		self.image = dilateImg
	
	def applyThreshold(self):
		_, thresImg = cv2.threshold(self.image, 220, 255, cv2.THRESH_BINARY)
		self.image = thresImg
	
	def medianFilter(self):
		medImg = cv2.medianBlur(self.image, 5)
		self.image = medImg
		
	def getImage(self):
		img = convert_cv2_image_to_PIL_image(self.image)
		self.image = convert_PIL_image_to_base64(img)
		return self.image
	
	def extractExudates(self):
		self.extractGreenComponent()
		self.CLAHE()
		self.dilation()
		self.applyThreshold()
		self.medianFilter()

		return self.getImage()
