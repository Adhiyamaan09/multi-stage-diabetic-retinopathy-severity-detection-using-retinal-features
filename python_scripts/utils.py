import io
import base64
import numpy as np
import cv2
from PIL import Image


def convert_cv2_image_to_PIL_image(arr):
    return Image.fromarray(arr)

def convert_PIL_image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format = "PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def convert_plt_figure_to_base64(fig):
	buf = io.BytesIO()
	fig.savefig(buf)
	buf.seek(0)
	img = Image.open(buf)
	img_str = convert_PIL_image_to_base64(img)
	return img_str


def getContours(img, thresh):
	mask1 = np.ones(img.shape, dtype = "uint8") * 255	
	cnts0, _ = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	for i in cnts0:
		if cv2.contourArea(i) <= thresh:
			cv2.drawContours(mask1, [i], -1, 0, 1)
	mask1 = cv2.bitwise_and(img, img, mask = mask1)
	return mask1
