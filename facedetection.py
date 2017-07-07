import numpy as np
import sys, cv2

def main(argv):
	inputfile = sys.argv[1]
	face_recognizer = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	eye_recognizer = cv2.CascadeClassifier('haarcascade_eye.xml')
	img = cv2.imread(inputfile)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_recognizer.detectMultiScale(gray, 1.3, 5)

	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = img[y:y+h, x:x+w]
		eyes = eye_recognizer.detectMultiScale(roi_gray)
		for (ex,ey,ew,eh) in eyes:
			cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	cv2.imshow('Output',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

if __name__ == "__main__":
	try:
		if sys.argv[1] == "-help":
			print "Usage: python facedetection.py inputimage.jpg\nFor a better result use an hires picture"
		else:
			main(sys.argv[1])
	except:
		print("Usage: python facedetection.py {<inputimg.jpg> | -help}")
		exit(1)