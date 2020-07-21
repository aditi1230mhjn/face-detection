import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0) # 0 for built in camera # 1 for external

while True:
	ret, frame = video_capture.read() # ret gets boolean value of frame capture
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
	gray,
	scaleFactor=1.1,
	minNeighbors = 5,
	minSize=(30,30) # 30*30 pixel face then only captured
	)
	
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2) # rectangle is formed around the face colour has rgb value from 0to 255, thickness in last parameter
		
	cv2.imshow('Video',frame)
	
	if cv2.waitKey(1) & 0xFF  == ord('q'):  # bitwise operator  ord takes the ascii value n act for escaping
		break

video_capture.release()
cv2.destroyAllWindows()
# rgb image has three channels
# gray is the name of image#1.1 means 10% lesser image#if 1.2 then 20% smaller