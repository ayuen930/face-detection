import numpy as np
import cv2

# taking main capture device
cap = cv2.VideoCapture(0)

#to make continuous usage of video
while True:
    ret, frame = cap.read()#reading frame by frame

#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # color of frame #BGR = Blue Green Red

    cv2.imshow('frame', frame) # shows image
#    cv2.imshow('gray', gray) # shows grat image
    if cv2.waitKey(20) & 0xFF == ord('q'): # key to break loop
            break


# When everything doen, release capture
cap.release()
cv2.destroyAllWindows()