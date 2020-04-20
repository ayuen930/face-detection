import numpy as np
import cv2

# taking main capture device
cap = cv2.VideoCapture(0)

def make_1080p():
    cap.set(3, 1920) # (width, pixle)
    cap.set(4, 1080) # (length, pixle) # 3 x 4 , 1920 x 1080

def make_720p():
    cap.set(3, 1280)
    cap.set(4, 720)

def make_480p():
    cap.set(3, 640)
    cap.set(4, 480)

def change_res(width, height):
    cap.set(3, width)
    cap.set(4, height)

#re-scaling frame/re-sized frame
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

#to make continuous usage of video
while True:
    ret, frame = cap.read()#reading frame by frame
#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # color of frame #BGR = Blue Green Red
    frame = rescale_frame(frame, percent=30)
    cv2.imshow('frame', frame) # shows image
#   cv2.imshow('gray', gray) # shows grat image
    if cv2.waitKey(20) & 0xFF == ord('q'): # key to break loop
            break


# When everything doen, release capture
cap.release()
cv2.destroyAllWindows()