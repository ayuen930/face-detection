import numpy as np
import cv2
import os

#ffmpeg to record video and sound

filename = 'video.mp4' #.avi , .mp4
frame_per_seconds = 24.0
my_res = '720p' # 1080p

# Set resolution for the video capture
def change_res(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

# Standard Video Dimensions Sizes
STD_DIMENSIONS =  {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "4k": (3840, 2160),
}
def get_dims(cap, res='1080'):
    width, height = STD_DIMENSIONS['480p'] #have some sort relation incase it doesnt work
    if res in STD_DIMENSIONS:
        width, height = STD_DIMENSIONS[res]
    change_res(cap, width,height)
    return width, height

# Video Encoding, might require additional installs
VIDEO_TYPE = {
    'avi': cv2.VideoWriter_fourcc(*'XVID'),
    'mp4': cv2.VideoWriter_fourcc(*'XVID'),
}

def get_video_type(filename):
    filename, ext = os.path.splitext(filename)
    if ext in VIDEO_TYPE:
      return  VIDEO_TYPE[ext]
    return VIDEO_TYPE['avi']

cap = cv2.VideoCapture(0)
dims = get_dims(cap, res=my_res)
video_type_cv2 = get_video_type(filename)

out = cv2.VideoWriter(filename, video_type_cv2, frame_per_seconds, dims) # dims can be just width, height
while True:
    # Capture frame
    ret, frame = cap.read()
    out.write(frame)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
            break


# When everything doen, release capture
cap.release()
out.release()
cv2.destroyAllWindows()
