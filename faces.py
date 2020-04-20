import numpy as np
import cv2
import pickle


# Importing cascade into program
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#side_face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_profileface.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')

labels = {'person_name': 1}
with open('labels.pickle', 'rb') as f: #using pickle to save label id's
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

# taking main capture device
cap = cv2.VideoCapture(0)

#to make continuous usage of video
while True:
    ret, frame = cap.read()#reading frame by frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # color of frame #BGR = Blue Green Red
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    #profile = side_face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for(x, y, w, h) in faces:
        #print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w] # roi = region of interest
        roi_color = frame[y:y+h, x:x+w] #y cord start: y cord end, x cord start: x cord end

        #recognize/ deep learned model predict (keras, tensorflowm, pytorch, scikit learn)

        id_, conf = recognizer.predict(roi_gray)
        if conf>=45: #and conf<=85:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font , 1, color, stroke, cv2.LINE_AA)

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_color)

        color = (255, 0, 0) # in BGR (BLUE, GREEN, RED)
        stroke = 2 #width of stroke
        end_cord_x = x + w #width
        end_cord_y = y + h #height
        cv2.rectangle(frame,(x, y), (end_cord_x, end_cord_y), color, stroke)

    # for(x, y, w, h) in profile:
    #     print(x, y, w, h)
    #     roi_gray = gray[y:y+h, x:x+w] # roi = region of interest
    #     roi_color = frame[y:y+h, x:x+w] #y cord start: y cord end, x cord start: x cord end

    #     color = (255, 0, 0) # in BGR (BLUE, GREEN, RED)
    #     stroke = 2 #width of stroke
    #     end_cord_x = x + w #width
    #     end_cord_y = y + h #height
    #     cv2.rectangle(frame,(x, y), (end_cord_x, end_cord_y), color, stroke)
        
    cv2.imshow('frame', frame) # shows image
    if cv2.waitKey(20) & 0xFF == ord('q'): # key to break loop
        break


# When everything doen, release capture
cap.release()
cv2.destroyAllWindows()