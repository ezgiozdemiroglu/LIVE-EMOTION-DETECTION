from cgitb import text
from unittest import result
import cv2
import os

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential, model_from_json



json_file = open('cnn.json', 'r')
loaded_cnn_json = json_file.read()
json_file.close()
loaded_cnn = model_from_json(loaded_cnn_json)
# load weights into new model
loaded_cnn.load_weights("cnn.h5")
print("Loaded cnn from disk")




cascPath = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frameq
    ret, frame = video_capture.read()
     
    cv2.imwrite("dataset/single_prediction2/happy_or_sad%d.jpg" % ret, frame)     # save frame as JPEG file
    
    test_image = tf.keras.preprocessing.image.load_img('dataset/single_prediction2/happy_or_sad1.jpg',target_size=(64,64))
    test_image= tf.keras.preprocessing.image.img_to_array(test_image)
    test_image= np.expand_dims(test_image, axis=0)


    result = loaded_cnn.predict(test_image/255.0)
    if (result[0,1]> result[0,0] and result[0,1]> result[0,2]):
        result=('happy')
    elif (result[0,0]> result[0,1] and result[0,0]> result[0,2]):
        result=('sad')
    else :
        result=('shock')

    

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE
                                         )
    for (x,y,w,h) in faces:
        
        cv2.putText(img=frame, text=(result), org=(150, 100), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=2, color=(0,255,0),thickness=2)
        
        # Display the resulting frame
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()