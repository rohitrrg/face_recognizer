# Importing the libraries
from PIL import Image
import cv2
from tensorflow.keras.models import load_model
import numpy as np

from tensorflow.keras.preprocessing import image

# Loading the cascades
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def recognize(img, model):
    face = cv2.resize(img, (224, 224))
    im = Image.fromarray(face, 'RGB')
    # Resizing into 128x128 because we trained the model with this image size.
    img_array = np.array(im)
    # Our keras model used a 4D tensor, (images x height x width x channel)
    # So changing dimension 128x128x3 into 1x128x128x3
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict_on_batch(img_array)
    pred = np.argmax(pred)

    if pred==0:
        name = 'Rohit'
    if pred==1:
        name = 'Vedant'
    if pred==2:
        name = 'Ramesh'

    return name


def detect():
    vc = cv2.VideoCapture(0)
    model = load_model('facefeatures_new_model.h5')
    while vc.isOpened:
        _, frame = vc.read()
        img = frame

        height, width, channels = frame.shape

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        indentities = []
        for (x, y, h, w) in faces:
            x1 = x
            y1 = y
            x2 = x+w
            y2 = y+h

            face_img = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
            identity = recognize(face_img, model)

            if identity is not None:
                img = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(img, str(identity), (x1+5, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vc.release()
    cv2.destroyAllWindows()
