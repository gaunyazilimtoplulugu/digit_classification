import cv2
import numpy as np
from keras.saving.save import load_model
from tensorflow.keras import Sequential
import warnings
warnings.filterwarnings('ignore')
width = 640
height = 480
cap = cv2.VideoCapture(0)
cap.set(3,width)
cap.set(4,height)
threshold = 0.65
modela = load_model('model_trained.h5')

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

while True:
    success,imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    # cv2.imshow('Processed Image',img)
    img = img.reshape(1,32,32,1)
    # Predict

    predict_x= modela.predict(img) 
    classes_x=np.argmax(predict_x,axis=1) # çok onemli
    predictions = modela.predict(img)

    probVal= np.amax(predictions)
    if probVal> threshold:
        cv2.putText(imgOriginal,str(classes_x) + "   %"+str(int(probVal*100)),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)

    cv2.imshow("Original Image",imgOriginal)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break