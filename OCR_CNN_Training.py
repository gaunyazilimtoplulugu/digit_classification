from keras.saving.save import load_model
import numpy as np
import cv2 
import os
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout,Flatten
from tensorflow.keras.optimizers import Adam
import pickle
import warnings
warnings.filterwarnings('ignore')
path = 'myData'
myList = os.listdir(path)
noOfClasses = len(myList)
images = []
classNo = []
imageDimensions = (32,32,3)
print('Total No Of Classes Detected: ',len(myList))
print('Importing Classes')

for x in range(0,noOfClasses): # 0 dan 9 a kadar olan klasorlerin içindeyim
    myPiclist = os.listdir(path+'/'+str(x)) # 0. klasorden başlayarak içine girdim
    for y in myPiclist: # 0. klasorun imagelerini alıyorum tek tek
        curImg = cv2.imread(path+'/'+str(x)+'/'+y) # bu imageleri cv2 de açıp
        curImg = cv2.resize(curImg,(imageDimensions[0],imageDimensions[1])) # bunları resize yaptırmalıyım ki hepsi aynı boyutta olsun ve ben bunları işleyebileyim
        images.append(curImg) # bu resimleri listeye ekliyorum
        classNo.append(x) # bu resimlerin labellarını eklemiş olduk
#     print(x,end=' ')
# print(' ')
# print(len(images)) # 10160
images = np.array(images) # arraye çevirdik
classNo = np.array(classNo)
# print(images.shape) # (10160, 32, 32, 3) 10160 tane 32 ye 32 lik 3 kanallı
# print(classNo.shape) # (10160,)


# Spliting The Data 
testRatio = 0.2
valRatio = 0.2
x_train, x_test,y_train,y_test = train_test_split(images,classNo,test_size=testRatio) # test 0.2 demek 10 da 2si test olsn datanın geri kalanı da train
# print(x_train.shape) # (8128, 32, 32, 3)
# print(y_train.shape) # (8128,)
# print(x_test.shape) # (2032, 32, 32, 3)
# print(y_test.shape) # (2032,)

# shuffle true demezsek mesela 0.2 'e denk gelir diğerleri karışmamış olur
x_train,x_validation,y_train,y_validation = train_test_split(x_train,y_train,test_size=valRatio)
# print(x_train.shape) # (6502, 32, 32, 3)
# print(x_test.shape) # (2032, 32, 32, 3)
# print(x_validation.shape) # (1626, 32, 32, 3)

# print(np.where(y_train==0)) # 0 sayılarının indexlerini verir
numOfSamples =[]
for x in range(0,noOfClasses):
    # print(f'{x}:',len(np.where(y_train==x)[0])) # 0 sayılarının sayısını veriverir 657
    numOfSamples.append(len(np.where(y_train==x)[0]))
# print(numOfSamples)

plt.figure(figsize=(10,5))
plt.bar(range(0,noOfClasses),numOfSamples)
plt.title('No of Images for each Class')
plt.xlabel('Class ID')
plt.ylabel('Number of Images')
# plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img
# img = preProcessing(x_train[30])
# img = cv2.resize(img,(300,300))
# cv2.imshow('PreProcessed',img)
# cv2.waitKey(0) # 2 geldi

# print(x_train[30].shape) # (32, 32, 3)
x_train = np.array(list(map(preProcessing,x_train))) # her eleman one by one bu function a gönderilir
# print(x_train[30].shape) # (32, 32)
x_test = np.array(list(map(preProcessing,x_test))) # her eleman one by one bu function a gönderilir
x_validation = np.array(list(map(preProcessing,x_validation))) # her eleman one by one bu function a gönderilir

# print(x_train.shape) # (6502, 32, 32)
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
# print(x_train.shape) # (6502, 32, 32, 1)
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
x_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)


dataGen = ImageDataGenerator(width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range = 0.2,
                            shear_range=0.1,
                            rotation_range=10) # degrees
dataGen.fit(x_train)
# print(x_train.shape) #(6502, 32, 32, 1)
y_train = to_categorical(y_train,noOfClasses) # one hot encoding
y_test = to_categorical(y_test,noOfClasses) # one hot encoding
y_validation = to_categorical(y_validation,noOfClasses) # one hot encoding validation = doğrulama

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5,5)
    sizeOfFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNode = 500
    model = Sequential()
    model.add(Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimensions[0],imageDimensions[1],1),activation='relu'))
    model.add(Conv2D(noOfFilters,sizeOfFilter1,activation='relu'))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu'))
    model.add(Conv2D(noOfFilters//2,sizeOfFilter2,activation='relu'))
    model.add(MaxPooling2D(sizeOfPool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(noOfNode,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax'))
    model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = myModel()
# print(model.summary())
batchSizeVal= 50
epochsVal = 15
stepsPerEpochVal =len(x_train)//batchSizeVal # 2000 girince hata veriyor
history = model.fit_generator(dataGen.flow(x_train,y_train,
                                 batch_size=batchSizeVal),
                                 steps_per_epoch=stepsPerEpochVal,
                                 epochs=epochsVal,
                                 validation_data=(x_validation,y_validation),
                                 shuffle=1)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training','validation'])
plt.title('Loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(x_test,y_test,verbose=0)
print('Test Score: ',score[0])
print('Test Accracy:: ',score[1])

model.save('model_traineddeneme.h5')
# predict_x= modela.predict(img) 
# classes_x=np.argmax(predict_x,axis=1)