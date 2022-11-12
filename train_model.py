import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.optimizers import Adam
import os
import pickle
dir = os.listdir('./archive/Train/')

images = []
classes = []
number = 0

for i in dir:
    if i != '.DS_Store':
        if int(i)>number:
            number = int(i)
        for j in os.listdir('./archive/Train/'+i):
            if j != '.DS_Store':
                cap = cv2.imread('./archive/Train/'+i+'/'+j)
                cap = cv2.cvtColor(cap, cv2.COLOR_RGB2GRAY)
                cap = cv2.resize(cap, (32, 32), interpolation=cv2.INTER_AREA)
                images.append(cap)
                classes.append(int(i))

images = np.array(images)
images = images/255
classes = np.array(classes)
images = images.reshape((-1,32,32,1))

def mymod():
    model = Sequential()
    model.add(Conv2D(60, (5,5), input_shape=(32,32,1), activation='relu'))
    model.add(Conv2D(60, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(Conv2D(30, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number+1, activation='softmax'))

    model.compile(Adam(lr=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = mymod()
print(model.summary())

model.fit(images,classes,batch_size=50,steps_per_epoch=1500, epochs=2)

pickle_out = open("model.p","wb")
pickle.dump(model, pickle_out)
pickle_out.close()
cv2.waitKey(0)