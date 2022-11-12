import pickle
import numpy as np
import cv2
import os
import pandas as pd

pickle_in = open('model.p','rb')
model = pickle.load(pickle_in)
l = []
p = []
test = cv2.imread('./archive/banibani.png')
test = cv2.cvtColor(test, cv2.COLOR_RGB2GRAY)
test = cv2.resize(test, (32, 32), interpolation=cv2.INTER_AREA)
l.append(test)
# p.append(i)
l = np.array(l)
l = l/255
l = l.reshape((-1,32,32,1))
# print(l)


pred = np.argmax(model.predict(l),axis=1)
print(pred)

