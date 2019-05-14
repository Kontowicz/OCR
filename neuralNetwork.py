import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
import cv2
import os
import re
K.set_image_dim_ordering('th')

def readData():
    data = []
    labels = open('label_synthetic_dataset_new.txt', 'r')
    labels_data = labels.read()
    labels_data = labels_data.split('\n')
    regex = '([\d]+) (.+) (.+)'
    p = re.compile(regex)

    contain = {}

    for line in labels_data:
        if line:
            m = p.match(line)
            number = m.group(1)
            char = m.group(2)
            font = m.group(3)
            contain[number+'.jpg'] = [char, font]
    
    for file in os.listdir('Synthetic_dataset_new'):
        if file.endswith('.jpg'):
            label = contain[file][0]
            img = cv2.imread('Synthetic_dataset_new/{}'.format(file), 0)
            img2 = cv2.resize(img, (28,28))
            data.append([label, img2])

    train = data[0:int(len(data)*0.7)]     
    test = data[int(len(data)*0.7):len(data)-1]

    X_train = []
    Y_train = []

    X_test = []
    Y_test = []

    for row in train:
        X_train.append(row[1])
        Y_train.append(ord(row[0]))
    
    for row in test:
        X_test.append(row[1])
        Y_test.append(ord(row[0]))

    return [np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)]
    
data = readData()

X_train = data[0]
y_train = data[1]
X_test = data[2]
y_test = data[3]

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

def baseline_model():
    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
scores = model.evaluate(X_test, y_test, verbose=0)
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
print("CNN Error: %.2f%%" % (100-scores[1]*100))