import numpy
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
K.set_image_dim_ordering('th')

def readData():
    data = []
    for file in os.listdir('Synthetic_dataset'):
        if file.endswith('.jpg'):
            label = file.replace('.jpg', '')
            label = label[-1]
            data.append([label, cv2.imread('Synthetic_dataset/{}'.format(file), 0)])
    train = data[0:int(len(data)*0.7)]     
    test = data[int(len(data)*0.7):len(data)-1]

    X_train = numpy.arange(len(train))
    Y_train = numpy.array(len(train))

    X_test = numpy.array(len(test))
    Y_test = numpy.array(len(test))

    for i in range(0, len(train)):
        X_train[i] = train[i][1]
        Y_train[i] = ord(train[i][0])
    
    for i in range(0, len(test)):
        X_test[i] = test[i][1]
        Y_test[i] = ord(test[i][0])

    return [X_train, Y_train, X_test, Y_test]
    
    

data = readData()
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
X_train = data[0]
y_train = data[1]
X_test = data[2]
y_test = data[3]
# reshape to be [samples][pixels][width][height]
# X_train = numpy.reshape(data[0], 1, 28, 28).astype('float32') 
# X_test = numpy.reshape(data[2], 1, 28, 28).astype('float32')

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

def baseline_model():
    	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))