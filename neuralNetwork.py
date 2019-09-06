import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import *
from keras import backend as K
from datasetReader import datasetReader
import cv2
# keras.__version__ == 2.2.4
from keras.models import model_from_json

K.set_image_dim_ordering('th')


class model():
    def __init__(self, data=[]):
        self.model = None
        self.dataset = data
        self.imageSize = 50

    def train(self):
        self.model.fit(self.dataset[0], self.dataset[1], validation_data=(self.dataset[2], self.dataset[3]), epochs=10,
                       batch_size=500, verbose=2)
        scores = self.model.evaluate(self.dataset[2], self.dataset[3], verbose=0)

        print("CNN Error: %.2f%%" % (100 - scores[1] * 100))

        return (100 - scores[1] * 100)

    def createSimpleModel(self):
        num_classes = self.dataset[4]
        print('Num classes: {}'.format(num_classes))
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(1, self.imageSize, self.imageSize), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.3))
        model.add(Flatten())
        model.add(Dense(num_classes * 2, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model

    def create_complex(self):
        num_classes = self.dataset[4]
        model = Sequential()

        model.add(Conv2D(32, (5,5), input_shape=(1, self.imageSize, self.imageSize)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Conv2D(16, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(8, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(num_classes * 2))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model = model



    def saveModel(self, modelName, modelWeight):
        model_json = self.model.to_json()
        with open(modelName, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(modelWeight)

    def readModel(self, modelName, modelWeight):
        json_file = open(modelName, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(modelWeight)
        # self.model.compile(loss= losses.categorical_crossentropy, optimizer= optimizers.Adam(), metrics=['accuracy'])
        print("Loaded model from disk")

    def predict(self, img):
        #img = cv2.imread(file, 0)
        imgResize = cv2.resize(img, (int(50), int(50)))
        new_tmp = np.array([imgResize])
        new = new_tmp.reshape(new_tmp.shape[0], 1, 50, 50).astype('float32')
        prediction = self.model.predict_classes(new)
        return chr(prediction)

if __name__ == '__main__':
    reader = datasetReader()

    #reader.read_data('/home/piotr/Desktop/develop/repo/out')
    reader.read_data('../out')

    model = model(reader.data)
    model.create_complex()
    model.train()