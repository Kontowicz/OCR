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
from keras.models import model_from_json
K.set_image_dim_ordering('th')

class model():
    def __init__(self):
        self.model = None

    def readData(self, pathToLabels, pathToDataset):
        data = []
        labels = open(pathToLabels, 'r')
        labels_data = labels.read()
        labels_data = labels_data.split('\n')
        regex = '([\d]+) (.+) (.+)'
        p = re.compile(regex)
        labels.close()
        contain = {}

        for line in labels_data:
            if line:
                m = p.match(line)
                number = m.group(1)
                char = m.group(2)
                font = m.group(3)
                contain[number+'.jpg'] = [char, font]
        
        for file in os.listdir(pathToDataset):
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

        data = [np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)]

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

        return [X_train, y_train, X_test, y_test, num_classes]

    def trainModel(self):
        data = self.readData()
        self.model = self.createModel(data[4])

        self.model.fit(data[0], data[1], validation_data=(data[2], data[3]), epochs=10, batch_size=200, verbose=2)
        scores = model.evaluate(X_test, y_test, verbose=0)

        print("CNN Error: %.2f%%" % (100-scores[1]*100))

    def createModel(self, num_classes):
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def saveModel(self, modelName, modelWeight):
        model_json = self.model.to_json()
        with open(modelName, "w") as json_file:
            json_file.write(model_json)
        model.save_weights(modelWeight)

    def readModel(self, modelName, modelWeight):
        json_file = open(modelName, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(modelWeight)
        print("Loaded model from disk")

    def predict(self, file):
        img = cv2.imread(file, 0)
        imgResize = cv2.resize(img, (int(28),int(28)))
        new_tmp = np.array([imgResize])
        new = new_tmp.reshape(new_tmp.shape[0], 1, 28, 28).astype('float32')
        prediction = self.model.predict_classes(new)
        return prediction

if __name__ == "__main__":
    cnn = model()
    cnn.readModel('model.json', 'model.h5')
    print(chr(cnn.predict('R.jpg')[0]))

    