import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras import optimizers
from keras import losses
import cv2
from datasetReader import datasetReader
from keras.models import model_from_json

K.set_image_dim_ordering('th')



class model():
    def __init__(self, data):
        self.model = None
        self.dataset = data

    def trainModel(self, isSimpleModel, epochs):
        data = self.dataset

        if isSimpleModel == True:
            self.model = self.createSimpleModel(data[4])
        else:
            self.model = self.createComplexModel(data[4])

        self.model.fit(data[0], data[1], validation_data=(data[2], data[3]), epochs= epoch, batch_size=200, verbose=2)
        scores = self.model.evaluate(data[2], data[3], verbose=0)
        print("CNN Error: %.2f%%" % (100-scores[1]*100))

    def createSimpleModel(self, num_classes):
        print('Num classes: {}'.format(num_classes))
        model = Sequential()
        model.add(Conv2D(32, (5, 5), input_shape=(1, 28, 28), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def createComplexModel(self, num_classes):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(1, 28, 28)))
        model.add(Activation('relu'))
        BatchNormalization(axis=-1)
        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        BatchNormalization(axis=-1)
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        BatchNormalization(axis=-1)
        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        BatchNormalization()
        model.add(Dense(512))
        model.add(Activation('relu'))
        BatchNormalization()
        model.add(Dropout(0.2))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.compile(loss= losses.categorical_crossentropy, optimizer= optimizers.Adam(), metrics=['accuracy'])

        return model

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
        print("Loaded model from disk")

    def predict(self, file):
        img = cv2.imread(file, 0)
        imgResize = cv2.resize(img, (int(28),int(28)))
        new_tmp = np.array([imgResize])
        new = new_tmp.reshape(new_tmp.shape[0], 1, 28, 28).astype('float32')
        prediction = self.model.predict_classes(new)
        return prediction

if __name__ == "__main__":
    # reader = datasetReader()
    # reader.readDataFromArchiwe('label_synthetic_dataset_new.txt', '../Synthetic_dataset_new.zip', True)
    cnn = model([])
    cnn.readModel('newweights.json', 'newmodels.h5')
    print(chr(cnn.predict('e.jpg')))

