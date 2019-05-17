import random
import cv2
import numpy as np
import re
import os
from MyZipFile import ZipFile
from keras.utils import np_utils

fonts_code = {'Calibri' : 0, 'Times' : 1, 'Arial' : 2, 'Aparaj' : 3, 'Book' : 4, 'Cambria' : 5, 'Candara' : 6, \
            'Century' : 7, 'Consola' : 8, 'Constan' : 9, 'Corbel' : 10, 'DokChamp' : 11, 'Euphemia' : 12, \
            'FRAD' : 13, 'FRAH' : 14, 'FRAMD' : 15, 'FRAB' : 16, 'Gara' : 17, 'Georgia' : 18, 'Impact' : 19,\
            'Verdana' : 20}

class datasetReader:
    def __init__(self):
        self.orginalData = None

    def getData(self):
        data = self.orginalData
        random.shuffle(data)
        train = data[0:int(len(data) * 0.7)]
        test = data[int(len(data) * 0.7):len(data) - 1]

        X_train = []
        Y_train = []

        X_test = []
        Y_test = []

        for row in train:
            X_train.append(row[1])
            Y_train.append(row[0])

        for row in test:
            X_test.append(row[1])
            Y_test.append(row[0])

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

    def readData(self, pathToLabels, pathToDataset, isFont):
        print('Data is reading from directory: {}'.format(pathToDataset))
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
                contain[number + '.jpg'] = [char, font]

        for file in os.listdir(pathToDataset):
            if file.endswith('.jpg'):
                label = None
                if isFont == True:
                    label = fonts_code[contain[file][1]]
                else:
                    label = ord(contain[file][0])
                img = cv2.imread('{}/{}'.format(pathToDataset, file), 0)
                img2 = cv2.resize(img, (28, 28))
                data.append([label, img2])

        self.orginalData = data

    def readDataFromArchiwe(self, pathToLables, pathToArchiwe, isFont):
        assert pathToArchiwe.endswith('zip') == True
        print('Data is reading from archiwe: {}'.format(pathToArchiwe))
        dataToReturn = []
        labels = open(pathToLables, 'r')
        labels_data = labels.read()
        labels_data = labels_data.split('\n')
        labels.close()
        labelRegex = re.compile('([\d]+) (.+) (.+)')
        contain = {}

        for line in labels_data:
            if line:
                m = labelRegex.match(line)
                contain[m.group(1)] = [m.group(2), m.group(3)]

        with ZipFile(pathToArchiwe) as archiwe:
            regex = '.+/(.+)\.jpg'
            matchObject = re.compile(regex)
            for entry in archiwe.infolist():
                fileName = matchObject.match(entry.filename).group(1)
                label = None
                if isFont == True:
                    label = fonts_code[contain[fileName][1]]
                else:
                    label = ord(contain[fileName][0])
                data = archiwe.read(entry)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), 0)
                img2 = cv2.resize(img, (28, 28))

                dataToReturn.append([label, img2])

        self.orginalData = dataToReturn
