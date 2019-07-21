import random
import glob
import cv2
import numpy as np
import re
import os
from MyZipFile import ZipFile
from keras.utils import np_utils


class datasetReader:
    def __init__(self):
        self.data = None

    def read_labels(self, path_to_labels):
        with open('{}/label'.format(path_to_labels), 'rb') as file:
            character_data = file.read()
            character_data = character_data.decode('utf-8')
            character_data = character_data.split('\n')
            pattern = '(.*) (.*) (.*) (.*)'
            # counter, character, font_name, font_counter

            character_dictionary = {}
            font_dictionary = {}

            for item in character_data:
                if item != '\n' and item != '':
                    match = re.match(pattern, item)

                    counter_picture = match.group(1)
                    font_name = match.group(3)
                    character = ord(match.group(2))
                    font_counter = match.group(4)

                    character_dictionary['{}.png'.format(counter_picture)] = character
                    font_dictionary[font_name] = font_counter

        return character_dictionary, font_dictionary

    def prepare_data(self, train, test):
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

        X_train = X_train.reshape(X_train.shape[0], 1, 50, 50).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], 1, 50, 50).astype('float32')

        X_train = X_train / 255
        X_test = X_test / 255

        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)

        num_classes = y_test.shape[1]

        self.data = [X_train, y_train, X_test, y_test, num_classes]

    def read_data(self, path_to_data):
        print('Data is reading from: {}'.format(path_to_data))

        character_dictionary, font_dictionary = self.read_labels(path_to_data)

        # Read train data
        train_data = []
        for picture in glob.glob('{}/train/*'.format(path_to_data)):
            pattern = '.*/(.*)'
            match = re.match(pattern, picture)
            train_data.append([character_dictionary[match.group(1)], cv2.imread(picture, 0)])

        # Read train data
        test_data = []
        for picture in glob.glob('{}/test/*'.format(path_to_data)):
            pattern = '.*/(.*)'
            match = re.match(pattern, picture)
            test_data.append([character_dictionary[match.group(1)], cv2.imread(picture, 0)])

        self.prepare_data(train_data, test_data)


    def read_data_from_archiwe(self, path_to_data):
        print('Data is reading from archiwe: {}/train and {}/test'.format(path_to_data, path_to_data))

        character_dictionary, font_dictionary = self.read_labels(path_to_data)

        # Read train data
        train_data = []
        with ZipFile('{}/train.zip'.format(path_to_data)) as archiwe:
            for entry in archiwe.infolist():
                data = archiwe.read(entry)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), 0)
                train_data.append([character_dictionary[entry.filename], img])

        test_data = []
        with ZipFile('{}/test.zip'.format(path_to_data)) as archiwe:
            for entry in archiwe.infolist():
                data = archiwe.read(entry)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), 0)
                test_data.append([character_dictionary[entry.filename], img])
        self.prepare_data(train_data, test_data)