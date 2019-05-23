from PIL import Image, ImageDraw, ImageFont
import glob
import os
import cv2
import numpy as np

def addPadding(img, targetSize, percent = 30):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    height = []
    for i in range(0, len(img)):
        if 0 in img[i]:
            height.append(i)

    width = []
    for i in range(0, len(img)):
        tmp = [row[i] for row in img]
        if 0 in tmp:
            width.append(i)

    minX = min(height)
    maxX = max(height)

    minY = min(width)
    maxY = max(width)

    newImg = img[minX: maxX + 2, minY:maxY + 2]

    w, h = newImg.shape[:2]
    p = int(h + 2 * h * percent / 100) // 2

    color = [255, 255, 255]
    img =  cv2.copyMakeBorder(newImg, p, p, p, p, cv2.BORDER_CONSTANT, value=color)
    return cv2.resize(img, (targetSize, targetSize))

def gausiabBlur(img, size = 3):
    return cv2.GaussianBlur(img,(size, size), 0)

def avg(img, size = 3):
    return cv2.blur(img,(size, size))

def median(img, size = 3):
    return cv2.medianBlur(img, size)

def generateDataset(fontSize = 40, imageSize = 50, datasetPath = 'Dataset', labesPath = 'labels.txt'):

    all_char_list = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZąęćźżĄĘĆŹŻ!@#$%^&*()_+=-[]\{\};:",.<>/?\''
    counter = 0

    if not os.path.exists(datasetPath):
        os.makedirs(datasetPath)

    fileLables = open(labesPath, 'w')
    fonts = glob.glob("C:\\Windows\\Fonts\\*.ttf")

    ff = open('Fonts_list.txt', 'r')

    fontList = [x.lower() for x in ff.read().split('\n')]

    pos = (0,0)
    for fontName in fonts:

        fName = str(fontName).split('\\')[-1].lower()
        fName = fName.split('.')[0]

        if fName not in fontList:
            continue

        font = ImageFont.truetype(fontName, fontSize)

        for char in all_char_list:
            image = Image.new('RGB', (imageSize, imageSize), (255, 255, 255))
            draw = ImageDraw.Draw(image)
            draw.text(pos, char, (0,0,0), font=font)

            img = np.array(image)

            img = addPadding(img, imageSize)
            imgName = '{}/{}.jpg'.format(datasetPath, counter)
            cv2.imwrite(imgName, img)
            fileLables.write('{} {} {}\n'.format(counter, char, fName))
            counter += 1

            img1 = gausiabBlur(img)
            imgName = '{}/{}.jpg'.format(datasetPath, counter)
            cv2.imwrite(imgName, img1)
            fileLables.write('{} {} {}\n'.format(counter, char, fName))
            counter += 1

            img1 = avg(img)
            imgName = '{}/{}.jpg'.format(datasetPath, counter)
            cv2.imwrite(imgName, img1)
            fileLables.write('{} {} {}\n'.format(counter, char, fName))
            counter += 1

            img1 = median(img)
            imgName = '{}/{}.jpg'.format(datasetPath, counter)
            cv2.imwrite(imgName, img1)
            fileLables.write('{} {} {}\n'.format(counter, char, fName))
            counter += 1

    fileLables.close()

def labelsToBin(txtLables, binLabels):
    file = open(txtLables, 'r')
    data = file.read().split('\n')
    file.close()
    file = open(binLabels, 'wb')
    for item in data:
        item = item + '\n'
        file.write(item.encode('utf-8'))

    file.close()

def readBin(binLabels):
    with open(binLabels, 'rb') as file:
        data = file.read()
        data = data.decode('utf-8')
        data = data.split('\n')
        for item in data:
            print(item)

if __name__ == '__main__':
    # generateDataset()
    # labelsToBin('labels.txt', 'labels.bin')
    readBin('labels.bin')