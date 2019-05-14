from PIL import Image, ImageDraw, ImageFont
import string
import ntpath
import numpy as np
import os
import glob
import os
import random
import cv2
import numpy as np
import re

random.seed(2)

def blur(img):
   return cv2.blur(img, (5,5))

def gaussian(img):
   return cv2.GaussianBlur(img, (5,5), 0)

def median(img):
   return cv2.medianBlur(img,5)

def generateBasicDataset():
    fontSize = 40
    imgSize = (50,50)
    position = (0,0)
    
    dataset_path = os.path.join (os.getcwd(), 'Synthetic_dataset')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)
    
    fhandle = open('Fonts_list.txt', 'r')

    all_char_list = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZąęćźżĄĘĆŹŻ!@#$%^&*()_+=-[]\{\};:",.<>/?\''

    counter = 0

    fonts_list = []
    for line in fhandle:
        fonts_list.append(line.rstrip('\n'))
    
    total_fonts = len(fonts_list)
    all_fonts = glob.glob("C:\\Windows\\Fonts\\*.ttf")
    f_flag = np.zeros(total_fonts)
    file = open('label.txt', 'w')
    
    for sys_font in all_fonts:
        font_file = ntpath.basename(sys_font)
        font_file = font_file.rsplit('.')
        font_file = font_file[0]
        f_idx = 0
        for font in fonts_list:
            tmp_font = font
            f_lower = font.lower()
            s_lower = sys_font.lower()
            if f_lower in s_lower:
                path = sys_font
                font = ImageFont.truetype(path, fontSize)
                f_flag[f_idx] = 1
                for ch in all_char_list:
                    image = Image.new("RGB", imgSize, (255,255,255))
                    draw = ImageDraw.Draw(image)
                    pos_x = 0
                    pos_y = 0
                    pos_idx=0
                    for y in [pos_y-1, pos_y, pos_y+1]:
                        for x in [pos_x-1, pos_x, pos_x+1]:
                            position = (x,y)
                            draw.text(position, ch, (0,0,0), font=font)
                            l_u_d_flag = "u"
                            if ch.islower():
                                l_u_d_flag = "l"
                            elif ch.isdigit():
                                l_u_d_flag = "d"

                            file_name = '{}.jpg'.format(counter)
                            file_name = os.path.join(dataset_path,file_name)
                            file.write('{} {} {}\n'.format(counter, ch, tmp_font))
                            image.save(file_name)
                            counter += 1
                            pos_idx = pos_idx + 1
            f_idx = f_idx + 1

def generateExtendedDataset():
    new_labels = open('new_label.txt', 'w')
    counter = 0

    labels = open('label.txt', 'r')
    data = labels.read()
    data = data.split('\n')
    contain = {}
    regex = '([\d]+) (.+) (.+)'
    p = re.compile(regex)

    for line in data:
        if line:
            m = p.match(line)
            number = m.group(1)
            char = m.group(2)
            font = m.group(3)

            contain[number] = [char, font]

    labels.close()

    for file in os.listdir('Synthetic_dataset'):
        name = file.replace('.jpg', '')
        img = cv2.imread('Synthetic_dataset/' + file, 0)
        
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

        newImg = img[minX : maxX+2, minY:maxY+2]
        h,w = newImg.shape[:2]
        paddingL = (48-w)//2
        paddingT = (48-h)//2

        canvas = np.ones((48,48), np.uint64)
        canvas[:,:] = (255)
        canvas[paddingT:paddingT+h, paddingL:paddingL+w] = newImg

        file_name = '{}.jpg'.format(counter)
        new_labels.write('{} {} {}\n'.format(counter, contain[name][0], contain[name][1]))
        cv2.imwrite('Synthetic_dataset_new/' + file_name, canvas)
        img = cv2.imread('Synthetic_dataset_new/' + file_name, 0)
        counter += 1

        img = cv2.imread('Synthetic_dataset_new/' + file_name, 0)

        file_name = '{}.jpg'.format(counter)
        new_labels.write('{} {} {}\n'.format(counter, contain[name][0], contain[name][1]))
        imgBlur = blur(img)
        cv2.imwrite('Synthetic_dataset_new/' + file_name, imgBlur)
        counter += 1

        file_name = '{}.jpg'.format(counter)
        new_labels.write('{} {} {}\n'.format(counter, contain[name][0], contain[name][1]))
        imgGaussian = gaussian(img)
        cv2.imwrite('Synthetic_dataset_new/' + file_name, imgGaussian)
        counter += 1

        file_name = '{}.jpg'.format(counter)
        new_labels.write('{} {} {}\n'.format(counter, contain[name][0], contain[name][1]))
        imgMedian = median(img)
        cv2.imwrite('Synthetic_dataset_new/' + file_name, imgMedian)
        counter += 1


