from PIL import Image, ImageDraw, ImageFont, ImageFilter
import glob
import os
import re
import shutil
import cv2
import numpy as np

def gausian_blur(img, size = 3):
    return cv2.GaussianBlur(img,(size, size), 0)

def avg(img, size = 3):
    return cv2.blur(img,(size, size))

def median(img, size = 3):
    return cv2.medianBlur(img, size)

def readBin(binLabels):
    with open(binLabels, 'rb') as file:
        data = file.read()
        data = data.decode('utf-8')
        data = data.split('\n')
        for item in data:
            print(item)

def center_text(img, font, text, strip_width, strip_height, text_color=(0,0,0)):
    draw = ImageDraw.Draw(img)
    text_width, text_height = draw.textsize(text, font)
    position = ((strip_width-text_width)/2,(strip_height-text_height)/2)
    draw.text(position, text, text_color, font=font)
    return img

def generate_dataset(out, path_to_fonts, image_width, image_height, sizes):

    if not os.path.exists(out):
        os.mkdir(out)
    else:
        shutil.rmtree(out)
        os.mkdir(out)
        os.mkdir('{}/train'.format(out))
        os.mkdir('{}/test'.format(out))

    file = open('{}/label'.format(out), 'wb')

    all_characters = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZąęćźżĄĘĆŹŻ!@#$%^&*()_+=-[]\{\};:",.<>/?\''

    counter = 0
    font_counter = 0
    pattern = '.*/(.*)'

    for size in sizes:
        for i, font in enumerate(glob.glob(path_to_fonts)):
            match = re.match(pattern, font)

            font_name = match.group(1)
            font_counter += 1

            myfont = ImageFont.truetype(font, size)

            if i % 6 == 0:
                path_to_save = '{}/{}/'.format(out, 'test')
            else:
                path_to_save = '{}/{}/'.format(out, 'train')


            for character in all_characters:
                background = Image.new('RGB', (image_width, image_height), (255, 255, 255))
                center_text(background, myfont, character, image_width, image_height)

                img = np.array(background)

                cv2.imwrite('{}/{}.png'.format(path_to_save, counter), img)
                file.write('{} {} {} {}\n'.format(counter, character, font_name, font_counter).encode('utf-8'))
                counter += 1

                # img_blur = gausian_blur(img)
                # cv2.imwrite('{}/{}.png'.format(path_to_save, counter), img_blur)
                # file.write('{} {} {} {}\n'.format(counter, character, font_name, font_counter).encode('utf-8'))
                # counter += 1
                #
                # img_blur = avg(img)
                # cv2.imwrite('{}/{}.png'.format(path_to_save, counter), img_blur)
                # file.write('{} {} {} {}\n'.format(counter, character, font_name, font_counter).encode('utf-8'))
                # counter += 1
                #
                # img_blur = median(img)
                # cv2.imwrite('{}/{}.png'.format(path_to_save, counter), img_blur)
                # file.write('{} {} {} {}\n'.format(counter, character, font_name, font_counter).encode('utf-8'))
                # counter += 1
    file.close()

if __name__ == '__main__':
    generate_dataset('../out', './fonts/*', 50, 50, [25, 30, 35, 40, 45])