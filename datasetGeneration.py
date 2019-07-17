from PIL import Image, ImageDraw, ImageFont, ImageFilter
import glob
import os
import re
import shutil

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

def generate_dataset(out, path_to_fonts, out_file_name, image_width, image_height, size, in_many_dir):

    if not os.path.exists(out):
        os.mkdir(out)
    else:
        shutil.rmtree(out)
        os.mkdir(out)

    file = open(out_file_name, 'wb')

    all_characters = '1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZąęćźżĄĘĆŹŻ!@#$%^&*()_+=-[]\{\};:",.<>/?\''

    counter = 0
    pattern = '.*/(.*)'

    for font in glob.glob(path_to_fonts):
        match = re.match(pattern, font)
        font_name = match.group(1)
        if in_many_dir:
            os.mkdir('{}/{}'.format(out, font_name))

        myfont = ImageFont.truetype(font, size)

        for character in all_characters:
            background = Image.new('RGB', (image_width, image_height), (255, 255, 255))
            center_text(background, myfont, character, image_width, image_height)
            if in_many_dir:
                background.save('{}/{}/{}.png'.format(out, font_name, counter), "PNG")
            else:
                background.save('{}/{}.png'.format(out, counter), "PNG")
            file.write('{} {} {}\n'.format(counter, character, font_name).encode('utf-8'))
            counter += 1

            image_blur = background.filter(ImageFilter.GaussianBlur(radius=3))
            if in_many_dir:
                image_blur.save('{}/{}/{}.png'.format(out, font_name, counter), "PNG")
            else:
                image_blur.save('{}/{}.png'.format(out, counter), "PNG")
            file.write('{} {} {}\n'.format(counter, character, font_name).encode('utf-8'))
            counter += 1

            image_blur = background.filter(ImageFilter.MedianFilter())
            if in_many_dir:
                image_blur.save('{}/{}/{}.png'.format(out, font_name, counter), "PNG")
            else:
                image_blur.save('{}/{}.png'.format(out, counter), "PNG")
            file.write('{} {} {}\n'.format(counter, character, font_name).encode('utf-8'))
            counter += 1

            image_blur = background.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            if in_many_dir:
                image_blur.save('{}/{}/{}.png'.format(out, font_name, counter), "PNG")
            else:
                image_blur.save('{}/{}.png'.format(out, counter), "PNG")
            file.write('{} {} {}\n'.format(counter, character, font_name).encode('utf-8'))
            counter += 1
    file.close()

generate_dataset('../out', './fonts/*', './labels', 50, 50, 30, False)