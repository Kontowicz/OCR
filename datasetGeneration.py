from PIL import Image, ImageDraw, ImageFont
import glob
import os

strip_width, strip_height = 50, 50

def center_text(img, font, text, color=(0,0,0)):
    draw = ImageDraw.Draw(img)
    text_width, text_height = draw.textsize(text, font)
    position = ((strip_width-text_width)/2,(strip_height-text_height)/2)
    draw.text(position, text, color, font=font)
    return img

def generate_dataset(out, path_to_fonts, image_width, image_height, size):

    for font in glob.glob(path_to_fonts):
        text = "A"

        background = Image.new('RGB', (strip_width, strip_height), (255, 255, 255))  # creating the black strip
        myfont = ImageFont.truetype(font, size)
        center_text(background, myfont, text)
        background.save("hello.png", "PNG")

generate_dataset('./out', './fonts/*', 50, 50, 30)