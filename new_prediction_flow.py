from neuralNetwork import model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

character_type_cnn = model([])
character_type_cnn.readModel('models/4/char_type/model.json', 'models/4/char_type/weight.h5')

numbers_cnn = model([])
numbers_cnn.readModel('models/4/numbers/model.json', 'models/4/numbers/weight.h5')

character_cnn = model([])
character_cnn.readModel('models/4/letters/model.json', 'models/4/letters/weight.h5')

specjal_cnn = model([])
specjal_cnn.readModel('models/4/specjal/model.json', 'models/4/specjal/weight.h5')

SMALL_HEIGHT = 800

def implt(img, cmp=None, t=''):
    """Show image using plt."""
    plt.imshow(img, cmap=cmp)
    plt.title(t)
    plt.show()

def implt_group(images, cmp=None, t=''):
    plt.figure()

    for i, im in enumerate(images):
        if i < 16:
            plt.subplot(2, 8, i + 1)
            plt.axis('off')
            plt.imshow(im, cmap=cmp, aspect='auto')
    plt.show()

def resize(img, height=SMALL_HEIGHT, always=False):
    """Resize image to given height."""
    if (img.shape[0] > height or always):
        rat = height / img.shape[0]
        return cv2.resize(img, (int(rat * img.shape[1]), height))

    return img

def ratio(img, height=SMALL_HEIGHT):
    """Getting scale ratio."""
    return img.shape[0] / height

image = cv2.cvtColor(cv2.imread('data/nowitam2.png'), cv2.COLOR_BGR2RGB)

img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def sobel(channel):
    """ The Sobel Operator"""
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1)
    # Combine x, y gradient magnitudes sqrt(x^2 + y^2)
    sobel = np.hypot(sobelX, sobelY)
    sobel[sobel > 255] = 255

    return np.uint8(sobel)

def edge_detect(im):
    """
    Edge detection
    The Sobel operator is applied for each image layer (RGB)
    """
    return np.max(np.array([sobel(im[:, :, 0]), sobel(im[:, :, 1]), sobel(im[:, :, 2])]), axis=0)

def union(a, b):
    x = min(a[0], b[0])
    y = min(a[1], b[1])
    w = max(a[0] + a[2], b[0] + b[2]) - x
    h = max(a[1] + a[3], b[1] + b[3]) - y
    return [x, y, w, h]

def contain(a, b):
    wMin = min(a[2], b[2])
    aL = a[0]
    aR = a[0] + a[2]
    bL = b[0]
    bR = b[0] + b[2]

    # b is smaller
    if wMin == b[2]:
        return bL >= aL and bR <= aR
    # a is smaller
    else:
        return aL >= bL and aR <= bR

def erase_top_white_padding(image):
    imgray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blacks = []
    for x in range(imgray.shape[1]):
        for y in range(imgray.shape[0]):
            if imgray[y, x] != 255:
                blacks.append([y, x])
    minY = (min(blacks, key=lambda t: t[0]))[0]
    minX = (min(blacks, key=lambda t: t[1]))[1]
    maxX = (max(blacks, key=lambda t: t[1]))[1]

    return image[minY:, minX:maxX]

def intersect(a, b):
    # (10, 20, )
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return False
    return True

def group_rectangles2(rec):
    """
    Union intersecting rectangles
    Args:
        rec - list of rectangles in form [x, y, w, h]
    Return:
        list of grouped ractangles
    """
    tested = [False for i in range(len(rec))]
    final = []
    i = 0
    while i < len(rec):
        if not tested[i]:
            j = i + 1
            while j < len(rec):
                if not tested[j] and contain(rec[i], rec[j]):
                    rec[i] = union(rec[i], rec[j])
                    tested[j] = True
                    j = i
                j += 1
            final += [rec[i]]
        i += 1

    return final

def group_rectangles(rec):
    """
    Union intersecting rectangles
    Args:
        rec - list of rectangles in form [x, y, w, h]
    Return:
        list of grouped ractangles
    """
    tested = [False for i in range(len(rec))]
    final = []
    i = 0
    while i < len(rec):
        if not tested[i]:
            j = i + 1
            while j < len(rec):
                if not tested[j] and intersect(rec[i], rec[j]):
                    rec[i] = union(rec[i], rec[j])
                    tested[j] = True
                    j = i
                j += 1
            final += [rec[i]]
        i += 1

    return final

def letters_detect(img, original, group=True, draw=False, offset=0):
    """ Text detection using contours """

    small = resize(img, 2000)
    image = resize(original, 2000)

    # Finding contours
    mask = np.zeros(small.shape, np.uint8)
    print(len(cv2.findContours(np.copy(small), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)))
    cnt, hierarchy = cv2.findContours(np.copy(small), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Variables for contour index and words' bounding boxes
    index = 0
    boxes = []
    while (index >= 0):
        x, y, w, h = cv2.boundingRect(cnt[index])
        # Get only the contour
        cv2.drawContours(mask, cnt, index, (255, 255, 255), cv2.FILLED)
        maskROI = mask[y:y + h, x:x + w]
        # Ratio of white pixels to area of bounding rectangle
        r = cv2.countNonZero(maskROI) / (w * h)
        # TODO Test h/w and w/h ratios
        if r > 0.1 and 2000 > w > 10 and 1600 > h > 10 and 0.25 < h / w < 10 and 0.1 < w / h < 3.5:
            boxes += [[x, y, w, h]]
        index = hierarchy[0][index][0]

    if group:
        boxes = group_rectangles2(boxes)

    offsetH = int(h * offset / 100)
    offsetW = int(w * offset / 100)

    bounding_boxes = np.array([0, 0, 0, 0])
    for (x, y, w, h) in boxes:
        if draw:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        bounding_boxes = np.vstack(
            (bounding_boxes, np.array([x - offsetW, y - offsetH, x + w + offsetW, y + h + offsetH])))
    # if draw:
    #     implt(image, t='Bounding rectangles')

    # Recalculate coordinates to original scale
    boxes = bounding_boxes.dot(ratio(image, small.shape[0])).astype(np.float32)
    return boxes[1:]

def textDetectWatershed(thresh, original, draw=False, group=False):
    """ Text detection using watershed algorithm """
    # According to: http://docs.opencv.org/trunk/d3/db4/tutorial_py_watershed.html
    img = resize(original, 3000)
    thresh = resize(thresh, 3000)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.001 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers += 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)

    image = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Creating result array
    boxes = []
    for mark in np.unique(markers):
        # mark == 0 --> background
        if mark == 0:
            continue

        # Draw it on mask and detect biggest contour
        mask = np.zeros(gray.shape, dtype="uint8")
        mask[markers == mark] = 255

        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)

        # Draw a bounding rectangle if it contains text
        x, y, w, h = cv2.boundingRect(c)
        cv2.drawContours(mask, c, 0, (255, 255, 255), cv2.FILLED)
        maskROI = mask[y:y + h, x:x + w]
        # Ratio of white pixels to area of bounding rectangle
        r = cv2.countNonZero(maskROI) / (w * h)

        # Limits for text
        if r > 0.1 and 2000 > w > 15 and 1500 > h > 15:
            boxes += [[x, y, w, h]]

    # Group intersecting rectangles
    if group:
        boxes = group_rectangles(boxes)

    boxes.sort(key=lambda b: (b[1] + b[3], b[0]))

    bounding_boxes = np.array([0, 0, 0, 0])
    rois = []

    for (x, y, w, h) in boxes:
        rois.append(img[y:y + h, x:x + w])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        bounding_boxes = np.vstack((bounding_boxes, np.array([x, y, x + w, y + h])))
    if draw:
        implt(sure_bg, t='sure_background')
        implt(sure_fg, t='sure_foreground')
        implt(unknown, t='unknown area')
        implt(markers, t='Markers')
        implt(image)
    # Recalculate coordinates to original size
    boxes = bounding_boxes[1:].dot(ratio(original, img.shape[0])).astype(np.int64)

    return boxes, rois

def sharpen(img):
    ret, newImg = cv2.threshold(img, 123, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    newImg = cv2.dilate(newImg, kernel, 3)
    blurred2 = cv2.GaussianBlur(newImg, (3, 3), 1)
    edges2 = edge_detect(blurred2)
    bw_image2 = cv2.morphologyEx(edges2, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return bw_image2

def sharpen_small(img):
    ret, newImg = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    newImg = cv2.erode(newImg, kernel, 1)
    newImgCopy = newImg.copy()


    h, w = newImg.shape[:2]
    seedPoints = []
    times = 0
    x = w // 2
    for i, y in enumerate(newImg[:, x]):

        if y == 0:
            times += 1
            if times == 3:
                #         newImgCopy[i,x]=180
                seedPoints.append((i, x))
        else:
            times = 0
    for i, seedPoint in enumerate(seedPoints):
        floodFill(image=newImg, seedPoint=seedPoint, replacementColor=180)
    newImg = (erase_noise(newImgCopy, newImg, seedPoints[0]))
    newImg[newImg != 255] = 0
    newImg = newImg.astype(np.float32)
    return newImg

def erase_noise(original, floodfilled, seedPoint):
    im_floodfill_inv = cv2.bitwise_not(floodfilled)
    im_out = original | im_floodfill_inv
    return im_out

def floodFill(image, seedPoint, replacementColor):
    h, w = image.shape[:2]
    (y, x) = seedPoint
    if x - 1 <= 0 or y - 1 <= 0 or y + 1 >= h or x + 1 >= w:
        return
    targetColor = image[seedPoint]
    if targetColor == replacementColor:
        return
    image[seedPoint] = replacementColor
    pixelsToCheck = [seedPoint]
    while 0 < len(pixelsToCheck):
        (y, x) = pixelsToCheck.pop(0)
        if x - 1 <= 0 or y - 1 <= 0 or y + 1 >= h or x + 1 >= w:
            continue
        for position in [(y, x - 1), (y + 1, x), (y, x + 1), (y - 1, x)]:
            if image[position] == targetColor:
                image[position] = replacementColor
                pixelsToCheck.append(position)

def recreate_coordinates(coords, scale, shape):
    if len(coords) != 4:
        return
    [h, w] = shape
    [x1, y1, x2, y2] = [c * scale for c in coords]
    x1 = floor(x1)
    x2 = ceil(x2)
    y1 = floor(y1)
    y2 = ceil(y2)
    [x1, y1, x2, y2] = [0 if c < 0 else c for c in [x1, y1, x2, y2]]

    [x1, x2] = [w if c > w else c for c in [x1, x2]]
    [y1, y2] = [h if c > h else c for c in [y1, y2]]

    return [x1, y1, x2, y2]

def add_padding(image, percent=30):
    img = image.copy()
    h, w = img.shape[:2]
    height = []
    for i in range(0, h):
        if 0 in img[i]:
            height.append(i)

    width = []
    for i in range(0, w):
        tmp = [row[i] for row in img]
        if 0 in tmp:
            width.append(i)
    if len(height) == 0:
        height.append(0)
        height.append(h)
    if len(width) == 0:
        width.append(0)
        width.append(w)
    minY = min(height)
    maxY = max(height)

    minX = min(width)
    maxX = max(width)
    newImg = img[minY: maxY, minX:maxX]

    h, w = newImg.shape[:2]
    # TU MAMY OBCIĘTĄ LITERKĘ ŁADNIE
    # TODO: RESIZE DO PROPORCJI
    X = int(w * 40 / h)
   # implt(newImg, 'gray', t='Before: Add padding')

    imgResized = cv2.resize(newImg, (X, 40))

    color = [255, 255, 255]
    new_im = cv2.copyMakeBorder(imgResized.copy(), 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=color)

    #implt(new_im, 'gray', t='After: Add padding')
    return new_im

grayIMG = img.copy()

b_w = cv2.adaptiveThreshold(grayIMG, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2);

horizontal_img = b_w.copy()
vertical_img = b_w.copy()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
horizontal_img = cv2.erode(horizontal_img, kernel, iterations=1)
horizontal_img = cv2.dilate(horizontal_img, kernel, iterations=1)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 100))
vertical_img = cv2.erode(vertical_img, kernel, iterations=1)
vertical_img = cv2.dilate(vertical_img, kernel, iterations=1)

mask_img = horizontal_img + vertical_img
joints = np.bitwise_and(horizontal_img, vertical_img)

inv_mask_img = np.bitwise_not(mask_img)
no_border = np.bitwise_xor(b_w, inv_mask_img)

b_w = cv2.cvtColor(no_border, cv2.COLOR_GRAY2RGB)
blurredIMG = cv2.GaussianBlur(b_w, (5, 5), 30)
edgesIMG = edge_detect(blurredIMG)
_, edgesIMG = cv2.threshold(edgesIMG, 50, 255, cv2.THRESH_BINARY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
b_w = cv2.morphologyEx(edgesIMG, cv2.MORPH_CLOSE, kernel, 3)

boxes, rois = textDetectWatershed(b_w, image.copy(), draw=False, group=True)

# Zrobić sortowanie boxów od
newHeight = 200
from math import ceil, floor

for i, roi in enumerate(rois[:34]):
    sharpened = sharpen(resize(roi, always=True))
    #   implt(sharpened, 'gray')
    [X1, Y1, X2, Y2] = boxes[i]
    roiFromOriginal = image[Y1:Y2, X1:X2]
    print("Coords", (X1, Y1), (X2, Y2))
    implt(roi)

    yRatio = ratio(roiFromOriginal, height=newHeight)
    characters = letters_detect(resize(sharpened, height=newHeight, always=True),
                                resize(roi, height=newHeight, always=True), group=True, draw=False, offset=10)

    letters = []
    for coords in sorted(characters, key=lambda t: t[0]):
        [x1, y1, x2, y2] = recreate_coordinates(coords, yRatio, roiFromOriginal.shape[:2])


        letters.append(roiFromOriginal[y1:y2, x1:x2])
    word = ""
    Lekk = 55
    for letter in letters:
        letterTmp = cv2.cvtColor(letter, cv2.COLOR_RGB2GRAY)

        letterTmp = sharpen_small(letterTmp)
        #implt(letterTmp, 'gray', t='After: Sharpen small')
        letterTmp = add_padding(letterTmp, 30)
        imgResized = cv2.resize(letterTmp, (int(50), int(50)))

        implt(imgResized, 'gray', 'Resized')
        img_tmp = Image.fromarray(letterTmp)
        result = None
        char_class = int(character_type_cnn.predict(letterTmp))
        print(f'Character class: {char_class}')
        if char_class == 0: # numbser
            result = chr(numbers_cnn.predict(letterTmp))
        if char_class == 1: # letters
            result = chr(character_cnn.predict(letterTmp))
        if char_class == 2: # specjal
            result = chr(specjal_cnn.predict(letterTmp))
        #cv2.imwrite('data/dupa.png', imgResized)
        print(f'prediction: {result}')
        # # prediction = loaded_model.predict(imgResized)
        # # print(prediction)
        # print(result)
        word += result

    print("Predicted = ", word)
