import pytesseract
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

SMALL_HEIGHT = 800

def implt(img, cmp=None, t=''):
    """Show image using plt."""
    plt.imshow(img, cmap=cmp)
    plt.title(t)
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

def intersect(a, b):
    # (10, 20, )
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[0] + a[2], b[0] + b[2]) - x
    h = min(a[1] + a[3], b[1] + b[3]) - y
    if w < 0 or h < 0:
        return False
    return True

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

def get_words_cords(images):
    text_cords = {}
    for path, image in images:
        img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

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

        inv_mask_img = np.bitwise_not(mask_img)
        no_border = np.bitwise_xor(b_w, inv_mask_img)

        b_w = cv2.cvtColor(no_border, cv2.COLOR_GRAY2RGB)
        blurredIMG = cv2.GaussianBlur(b_w, (5, 5), 30)
        edgesIMG = edge_detect(blurredIMG)
        _, edgesIMG = cv2.threshold(edgesIMG, 50, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        b_w = cv2.morphologyEx(edgesIMG, cv2.MORPH_CLOSE, kernel, 3)

        boxes, rois = textDetectWatershed(b_w, image.copy(), draw=False, group=True)

        for i, roi in enumerate(rois[:34]):
            [X1, Y1, X2, Y2] = boxes[i]
            cords = [(X1, Y1), (X2, Y2)]
            tmp_img = Image.fromarray(roi)
            word = pytesseract.image_to_string(tmp_img)
            if word != '':
                if word not in text_cords:
                    text_cords[word] = {path : cords}
                else:
                    if path not in text_cords[word]:
                        text_cords[word][path] = cords
                    else:
                        text_cords[word][path] = text_cords[word][path].append(cords)
    print(text_cords)
    return text_cords

