import os
import cv2
import numpy as np

scanPath = os.path.join('gdrive','My Drive','Files','scan1.jpg')

def BGRtoGRAY(image):
  if image.shape[2] == 3:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  else:
    return image

def morphOpen(image, kernelSize):
  kernel = np.ones((kernelSize, kernelSize),np.uint8)
  return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
  
def straightenImage(image, kernelSize=3):
  gray = BGRtoGRAY(image)
  gray = cv2.bitwise_not(gray)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
  thresh = morphOpen(thresh, kernelSize)
  coords = np.column_stack(np.where(thresh > 0))

  angle = cv2.minAreaRect(coords)[-1]
  if angle < -45:
    angle = -(90 + angle)
  else:
    angle = -angle
  (h, w) = image.shape[:2]
  center = (w // 2, h // 2)
  M = cv2.getRotationMatrix2D(center, angle, 1.0)
  print("Rotated by an angle of: {:.3f}".format(angle))
  return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def showResizedImage(image, divider):
  height, width = image.shape[:2]
  cv2.imshow(cv2.resize(image, (width//divider, height//divider), interpolation = cv2.INTER_CUBIC))

scan = cv2.imread(scanPath)
#showResizedImage(scan, 3)

straightImage = (straightenImage(scan))
#showResizedImage(straightImage, 3)