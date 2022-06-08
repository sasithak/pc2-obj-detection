import cv2 as cv
import numpy as np

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)

CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
YELLOW = (0, 255, 255)

colors = (RED, GREEN, BLUE, MAGENTA, CYAN, YELLOW, WHITE)
p0 = p1 = 0, 0

img0 = np.zeros((200, 500, 3), np.uint8)
img = img0.copy()
cv.imshow('window', img)

pts = []

def draw(x):
    d = cv.getTrackbarPos('thickness', 'window')
    d = -1 if d==0 else d
    i = cv.getTrackbarPos('color', 'window')
    color = colors[i]
    img[:] = img0
    cv.polylines(img, np.array([pts]), True, color, d)
    cv.imshow('window', img)
    text = f'color={color}, thickness={d}'
    cv.displayOverlay('window', text)

def mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        pts.append((x, y))
        draw(0)

cv.setMouseCallback('window', mouse)
cv.createTrackbar('color', 'window', 0, 6, draw)
cv.createTrackbar('thickness', 'window', 2, 10, draw)
draw(0)

cv.waitKey(0)
cv.destroyAllWindows()