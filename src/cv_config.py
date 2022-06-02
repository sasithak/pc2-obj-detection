import cv2 as cv
import numpy as np


WHITE = (255, 255, 255)

# Load names of classes and get random colors
classes = open('config/coco.names').read().strip().split('\n')
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classes), 3), dtype='uint8')

# Load the yolov3 algorithm
net = cv.dnn.readNetFromDarknet(
    'config/yolov3.cfg', 'config/yolov3.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

output_layers = net.getLayerNames()
output_layers = [output_layers[i - 1] for i in net.getUnconnectedOutLayers()]
