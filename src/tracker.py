import cv2 as cv
import numpy as np

from src.window import Window
from src.cv_config import *


class Tracker:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        self.window = Window('Tracker')

    def process_frame(self, frame, outputs, confidence_level):
        frame_height, frame_width = frame.shape[:2]

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            scores = output[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_level:
                box_x, box_y, box_width, box_height = output[:4] * np.array(
                    [frame_width,  frame_height, frame_width,  frame_height])
                point = int(box_x - box_width //
                            2), int(box_y - box_height // 2)
                boxes.append([*point, int(box_width), int(box_height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

        indices = cv.dnn.NMSBoxes(
            boxes, confidences, confidence_level, confidence_level - 0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                (box_x, box_y) = (boxes[i][0], boxes[i][1])
                (box_width, box_height) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[class_ids[i]]]
                self.window.show_rectangle(
                    frame, (box_x, box_y), (box_x + box_width, box_y + box_height), color, 2)
                # label = ""
                # self.window.put_label(frame, label, (box_x, box_y - 5), color, 1)

    def run(self):
        while True:
            ret, frame = self.cap.read()

            blob = cv.dnn.blobFromImage(
                frame, 1/255.0, (320, 320), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outputs = np.vstack(net.forward(output_layers))

            self.process_frame(frame, outputs, 0.5)
            self.window.show_frame(frame)

            key = cv.waitKey(1)
            if key == 27:
                self.cap.release()
                self.window.close()
                break
            elif key == ord(' '):
                cv.waitKey(0)
