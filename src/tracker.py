import cv2 as cv
import numpy as np
from imutils.video import FileVideoStream
import math
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

try:
    import winsound
except ImportError:
    import os
    def playSound(frequency,duration):
        os.system('beep -f %s -l %s' % (frequency,duration))
else:
    def playSound(frequency,duration):
        winsound.Beep(frequency,duration)

from src.window import Window
from src.cv_config import *
from src.colors import RED
from src.mouse_event import *


class Tracker:
    def __init__(self, video_path, fps, use_imutils):
        self.use_imutils = use_imutils
        self.video_path = video_path
        self.fps = fps
        self.window = Window('Tracker')
        if use_imutils:
            self.cap = FileVideoStream(video_path).start()
        else:
            self.cap = cv.VideoCapture(video_path)
        self.detected_objects = {}
        self.next_id = 0
        self.points = []

    def addPoint(self, point):
        self.points.append(point)

        

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
            objects = []
            for i in indices.flatten():
                box_x, box_y = boxes[i][0], boxes[i][1]
                box_width, box_height = boxes[i][2], boxes[i][3]
                color = [int(c) for c in colors[class_ids[i]]]
                box_center_x = (box_x + box_x + box_width) // 2
                box_center_y = (box_y + box_y + box_height) // 2

                already_detected = False
                for id, point in self.detected_objects.items():
                    object_point_distance = math.hypot(box_center_x - point[0], box_center_y - point[1])

                    if object_point_distance < 25:
                        left_point = (box_x, box_y + box_height)
                        right_point = (box_x + box_width, box_y + box_height)
                        self.detected_objects[id] = (box_center_x, box_center_y)
                        warning = self.detect(point, (box_center_x, box_center_y), left_point, right_point)
                        objects.append([box_x, box_y, box_width, box_height, id, color, warning])
                        already_detected = True
                        break

                if not already_detected:
                    self.detected_objects[self.next_id] = (box_center_x, box_center_y)
                    objects.append([box_x, box_y, box_width, box_height, self.next_id, color, False])
                    self.next_id += 1
                    
            new_objects = {}
            for object in objects:
                _, _, _, _, object_id, _, _ = object
                new_objects[object_id] = self.detected_objects[object_id]

            self.detected_objects = new_objects.copy()

            for object in objects:
                box_x, box_y, box_width, box_height, id, color, warning = object
                label = str(id)
                if (warning):
                    label = f"Warning!!! - {id}"
                    color = RED
                    frequency = 2500  # Set Frequency To 2500 Hertz
                    duration = 1000  # Set Duration To 1000 ms == 1 second
                    playSound(frequency, duration)
                self.window.show_rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), color, 2)
                self.window.put_label(frame, label, (box_x, box_y - 15), color, 2)

    def detect(self, old_location, new_location, left, right):
        initial_point = Point(new_location[0], new_location[1])
        left_point = Point(left[0], left[1])
        right_point = Point(right[0], right[1])
        x_speed = (new_location[0] - old_location[0]) * self.fps
        y_speed = (new_location[1] - old_location[1]) * self.fps
        x = new_location[0] + x_speed * 2
        y = new_location[1] + y_speed * 2
        point = Point(x, y)
        polygon = Polygon(self.points)
        return polygon.contains(point) or polygon.contains(initial_point) or polygon.contains(left_point) or polygon.contains(right_point)

    def run(self):
        if self.use_imutils:
            frame = self.cap.read()
        else:
            ret, frame = self.cap.read()

        self.window.show_frame(frame)
        self.window.setMouseCallback(self)

        cv.waitKey(0)

        while self.cap.more() if self.use_imutils else True:
            if self.use_imutils:
                frame = self.cap.read()
            else:
                ret, frame = self.cap.read()

            blob = cv.dnn.blobFromImage(
                frame, 1/255.0, (320, 320), (0, 0, 0), True, crop=False)

            net.setInput(blob)
            outputs = np.vstack(net.forward(output_layers))

            self.process_frame(frame, outputs, 0.5)
            self.window.show_danger_zone(frame, self.points)
            self.window.show_frame(frame)

            key = cv.waitKey(1)
            if key == 27:
                if self.use_imutils:
                    self.cap.stop()
                else:
                    self.cap.release()
                self.window.close()
                break
            elif key == ord(' '):
                cv.waitKey(0)
