import cv2 as cv
import cv2
import numpy as np
from imutils.video import FileVideoStream
import math

from src.window import Window
from src.cv_config import *


class Tracker:
    def __init__(self, video_path, use_imutils):
        self.use_imutils = use_imutils
        self.video_path = video_path
        self.window = Window('Tracker')
        if use_imutils:
            self.cap = FileVideoStream(video_path).start()
        else:
            self.cap = cv.VideoCapture(video_path)
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0

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
            objects_bbs_ids = []
            for i in indices.flatten():
                (box_x, box_y) = (boxes[i][0], boxes[i][1])
                (box_width, box_height) = (boxes[i][2], boxes[i][3])
                color = [int(c) for c in colors[class_ids[i]]]
                x, y, w, h = box_x,box_y,box_width,box_height
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2

                # Find out if that object was detected already
                same_object_detected = False
                for id, pt in self.center_points.items():
                    dist = math.hypot(cx - pt[0], cy - pt[1])

                    if dist < 25:
                        self.center_points[id] = (cx, cy)
                        objects_bbs_ids.append([x, y, w, h, id])
                        same_object_detected = True
                        break

                # New object is detected we assign the ID to that object
                if same_object_detected is False:
                    self.center_points[self.id_count] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, self.id_count])
                    self.id_count += 1
                    
            new_center_points = {}
            for obj_bb_id in objects_bbs_ids:
                _, _, _, _, object_id = obj_bb_id
                center = self.center_points[object_id]
                new_center_points[object_id] = center

            # Update dictionary with IDs not used removed
            self.center_points = new_center_points.copy()

            for objects_bbs in objects_bbs_ids:
                x, y, w, h, id = objects_bbs
                self.window.show_rectangle(frame, (x, y), (x + w, y + h), color, 2)
                self.window.put_label(frame, str(id),(x,y-15),(255,0,0),2)

    def run(self):
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
