import cv2 as cv
import numpy as np

from src.colors import RED
from src.mouse_event import *


class Window:
    def __init__(self, win_name):
        self.win_name = win_name
        self.font_face = cv.FONT_HERSHEY_SIMPLEX
        self.font_scale = 1
        self.label_font_scale = 0.5

    def show_frame(self, frame):
        cv.imshow(self.win_name, frame)

    def show_rectangle(self, frame, pt1, pt2, color, thickness):
        cv.rectangle(frame, pt1, pt2, color, thickness)

    def show_danger_zone(self, frame, points):
        cv.polylines(frame, np.array([points]), True, RED, 2)

    def put_text(self, frame, text, org, color, thickness):
        cv.putText(frame, text, org, self.font_face,
                   self.font_scale, color, thickness)

    def put_label(self, frame, text, org, color, thickness):
        cv.putText(frame, text, org, self.font_face,
                   self.label_font_scale, color, thickness)

    def put_overlay(self, text):
        cv.displayOverlay(self.win_name, text)

    def setMouseCallback(self, tracker):
        cv.setMouseCallback(self.win_name, left_click_detect, tracker)

    def close(self):
        cv.destroyAllWindows()
