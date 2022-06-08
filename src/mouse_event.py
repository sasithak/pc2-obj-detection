import cv2


def left_click_detect(event, x, y, flags, tracker):
    if (event == cv2.EVENT_LBUTTONDOWN):
        tracker.addPoint((x, y))
