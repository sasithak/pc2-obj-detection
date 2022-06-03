import os
from dotenv import load_dotenv
from src.tracker import Tracker

load_dotenv(".env")

if __name__ == "__main__":
    video_path = os.getenv("VIDEO_PATH")
    use_imutils = os.getenv("USE_IMUTILS") == "True"
    tracker = Tracker(video_path, use_imutils)
    tracker.run()
