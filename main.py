import os
from dotenv import load_dotenv
from src.tracker import Tracker

load_dotenv(".env")

if __name__ == "__main__":
    video_path = os.getenv("VIDEO_PATH")
    fps = int(os.getenv("VIDEO_FPS"))
    use_imutils = os.getenv("USE_IMUTILS") == "True"
    tracker = Tracker(video_path, fps, use_imutils)
    tracker.run()
