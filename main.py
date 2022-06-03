import os
from dotenv import load_dotenv
from src.tracker import Tracker

load_dotenv(".env")

if __name__ == "__main__":
    useImUtils = True
    tracker = Tracker(os.getenv("VIDEO_PATH"), useImUtils)
    tracker.run()
