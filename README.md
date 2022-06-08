# Vehicle Detection using YoloV3

## Installing required packages

1. Run `pip install opencv-python` to install OpenCv.
2. Run `pip install numpy` to install Numpy.
3. Run `pip install python-dotenv` to install Dotenv.
4. Run `pip install --upgrade imutils` to install imutils.
5. Run `pip install Shapely` to install Shapely.

## Configuration

1. Create a new directory named `config` at the root `/pc2-obj-detection` directory.
2. Download [coco.names](https://opencv-tutorial.readthedocs.io/en/latest/_downloads/a9fb13cbea0745f3d11da9017d1b8467/coco.names), [yolov3.cfg](https://opencv-tutorial.readthedocs.io/en/latest/_downloads/10e685aad953495a95c17bfecd1649e5/yolov3.cfg), and [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) files and copy those files to `/pc2-obj-detection/config` directory.
3. Make a copy of `.env.example` and name it `.env` at the root `/pc2-obj-detection` directory.
4. Enter the path of the video to the `VIDEO_PATH` variable of the `.env` file.
5. Enter the frame rate of the video as an FPS value to the `VIDEO_FPS` variable of the `.env` file.
6. Set `USE_IMUTILS` variable of the `.env` file to `True` if you want to buffer frames using imutils package and to `False` otherwise.

## Running

1. Move the terminal to root `/pc2-obj-detection` directory.
2. Run command `python main.py` or `python3 main.py` to run the program.
3. The video will be paused at the first frame to mark the danger zone.
4. Click the points where vertices of the danger zone should be.
5. Then press any key to continue.
6. Press the `space bar` to pause the video and then press any key to resume.
7. Press the `esc` key to exit from the program.
