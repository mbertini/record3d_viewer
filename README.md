# Record3D Viewer

A viewer for sequences of JPEG images and EXR depth maps recorded with Record3D or similar devices/software.

## Features

- Synchronized display of RGB images and depth maps
- Playback controls (play/pause, forward/backward, speed)
- Depth contrast adjustment
- Histogram equalization option for improved depth visualization

## Requirements

- Python 3.7+
- OpenCV
- OpenEXR
- NumPy
- Pillow (PIL)
- Tkinter

## Installation

```bash
pip install -r requirements.txt
```

Note: OpenEXR might require additional system dependencies:
- On Linux: sudo apt-get install libopenexr-dev
- On macOS: brew install openexr
- On Windows: Use precompiled binaries from Christoph Gohlke

## Usage

```bash
python record3d_viewer.py
```

Once the application is running:
Click on "Select Folder" and choose a folder containing two subfolders:
One with JPEG files (RGB images)
One with EXR files (depth maps)
Use the playback controls to navigate through the frames

##Running with Docker
Build the image

```bash
docker build -t record3d_viewer .
```

### Run on macOS with XQuartz
1. Install XQuartz: brew install --cask xquartz
2. Configure XQuartz to accept remote connections:
   - Open XQuartz
   - Go to XQuartz → Preferences in the menu bar
   - Select the "Security" tab
   - Check the option "Allow connections from network clients"
   - Close the preferences window
3. Restart XQuartz and enable connections:
   - Quit XQuartz completely
   - Reopen XQuartz from Applications
   - Open Terminal and run: `xhost +localhost`4. Run the container:

```bash
docker run -it --rm \
  -e DISPLAY=host.docker.internal:0 \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /path/to/your/data:/data \
  record3d_viewer
```

## Expected Data Structure
The viewer expects a folder structure like this:

```
main_folder/
├── images/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── depth/
    ├── 0.exr
    ├── 1.exr
    └── ...
```

Files should be sequentially numbered starting from 0.


## Testing the Application
To test the application:

1. Clone the repository or create the record3d_viewer.py and requirements.txt files with the provided code.

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Test the application in two ways:

- Directly on your system:

```bash
python record3d_viewer.py
```

Then select a folder containing subfolders with JPEG images and EXR depth files.

- Using Docker:

```bash
docker build -t record3d_viewer .
```

Follow the instructions in this README to set up XQuartz and launch the container.


If you don't have test data, you can download some Record3D examples or create a test folder structure with sample JPEG and EXR files.