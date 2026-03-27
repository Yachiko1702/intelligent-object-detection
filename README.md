# Camera Detector

This repository provides a small Python utility for real-time object detection
using the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
model and OpenCV. The code is designed to be both a command-line tool and a
library component that can be imported into larger projects.

## Features

* Works with any YOLOv8 model (nano, small, medium, large, custom weights)
* Command-line arguments for model path, camera index, confidence threshold,
  device selection and more
* Optional asynchronous capture for smoother display on slow CPUs
* Optional recording of annotated video to disk
* FPS counter and logging output
* Clean shutdown and error handling

The pretrained YOLOv8 models are trained on the COCO dataset, which
contains 80 common object categories.  Objects such as `mouse`, `cup`,
`laptop`, `tv`, `keyboard` (and many others) are recognized out of the box.
Categories not present in COCO (e.g. `pencil`, `ballpen`, `cap`) would
require training a custom model and supplying its weights via `--model`.

## Installation

Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate     # or \`venv\Scripts\activate\` on Windows
python -m pip install -r requirements.txt
```

If you plan to use a GPU, install a compatible PyTorch wheel before installing
`ultralytics` (see [PyTorch docs](https://pytorch.org/get-started/locally/)).

## Usage

By default the script will download any of the official YOLOv8
pretrained weights if you pass a recognised name such as `yolov8n`,
`yolov8s.pt`, `yolov8m`, etc.  Ultralytics automatically fetches the
weights from its online repository, so you don’t need to pre‑download
them yourself.

You’re not limited to those five keys – the `ultralytics.YOLO` class also
accepts any valid URL pointing to a `.pt` file.  For example, you could
use a model hosted on GitHub or on your own server:

```bash
python camera_detector.py --model https://example.com/myweights.pt
```

Similarly, if you train or obtain custom weights and save them locally,
simply point `--model` at the file path.  The CLI makes no distinction
between downloaded and locally stored models.

A pair of diagnostic flags can help you explore model capabilities:

* `--list-classes` prints the class index and name for the chosen model and
  then exits.  For example:

  ```bash
  python camera_detector.py --list-classes
  # 0: person
  # 1: bicycle
  # …
  ```

  This lets you quickly see what objects can be detected by the model without
  opening a camera window.

* `--list-models` prints the supported YOLOv8 pretrained keys and a brief
  description of their speed/accuracy trade‑offs:

  ```bash
  python camera_detector.py --list-models
  # yolov8n: nano – fastest, least accurate
  # yolov8s: small – good speed/accuracy balance
  # yolov8m: medium – slower but more accurate
  # yolov8l: large – higher accuracy at reduced speed
  # yolov8x: extra-large – highest accuracy, slowest
  ```

  Use this when you want a quick reminder of the available model variants.

```bash
# run with default model and camera 0
python camera_detector.py

# use a different model file or pretrained name
python camera_detector.py --model yolov8s.pt

# change camera index and minimum confidence
python camera_detector.py --cam 1 --conf 0.5

# run on GPU, optionally with async capture
python camera_detector.py --device 0 --async

# save output to MP4
python camera_detector.py --output detections.mp4
```

Press **q** in the display window to quit.

## Training Custom Models

Train your own weights using the ultralytics CLI:

```bash
yolo train data=/path/to/data.yaml model=yolov8s.pt
```

Then point the `--model` argument at the resulting `.pt` file.

## Development

* `requirements.txt` lists the runtime dependencies.
* Add unit tests under `tests/` and run them with `pytest`.
* Use the module (`import camera_detector`) to embed detection in other code.


python camera_detector.py --model yolov8s.pt --device 0 --async --output out.mp4