# Generic TFLite Object Detection Harness

![BirdEye](https://gainsec.com/wp-content/uploads/2025/11/birdeye.png)

## Overview
This tool provides a clean, general-purpose Python runner for **TensorFlow Lite (TFLite)** object detection models.  
It can execute inference over **live camera feeds**, **video files**, **image directories**, or **session-style datasets**, producing structured JSON reports and Markdown summaries.

All vendor-specific content and model references have been removed.  
The harness is **model-agnostic** and can be used with any `.tflite` model and associated metadata you legally own or are licensed to use.

---

## Features
- Supports **YOLO**, **SSD**, or compatible TFLite model architectures.
- Works with webcam streams, video files, image directories, or recursive session folders.
- Produces JSON detection reports per frame or per session.
- Generates an ongoing Markdown summary of inference statistics.
- Optional OpenCV visualization of real-time detections.
- Session watcher mode for continuous processing of incoming datasets.

---

## Installation
### 1. Clone and prepare environment
```bash
git clone https://github.com/GainSec/tensorflow-generic-harness
cd generic-tflite-harness
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### Alternative:
1. Create/activate a virtualenv and install prerequisites (run `./setup.sh`, which defaults to `/opt/homebrew/bin/python3.11` on macOS and `python3` on Linux; override with `PYTHON_BIN=/path/to/python ./setup.sh` if needed):
```bash
./setup.sh        # creates .venv + installs platform-specific deps
source .venv/bin/activate
```

## How to Use
2. Example webcam run:  
   ```bash
   python3 generic_tflite_harness.py --models-json assets/models/objects/models.json --assets-root assets/models/objects --model-name model_a_float16 --camera 0
   ```
3. Example session replay:  
   ```bash
   python3 generic_tflite_harness.py --models-json assets/models/objects/models.json --assets-root assets/models/objects --model-name model_a_float16 --session-root /path/to/session/root --session-stage processed
   ```

## CLI Flags
- `--models-json PATH` – path to a lawful, user-supplied `models.json`.
- `--assets-root PATH` – directory containing compatible `.tflite` or equivalent model files.
- `--model-name NAME` – model identifier from `models.json`.
- `--camera INDEX`, `--video PATH`, `--frames-dir DIR`, `--session-root DIR`, `--session-stage NAME`, and related flags behave as described above.

### Disclaimer
Use only models that you own or are explicitly licensed to use.
Do not distribute proprietary or third-party assets.
This harness is provided strictly for research, education, and lawful interoperability testing.
