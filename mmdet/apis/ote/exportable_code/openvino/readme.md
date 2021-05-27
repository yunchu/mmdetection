# OTE Detection for OpenVINO models

## Installation

The sample is tested with Python 3.6.9 and OpenVINO 2021.3.
OpenVINO and OpenCV can be installed with `pip install -r requirements.txt`
It is recommended to make a virtual environment, for example with Anaconda. See https://docs.anaconda.com/anaconda/user-guide/getting-started/.

The system PATH needs to be modified for OpenVINO. Please follow the instructions on
https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_pip.html

| DISCLAIMER: |
| ----------- |
This repo is tested based on OpenVINO version 2021.3. The installation may not work, or the stability of the code might be affected with other versions.

### Model
The labels and model can be found in the `model` directory. A new model that is exported from OTE can be placed in this folder.

### Starting the sample

You can choose to either supply a video file or start the script without specifying anything. If nothing is specified by default the first camera availabe on the system is selected.
You can start the sample with a video like this: `python detector.py --file path/to/video.mp4`. The sample can be quit by pressing the Q key.

## Troubleshooting

If you get an `ImportError` it is likely that OpenVINO is not initialized.
Please follow the instructions on
https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_pip.html
to set the PATH and try the sample again.
