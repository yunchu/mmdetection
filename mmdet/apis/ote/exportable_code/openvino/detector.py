#!/usr/bin/env python

"""
detector.py: This script will take either a video file or capture footage from the first available camera on the
system and pass it through an OpenVINO model for inference. Finally the frame will be shown along with the detected
shapes. Press Q to quit the program.
"""

import json
import os.path as osp
from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

try:
    from openvino.inference_engine.ie_api import IECore
except ImportError:
    # If OpenVINO import fails, add the library dir to PATH
    # https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_pip.html
    import os
    import sys
    if "win" in sys.platform:
        try:
            library_dir = os.path.dirname(sys.executable) + "\..\Library\\bin"
            current_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{library_dir};{current_path}"
            from openvino.inference_engine.ie_api import IECore
        except ImportError:
            raise ImportError(
                "Please add the library dir to the system PATH. See https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_pip.html")
    else:
        raise ImportError(
            "Please add the library dir to the system PATH. See https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_pip.html")


__author__ = "OTE Development Team"
__copyright__ = "Intel © All rights reserved"
__credits__ = ["OTE Development Team"]
__version__ = "1.0"
__maintainer__ = "OTE Development Team"
__email__ = ""
__status__ = "Alpha"
__updated__ = "05.27.2021"


def create_label_colour_map(labels: List[Dict[str, Any]]) -> Dict[int, Tuple]:
    """
    Maps the labels in a dictionary with its color values
    The background class is ignored

    :param labels: List of dictionaries with labels from labels.json
    :return: dictionary that maps label index to color from NOUS, e.g. {1: (255,0,0,255)}
    """
    label_map = {}
    for index, label in enumerate(labels):
        label_map[index] = tuple(label["color"].values())
    return label_map


class Streamer:
    def __init__(self, fname: str):
        """
        Utility class for streaming a video file.

        :param fname: Filename of video to stream
        """
        self.frame = None

        if fname != "":
            fname = fname.replace("\\", "/")
            # capture from video fname
            self.capture = cv2.VideoCapture(fname)
        else:
            # capture from first available camera on the system
            self.capture = cv2.VideoCapture(0)

    def __del__(self):
        self.capture.release()

    def check_frame_available(self) -> bool:
        """
        Checks if a frame is available from the capture.
        :return: bool
        """
        frame_available, self.frame = self.capture.read()
        return frame_available

    def get_frame(self):
        """
        :return: latest available frame as numpy array
        """
        return self.frame


class Model:
    def __init__(self, model_path, ie=None, device="CPU", classes=None):
        self.ie = IECore() if ie is None else ie
        bin_path = osp.splitext(model_path)[0] + ".bin"
        self.net = self.ie.read_network(model_path, bin_path)

        self.device = None
        self.exec_net = None
        self.to(device)

    def to(self, device):
        if self.device != device:
            self.device = device
            self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=1)
        return self

    def unify_inputs(self, inputs):
        if not isinstance(inputs, dict):
            inputs_dict = {next(iter(self.net.input_info)): inputs}
        else:
            inputs_dict = inputs
        return inputs_dict

    def reshape(self, inputs=None, input_shapes=None):
        assert (inputs is None) != (input_shapes is None)
        if input_shapes is None:
            input_shapes = {name: data.shape for name, data in inputs.items()}
        reshape_needed = False
        for input_name, input_shape in input_shapes.items():
            blob_shape = self.net.input_info[input_name].input_data.shape
            if not np.array_equal(input_shape, blob_shape):
                reshape_needed = True
                break
        if reshape_needed:
            print(f"reshape net to {input_shapes}")
            self.net.reshape(input_shapes)
            self.exec_net = self.ie.load_network(network=self.net, device_name=self.device, num_requests=1)

    def get(self, outputs, name):
        try:
            key = self.net.get_ov_name_for_tensor(name)
            assert key in outputs, f'"{key}" is not a valid output identifier'
        except KeyError:
            if name not in outputs:
                raise KeyError(f'Failed to identify output "{name}"')
            key = name
        return outputs[key]

    def preprocess(self, inputs):
        return inputs

    def postprocess(self, outputs):
        return outputs

    def __call__(self, inputs):
        inputs = self.unify_inputs(inputs)
        inputs = self.preprocess(inputs)
        self.reshape(inputs=inputs)
        outputs = self.exec_net.infer(inputs)
        outputs = self.postprocess(outputs)
        return outputs


class Detector(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        batch_size = self.net.input_info["image"].input_data.shape[0]
        assert batch_size == 1, "Only batch 1 is supported."

    def __call__(self, inputs):
        inputs = self.unify_inputs(inputs)
        output = super().__call__(inputs)

        if "detection_out" in output:
            detection_out = output["detection_out"]
            output["labels"] = detection_out[0, 0, :, 1].astype(np.int32)
            output["boxes"] = detection_out[0, 0, :, 3:] * np.tile(inputs["image"].shape[:1:-1], 2)
            output["boxes"] = np.concatenate((output["boxes"], detection_out[0, 0, :, 2:3]), axis=1)
            del output["detection_out"]
            return output

        outs = output
        output = {}
        output = {
            "labels": self.get(outs, "labels"),
            "boxes": self.get(outs, "boxes")
        }
        valid_detections_mask = output["labels"] >= 0
        output["labels"] = output["labels"][valid_detections_mask]
        output["boxes"] = output["boxes"][valid_detections_mask]
        try:
            output["masks"] = self.get(outs, "masks")
            output["masks"] = output["masks"][valid_detections_mask]
        except RuntimeError:
            pass

        return output


def resize_image(image, size, keep_aspect_ratio=False):
    if not keep_aspect_ratio:
        resized_frame = cv2.resize(image, size)
    else:
        h, w = image.shape[:2]
        scale = min(size[1] / h, size[0] / w)
        resized_frame = cv2.resize(image, None, fx=scale, fy=scale)
    return resized_frame


def preprocess(image, target_size=(800, 800), keep_aspect_ratio_resize=False):
    orig_w, orig_h = target_size
    resized_image = resize_image(image, (orig_w, orig_h), keep_aspect_ratio_resize)
    meta = {"original_shape": image.shape,
            "target_size": target_size,
            "resized_shape": resized_image.shape}

    h, w = resized_image.shape[:2]
    if h != orig_h or w != orig_w:
        resized_image = np.pad(resized_image, ((0, orig_h - h), (0, orig_w - w), (0, 0)),
                               mode="constant", constant_values=0)
    resized_image = resized_image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
    resized_image = resized_image.reshape((1, 3, orig_h, orig_w))
    return resized_image, meta


def postprocess(outputs, meta):
    orginal_image_shape = meta["original_shape"]
    resized_image_shape = meta["resized_shape"]
    w, h = meta["target_size"]
    scale_x = orginal_image_shape[1] / resized_image_shape[1]
    scale_y = orginal_image_shape[0] / resized_image_shape[0]
    for detection in outputs["boxes"]:
        detection[0] *= scale_x
        detection[2] *= scale_x
        detection[1] *= scale_y
        detection[3] *= scale_y
    return outputs


def draw_detections(frame, detections, palette, label_names, threshold):
    size = frame.shape[:2]
    bboxes = detections["boxes"]
    labels = detections["labels"]
    for bbox, label in zip(bboxes, labels):
        score = bbox[4]
        if score > threshold:
            xmin = max(int(bbox[0]), 0)
            ymin = max(int(bbox[1]), 0)
            xmax = min(int(bbox[2]), size[1])
            ymax = min(int(bbox[3]), size[0])
            class_id = int(label)
            color = palette[class_id]
            det_label = label_names[class_id] if len(label_names) > class_id else f"#{class_id}"
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, "{} {:.1%}".format(det_label, score),
                        (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    return frame


def run(streamer: Streamer):
    """
    Starts a loop that does inference on available frames from passed streamer.
    If there are no frames available, the program will stop.

    :param streamer: Streamer instance with frames to analyze
    """
    labelfname = "model/labels.json"
    configparamfname = "model/configurable_parameters.json"
    modelfname = "model/inference_model.xml"

    # Load labels and settings
    with open(labelfname, "r") as labelfile:
        label_data = json.load(labelfile)
    label_colour_map = create_label_colour_map(label_data)
    labels = list(x["name"] for x in label_data)

    with open(configparamfname, "r") as configfile:
        config = json.load(configfile)

    # Get parameters that affect inference.
    thr = config.get("postprocessing", {}).get("confidence_threshold", {}).get("value", 0.3)

    detector = Detector(modelfname, device="CPU")

    # While a next frame is available do inference
    while streamer.check_frame_available():
        frame = streamer.get_frame()
        image, meta = preprocess(frame)
        output = detector(image)
        results = postprocess(output, meta)
        frame = draw_detections(frame, results, palette=label_colour_map, label_names=labels, threshold=thr)
        cv2.imshow("Results", frame)
        if ord("q") == cv2.waitKey(1):
            break


def main():
    print("Press Q to quit the sample.")
    arguments_parser = ArgumentParser()
    arguments_parser.add_argument(
        "--file",
        required=False,
        help="Specify a video filename. If nothing is specified, the webcam will be used.",
        default="",
    )
    arguments = arguments_parser.parse_args()
    fname = str(arguments.file)
    run(Streamer(fname))


if __name__ == "__main__":
    main()
