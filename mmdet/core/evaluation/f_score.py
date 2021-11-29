import math
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple


class ScoreMetric:
    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value

        if math.isnan(value):
            raise ValueError(
                "The value of a ScoreMetric is not allowed to be NaN.")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ScoreMetric):
            return False
        return self.name == other.name and self.value == other.value

    def __repr__(self):
        return f"ScoreMetric(name=`{self.name}`, score=`{self.value}`)"

    @staticmethod
    def type():
        return "score"


class CurveMetric:
    def __init__(self, name: str, ys: List[float], xs: Optional[List[float]] = None):
        self.name = name
        self.__ys = ys
        if xs is not None:
            if len(xs) != len(self.__ys):
                raise ValueError(
                    f"Curve error must contain the same length for x and y: ({len(xs)} vs {len(self.ys)})")
            self.__xs = xs
        else:
            # if x values are not provided, set them to the 1-index of the y values
            self.__xs = list(range(1, len(self.__ys) + 1))

    @property
    def ys(self) -> List[float]:
        """
        Returns the list of floats on y-axis.
        """
        return self.__ys

    @property
    def xs(self) -> List[float]:
        """
        Returns the list of floats on x-axis.
        """
        return self.__xs

    def __repr__(self):
        return f"CurveMetric(name=`{self.name}`, ys=({len(self.ys)} values), " \
               f"xs=({len(self.xs) if self.xs is not None else 'None'} values))"

    @staticmethod
    def type():
        return "curve"


class _ResultCounters:
    def __init__(self,
                 n_false_negatives: int,
                 n_true: int,
                 n_predicted: int):
        self.n_false_negatives = n_false_negatives
        self.n_true = n_true
        self.n_predicted = n_predicted


class _Metrics:
    def __init__(self,
                 f_measure: float,
                 precision: float,
                 recall: float):
        self.f_measure = f_measure
        self.precision = precision
        self.recall = recall


class _AggregatedResults:
    def __init__(self,
                 f_measure_curve: Dict[str, List[float]],
                 precision_curve: Dict[str, List[float]],
                 recall_curve: Dict[str, List[float]],
                 all_classes_f_measure_curve: List[float],
                 best_f_measure: float,
                 best_threshold: float):
        self.f_measure_curve = f_measure_curve
        self.precision_curve = precision_curve
        self.recall_curve = recall_curve
        self.all_classes_f_measure_curve = all_classes_f_measure_curve
        self.best_f_measure = best_f_measure
        self.best_threshold = best_threshold


class _OverallResults:
    def __init__(self,
                 per_confidence: _AggregatedResults,
                 best_f_measure_per_class: Dict[str, float],
                 best_f_measure: float):
        self.per_confidence = per_confidence
        self.best_f_measure_per_class = best_f_measure_per_class
        self.best_f_measure = best_f_measure


class FMeasure:
    box_score_index = 4
    box_class_index = 5

    def __init__(self, cocoDt, cocoGt, vary_confidence_threshold: bool = False):
        confidence_range = [0.025, 1.0, 0.025]
        confidence_values = list(np.arange(*confidence_range))
        prediction_boxes_per_image = FMeasure.__prepare(cocoDt)
        ground_truth_boxes_per_image = FMeasure.__prepare(cocoGt)
        assert len(prediction_boxes_per_image) == len(
            ground_truth_boxes_per_image)
        classes = {v['id']: v['name'] for k, v in cocoGt.cats.items()}

        result = FMeasure.__evaluate_detections(
            ground_truth_boxes_per_image=ground_truth_boxes_per_image,
            predicted_boxes_per_image=prediction_boxes_per_image,
            confidence_range=confidence_range,
            classes=classes,
            img_ids=cocoGt.getImgIds()
        )
        self._f_measure = ScoreMetric(
            name="f-measure", value=result.best_f_measure)
        f_measure_per_label: Dict[str, ScoreMetric] = {}
        for class_idx, class_name in classes.items():
            f_measure_per_label[class_name] = ScoreMetric(
                name=class_name,
                value=result.best_f_measure_per_class[class_name]
            )
        self._f_measure_per_label = f_measure_per_label

        self._f_measure_per_confidence: Optional[CurveMetric] = None
        self._best_confidence_threshold: Optional[ScoreMetric] = None

        if vary_confidence_threshold:
            f_measure_per_confidence = CurveMetric(name="f-measure per confidence", xs=confidence_values,
                                                   ys=result.per_confidence.all_classes_f_measure_curve)
            best_confidence_threshold = ScoreMetric(name="Optimal confidence threshold",
                                                    value=result.per_confidence.best_threshold)
            self._f_measure_per_confidence = f_measure_per_confidence
            self._best_confidence_threshold = best_confidence_threshold

    @property
    def f_measure(self) -> ScoreMetric:
        return self._f_measure

    @property
    def f_measure_per_label(self) -> Dict[str, ScoreMetric]:
        return self._f_measure_per_label

    @property
    def f_measure_per_confidence(self) -> Optional[CurveMetric]:
        return self._f_measure_per_confidence

    @property
    def best_confidence_threshold(self) -> Optional[ScoreMetric]:
        return self._best_confidence_threshold

    @staticmethod
    def __prepare(cocoAPI) -> OrderedDict:
        new_annotations = OrderedDict()
        for image_id, bboxes in cocoAPI.imgToAnns.items():
            new_annotations[image_id] = []
            for b in bboxes:
                x1, y1, w, h = b['bbox']
                score = b['score'] if 'score' in b else 1.0
                new_annotations[image_id].append(
                    [x1, y1, x1 + w, y1 + h, score, b['category_id']])
        for image_id in cocoAPI.getImgIds():
            if image_id not in new_annotations:
                new_annotations[image_id] = []
        return new_annotations

    @staticmethod
    def __evaluate_detections(ground_truth_boxes_per_image: Dict,
                              predicted_boxes_per_image: Dict,
                              classes: Dict[int, str],
                              img_ids,
                              iou_threshold: float = 0.5,
                              confidence_range: List[float] = None) -> _OverallResults:

        best_f_measure_per_class = {}

        if confidence_range is None:
            confidence_range = [0.025, 1.0, 0.025]

        results_per_confidence = FMeasure.__get_results_per_confidence(
            ground_truth_boxes_per_image=ground_truth_boxes_per_image,
            predicted_boxes_per_image=predicted_boxes_per_image,
            classes=classes,
            confidence_range=confidence_range,
            iou_threshold=iou_threshold,
            img_ids=img_ids
        )

        best_f_measure = results_per_confidence.best_f_measure

        for class_idx, class_name in classes.items():
            best_f_measure_per_class[class_name] = max(
                results_per_confidence.f_measure_curve[class_name])

        result = _OverallResults(
            results_per_confidence,
            best_f_measure_per_class,
            best_f_measure
        )

        return result

    @staticmethod
    def __get_results_per_confidence(ground_truth_boxes_per_image: Dict,
                                     predicted_boxes_per_image: Dict,
                                     classes: Dict[int, str],
                                     confidence_range: List[float],
                                     iou_threshold: float,
                                     img_ids,
                                     all_classes_name: str = "All Classes") -> _AggregatedResults:
        result = _AggregatedResults(
            f_measure_curve={class_name: []
                             for class_idx, class_name in classes.items()},
            precision_curve={class_name: []
                             for class_idx, class_name in classes.items()},
            recall_curve={class_name: []
                          for class_idx, class_name in classes.items()},
            all_classes_f_measure_curve=[],
            best_f_measure=0.0,
            best_threshold=0.1)

        for confidence_threshold in np.arange(*confidence_range):
            result_point = FMeasure.__evaluate_classes(ground_truth_boxes_per_image=ground_truth_boxes_per_image,
                                                       predicted_boxes_per_image=predicted_boxes_per_image,
                                                       classes=classes,
                                                       iou_threshold=iou_threshold,
                                                       confidence_threshold=confidence_threshold,
                                                       img_ids=img_ids
                                                       )
            all_classes_f_measure = result_point[all_classes_name].f_measure
            result.all_classes_f_measure_curve.append(all_classes_f_measure)

            for class_idx, class_name in classes.items():
                result.f_measure_curve[class_name].append(
                    result_point[class_name].f_measure)
                result.precision_curve[class_name].append(
                    result_point[class_name].precision)
                result.recall_curve[class_name].append(
                    result_point[class_name].recall)
            if all_classes_f_measure > result.best_f_measure:
                result.best_f_measure = all_classes_f_measure
                result.best_threshold = confidence_threshold
        return result

    @staticmethod
    def __evaluate_classes(ground_truth_boxes_per_image: Dict,
                           predicted_boxes_per_image: Dict,
                           classes: Dict[int, str],
                           iou_threshold: float,
                           confidence_threshold: float, img_ids) -> Dict[str, _Metrics]:
        """
        Returns Dict of f_measure, precision and recall for each class.

        :param ground_truth_boxes_per_image: shape List[List[List[Tuple[float, str]]]]:
                a box: [x1: float, y1, x2, y2, class: str, score: float]
                boxes_per_image: [box1, box2, …]
                ground_truth_boxes_per_image: [boxes_per_image_1, boxes_per_image_2, boxes_per_image_3, …]
        :param predicted_boxes_per_image:  shape List[List[List[Tuple[float, str]]]]:
                a box: [x1: float, y1, x2, y2, class: str, score: float]
                boxes_per_image: [box1, box2, …]
                predicted_boxes_per_image: [boxes_per_image_1, boxes_per_image_2, boxes_per_image_3, …]
        :param classes:
        :param iou_threshold:
        :param confidence_threshold:
        :return: The metrics (e.g. F-measure) for each class.
            A special "All Classes" label represents the mean results across all classes.
        """
        all_classes_name = "All Classes"
        result: Dict[str, _Metrics] = {
        }

        all_classes_counters = _ResultCounters(0, 0, 0)
        for class_idx, class_name in classes.items():
            metrics, counters = FMeasure.__get_f_measure(ground_truth_boxes_per_image=ground_truth_boxes_per_image,
                                                         predicted_boxes_per_image=predicted_boxes_per_image,
                                                         class_idx=class_idx,
                                                         iou_threshold=iou_threshold,
                                                         confidence_threshold=confidence_threshold,
                                                         img_ids=img_ids)
            result[class_name] = metrics
            all_classes_counters.n_false_negatives += counters.n_false_negatives
            all_classes_counters.n_true += counters.n_true
            all_classes_counters.n_predicted += counters.n_predicted

        # for all classes
        result[all_classes_name] = FMeasure.__calculate_f_measure(
            all_classes_counters)
        return result

    @staticmethod
    def __get_f_measure(ground_truth_boxes_per_image: Dict,
                        predicted_boxes_per_image: Dict,
                        class_idx: int,
                        iou_threshold: float,
                        confidence_threshold: float,
                        img_ids: List[int]) -> Tuple[_Metrics, _ResultCounters]:
        class_ground_truth_boxes_per_image = FMeasure.__filter_class(
            ground_truth_boxes_per_image, class_idx)
        confidence_predicted_boxes_per_image = FMeasure.__filter_confidence(predicted_boxes_per_image,
                                                                            confidence_threshold)
        class_predicted_boxes_per_image = FMeasure.__filter_class(
            confidence_predicted_boxes_per_image, class_idx)
        if len(class_ground_truth_boxes_per_image) > 0:
            result_counters = FMeasure.__get_counters(
                ground_truth_boxes_per_image=class_ground_truth_boxes_per_image,
                predicted_boxes_per_image=class_predicted_boxes_per_image,
                iou_threshold=iou_threshold,
                img_ids=img_ids
            )
            result_metrics = FMeasure.__calculate_f_measure(result_counters)
            results = (result_metrics, result_counters)
        else:
            # [f_measure, precision, recall, n_false_negatives, n_true, n_predicted]
            results = (_Metrics(0.0, 0.0, 0.0), _ResultCounters(0, 0, 0))
        return results

    @staticmethod
    def __calculate_f_measure(results: _ResultCounters) -> _Metrics:
        """
        Calculates and returns precision, recall, and f-measure
        :param results: Result counters (true positives, false positives, etc)
        :return: structure containing the computed metrics
        """
        n_true_positives = results.n_true - results.n_false_negatives

        if results.n_predicted == 0:
            precision = 1.0
            recall = 0.0
        elif results.n_true == 0:
            precision = 0.0
            recall = 1.0
        else:
            precision = n_true_positives / results.n_predicted
            recall = n_true_positives / results.n_true

        f_measure = (2 * precision * recall) / \
            (precision + recall + np.finfo(float).eps)
        return _Metrics(f_measure, precision, recall)

    @staticmethod
    def __filter_class(boxes_per_image: Dict, class_idx: int) -> OrderedDict:
        """
        Filters boxes to only keep members of one class
        :param boxes_per_image:
        :param class_name:
        :return:
        """
        filtered_boxes_per_image = OrderedDict()
        for image_id, boxes in boxes_per_image.items():
            filtered_boxes = []
            for box in boxes:
                if box[FMeasure.box_class_index] == class_idx:
                    filtered_boxes.append(box)
            filtered_boxes_per_image[image_id] = filtered_boxes
        return filtered_boxes_per_image

    @staticmethod
    def __filter_confidence(boxes_per_image: Dict, confidence_threshold: float) -> OrderedDict:
        """
        Filters boxes to only keep ones with higher confidence than a given confidence threshold
        :param boxes_per_image: shape List[List[[Tuple[float, str]]]:
                a box: [x1: float, y1, x2, y2, class: str, score: float]
                boxes_per_image: [box1, box2, …]
        :param confidence_threshold:
        :return:
        """
        filtered_boxes_per_image = OrderedDict()
        for image_id, boxes in boxes_per_image.items():
            filtered_boxes = []
            for box in boxes:
                if float(box[FMeasure.box_score_index]) > confidence_threshold:
                    filtered_boxes.append(box)
            filtered_boxes_per_image[image_id] = filtered_boxes
        return filtered_boxes_per_image

    @staticmethod
    def __get_counters(ground_truth_boxes_per_image: Dict,
                       predicted_boxes_per_image: Dict,
                       iou_threshold: float, img_ids) -> _ResultCounters:
        n_false_negatives = 0
        n_true = 0
        n_predicted = 0

        for image_index in img_ids:
            ground_truth_boxes = ground_truth_boxes_per_image[image_index]
            predicted_boxes = predicted_boxes_per_image[image_index]
            n_true += len(ground_truth_boxes)
            n_predicted += len(predicted_boxes)
            if len(predicted_boxes) > 0:
                if len(ground_truth_boxes) > 0:
                    iou_matrix = FMeasure.__get_iou_matrix(
                        ground_truth_boxes, predicted_boxes)
                    n_false_negatives += FMeasure.__get_n_false_negatives(
                        iou_matrix, iou_threshold)
            else:
                n_false_negatives += len(ground_truth_boxes)
        return _ResultCounters(n_false_negatives, n_true, n_predicted)

    @staticmethod
    def __get_n_false_negatives(iou_matrix: np.ndarray, iou_threshold: float) -> int:
        n_false_negatives = 0
        for row in iou_matrix:
            if max(row) < iou_threshold:
                n_false_negatives += 1
        for column in np.rot90(iou_matrix):
            indices = np.where(column > iou_threshold)
            n_false_negatives += max(len(indices[0]) - 1, 0)
        return n_false_negatives

    @staticmethod
    def __get_iou_matrix(boxes1: List, boxes2: List) -> np.ndarray:
        """
        Constructs an iou matrix of shape [num_ground_truth_boxes, num_predicted_boxes] each cell(x,y) in the iou matrix
        contains the intersection over union of ground truth box(x) and predicted box(y)
        An iou matrix corresponds to a single image
        :param boxes1: shape List[List[List[Tuple[float, str]]]]:
                a box: [x1: float, y1, x2, y2, class: str, score: float]
                boxes_per_image: [box1, box2, …]
                boxes1: [boxes_per_image_1, boxes_per_image_2, boxes_per_image_3, …]
        :param boxes2:  shape List[List[List[Tuple[float, str]]]]:
                a box: [x1: float, y1, x2, y2, class: str, score: float]
                boxes_per_image: [box1, box2, …]
                boxes2: [boxes_per_image_1, boxes_per_image_2, boxes_per_image_3, …]
        :return: the iou matrix
        """
        matrix = np.array(
            [[FMeasure.__bounding_box_intersection_over_union(box1, box2) for box2 in boxes2] for box1 in boxes1])
        return matrix

    @staticmethod
    def __bounding_box_intersection_over_union(box1: List[int], box2: List[int]) -> float:
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        :param box1: (x1, y1, x2, y2)
        :param box2: (x1, y1, x2, y2)
        :raises: value error in case iou outside of [0.0, 1.0]
        :return: intersection-over-union of box1 and box2
        """

        x_left, x_right, y_bottom, y_top = FMeasure.__intersection_box(
            box1, box2)

        if x_right <= x_left or y_bottom <= y_top:
            iou = 0.0
        else:
            intersection_area = (x_right - x_left) * (y_bottom - y_top)
            bb1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
            bb2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
            union_area = float(bb1_area + bb2_area - intersection_area)
            if union_area == 0:
                iou = 0.0
            else:
                iou = intersection_area / union_area
        if iou < 0.0 or iou > 1.0:
            raise ValueError(
                f"intersection over union should be in range [0,1], actual={iou}")
        return iou

    @staticmethod
    def __intersection_box(box1: List[int], box2: List[int]) -> List[int]:
        """
        Calculate the intersection rectangle of two bounding boxes
        :param box1: (x1, y1, x2, y2)
        :param box2: (x1, y1, x2, y2)
        :return: (x_left, x_right, y_bottom, y_top)
        """
        x_left = max(box1[0], box2[0])
        y_top = max(box1[1], box2[1])
        x_right = min(box1[2], box2[2])
        y_bottom = min(box1[3], box2[3])
        return [x_left, x_right, y_bottom, y_top]
