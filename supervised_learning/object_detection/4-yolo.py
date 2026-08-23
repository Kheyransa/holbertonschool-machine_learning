#!/usr/bin/env python3
"""Yolo v3 class - process_outputs, filter_boxes, non_max_suppression,
load_images"""
import numpy as np
import tensorflow.keras as K
import cv2
import glob
import os


class Yolo:
    """Uses the Yolo v3 algorithm to perform object detection"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        model_path: path to where a Darknet Keras model is stored
        classes_path: path to where the list of class names used for the
            Darknet model, listed in order of index, can be found
        class_t: float representing the box score threshold for the
            initial filtering step
        nms_t: float representing the IOU threshold for non-max suppression
        anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
            outputs: number of outputs (predictions) made by the model
            anchor_boxes: number of anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """Applies the sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        outputs: list of numpy.ndarrays containing the predictions from the
            Darknet model for a single image:
            Each output has shape (grid_height, grid_width, anchor_boxes,
                4 + 1 + classes)
        image_size: numpy.ndarray containing the image's original size
            [image_height, image_width]

        Returns: tuple of (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        image_height, image_width = image_size
        input_width = self.model.input.shape[1]
        input_height = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape

            t_xy = output[..., 0:2]
            t_wh = output[..., 2:4]
            box_confidence = self.sigmoid(output[..., 4:5])
            box_class_prob = self.sigmoid(output[..., 5:])

            box_confidences.append(box_confidence)
            box_class_probs.append(box_class_prob)

            # build grid of cell coordinates, shape (gh, gw, anchor_boxes)
            cx = np.arange(grid_width).reshape(1, grid_width, 1)
            cx = np.tile(cx, (grid_height, 1, anchor_boxes))

            cy = np.arange(grid_height).reshape(grid_height, 1, 1)
            cy = np.tile(cy, (1, grid_width, anchor_boxes))

            bx = (self.sigmoid(t_xy[..., 0]) + cx) / grid_width
            by = (self.sigmoid(t_xy[..., 1]) + cy) / grid_height

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            bw = (anchor_w * np.exp(t_wh[..., 0])) / input_width
            bh = (anchor_h * np.exp(t_wh[..., 1])) / input_height

            x1 = (bx - bw / 2) * image_width
            y1 = (by - bh / 2) * image_height
            x2 = (bx + bw / 2) * image_width
            y2 = (by + bh / 2) * image_height

            box = np.zeros((grid_height, grid_width, anchor_boxes, 4))
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

        return boxes, box_confidences, box_class_probs

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
            anchor_boxes, 4) containing the processed boundary boxes for
            each output, respectively
        box_confidences: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, 1) containing the processed box
            confidences for each output, respectively
        box_class_probs: a list of numpy.ndarrays of shape (grid_height,
            grid_width, anchor_boxes, classes) containing the processed
            box class probabilities for each output, respectively

        Returns: tuple of (filtered_boxes, box_classes, box_scores)
        """
        box_scores_list = []
        box_classes_list = []
        boxes_list = []

        for box, confidence, class_probs in zip(
                boxes, box_confidences, box_class_probs):
            scores = confidence * class_probs
            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            box_scores_list.append(class_scores.reshape(-1))
            box_classes_list.append(classes.reshape(-1))
            boxes_list.append(box.reshape(-1, 4))

        box_scores_all = np.concatenate(box_scores_list)
        box_classes_all = np.concatenate(box_classes_list)
        boxes_all = np.concatenate(boxes_list, axis=0)

        filtering_mask = box_scores_all >= self.class_t

        filtered_boxes = boxes_all[filtering_mask]
        box_classes = box_classes_all[filtering_mask]
        box_scores = box_scores_all[filtering_mask]

        return filtered_boxes, box_classes, box_scores

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
            the filtered bounding boxes
        box_classes: a numpy.ndarray of shape (?,) containing the class
            number for the class that filtered_boxes predicts, respectively
        box_scores: a numpy.ndarray of shape (?) containing the box scores
            for each box in filtered_boxes, respectively

        Returns: tuple of (box_predictions, predicted_box_classes,
            predicted_box_scores)
        """
        box_predictions_list = []
        predicted_box_classes_list = []
        predicted_box_scores_list = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)[0]

            cls_boxes = filtered_boxes[idxs]
            cls_scores = box_scores[idxs]

            # sort by score descending
            order = np.argsort(-cls_scores)
            cls_boxes = cls_boxes[order]
            cls_scores = cls_scores[order]

            x1 = cls_boxes[:, 0]
            y1 = cls_boxes[:, 1]
            x2 = cls_boxes[:, 2]
            y2 = cls_boxes[:, 3]
            areas = (x2 - x1) * (y2 - y1)

            keep = []
            indices = np.arange(len(cls_scores))

            while len(indices) > 0:
                current = indices[0]
                keep.append(current)

                if len(indices) == 1:
                    break

                rest = indices[1:]

                xx1 = np.maximum(x1[current], x1[rest])
                yy1 = np.maximum(y1[current], y1[rest])
                xx2 = np.minimum(x2[current], x2[rest])
                yy2 = np.minimum(y2[current], y2[rest])

                w = np.maximum(0, xx2 - xx1)
                h = np.maximum(0, yy2 - yy1)
                intersection = w * h

                union = areas[current] + areas[rest] - intersection
                iou = intersection / union

                indices = rest[iou <= self.nms_t]

            keep = np.array(keep)

            box_predictions_list.append(cls_boxes[keep])
            predicted_box_classes_list.append(
                np.full(len(keep), cls))
            predicted_box_scores_list.append(cls_scores[keep])

        box_predictions = np.concatenate(box_predictions_list, axis=0)
        predicted_box_classes = np.concatenate(
            predicted_box_classes_list, axis=0)
        predicted_box_scores = np.concatenate(
            predicted_box_scores_list, axis=0)

        return box_predictions, predicted_box_classes, predicted_box_scores

    @staticmethod
    def load_images(folder_path):
        """
        folder_path: a string representing the path to the folder holding
            all the images to load

        Returns: tuple of (images, image_paths)
            images: a list of images as numpy.ndarrays
            image_paths: a list of paths to the individual images in images
        """
        image_paths = glob.glob(os.path.join(folder_path, '*'))
        images = []

        for image_path in image_paths:
            image = cv2.imread(image_path)
            images.append(image)

        return images, image_paths
