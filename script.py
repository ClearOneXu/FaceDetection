import os
import numpy as np
import argparse
import sys

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    box1 and box2 should be in the format [x, y, width, height].
    """
    # Extract coordinates of the boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate coordinates of intersection area
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0  # No intersection, IoU is 0

    # Calculate area of intersection rectangle
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate area of both bounding boxes
    box1_area = w1 * h1
    box2_area = w2 * h2

    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def load_bboxes_from_directory(directory):
    """
    Load bounding boxes from text files in the specified directory.
    Each line in a text file represents a bounding box in the format: x y width height
    """
    bboxes = {}
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as file:
            lines = file.readlines()
            bounding_boxes = []
            for line in lines:
                values = line.strip().split()
                if len(values) < 4: continue
                bounding_boxes.append([float(val) for val in values])
            bboxes[filename.split('.')[0]] = bounding_boxes
    return bboxes

def calculate_ap(pred_bboxes, gt_bboxes, iou_threshold=0.5):
    """
    Calculate Average Precision (AP) for predicted bounding boxes compared to ground truth.
    """
    # Sort predictions by confidence score (if available)
    # For simplicity, assume confidence scores are part of the predictions
    for key in pred_bboxes:
        pred_bboxes[key] = sorted(pred_bboxes[key], key=lambda x: x[4], reverse=True)

    average_precisions = []
    for filename, pred_boxes in pred_bboxes.items():
        true_positives = np.zeros(len(pred_boxes))
        false_positives = np.zeros(len(pred_boxes))
        total_true_boxes = len(gt_bboxes[filename])
        detected_gt_boxes = np.zeros(len(gt_bboxes[filename]))

        for i, pred_box in enumerate(pred_boxes):
            iou_max = -np.inf
            for j, gt_box in enumerate(gt_bboxes[filename]):
                iou = calculate_iou(pred_box[:4], gt_box)
                if iou > iou_max:
                    iou_max = iou
                    max_index = j

            if iou_max >= iou_threshold:
                if not detected_gt_boxes[max_index]:
                    true_positives[i] = 1
                    detected_gt_boxes[max_index] = True  # Mark ground truth box as detected
                else:
                    false_positives[i] = 1
            else:
                false_positives[i] = 1

        # Compute precision and recall at each step
        cumul_true_positives = np.cumsum(true_positives)
        cumul_false_positives = np.cumsum(false_positives)
        precision = cumul_true_positives / (cumul_true_positives + cumul_false_positives + np.finfo(float).eps)
        recall = cumul_true_positives / total_true_boxes

        # Compute Average Precision (AP) using precision-recall curve (with interpolation)
        ap = 0
        for i in range(0, len(recall) - 1):
            ap += (recall[i + 1] - recall[i]) * precision[i + 1]

        average_precisions.append(ap)

    mean_average_precision = np.mean(average_precisions)
    return mean_average_precision

def calculate_precision_recall_f1(pred_bboxes, gt_bboxes, iou_threshold=0.5):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for filename, pred_boxes in pred_bboxes.items():
        for pred_box in pred_boxes:
            iou_max = -np.inf
            for gt_box in gt_bboxes[filename]:
                iou = calculate_iou(pred_box[:4], gt_box)
                if iou > iou_max:
                    iou_max = iou

            if iou_max >= iou_threshold:
                true_positives += 1
            else:
                false_positives += 1

        false_negatives += len(gt_bboxes[filename]) - true_positives

    precision = true_positives / (true_positives + false_positives + np.finfo(float).eps)
    recall = true_positives / (true_positives + false_negatives + np.finfo(float).eps)
    f1_score = 2 * (precision * recall) / (precision + recall + np.finfo(float).eps)

    return precision, recall, f1_score


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('data', default="./p_data/photos/")
    parser.add_argument('-df', '--detect_face', action="store_true", default=True, help='detect face')
    #parser.add_argument('-d', '--data', default="./p_data/photos/")
    parser.add_argument('-m', '--trained_model', default='./Pytorch_Retinaface/weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--save_folder', default='./p_data/pred/', type=str, help='Dir to save txt results')
    parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')

    parser.add_argument('-e', '--eval', action="store_true", default=False)
    parser.add_argument('-p', '--pred', default="./p_data/pred/")
    parser.add_argument('-g', '--gt', default="./p_data/ground_truth/")

    args = parser.parse_args()

    if args.detect_face:
        sys.path.append('./Pytorch_Retinaface/')
        from detect_face import detect_face
        detect_face(args)
    
    if args.eval:
        # Load predicted and ground truth bounding boxes
        pred_bboxes = load_bboxes_from_directory(args.pred)
        gt_bboxes = load_bboxes_from_directory(args.gt)

        # Calculate AP
        iou_threshold = 0.5
        precision, recall, f1_score = calculate_precision_recall_f1(pred_bboxes, gt_bboxes, iou_threshold)
        ap = calculate_ap(pred_bboxes, gt_bboxes, iou_threshold)

        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-score: {f1_score:.4f}")
        print(f"Average Precision (AP) with IoU threshold {iou_threshold}: {ap:.4f}")
