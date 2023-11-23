import unittest
from script import calculate_iou, load_bboxes_from_directory, calculate_precision_recall_f1, calculate_ap

class TestFaceDetectionFunctions(unittest.TestCase):

    def test_calculate_iou(self):
        # Test cases for calculate_iou function
        # Test 1: Intersection between two overlapping boxes
        box1 = [5, 5.5, 10, 10]
        box2 = [5, 5, 10, 10]
        iou = calculate_iou(box1, box2)
        self.assertAlmostEqual(iou, 0.9, places=2)  # Replace expected IoU value
        
        box1 = [0, 0, 10, 10]
        box2 = [5, 5, 10, 10]
        iou = calculate_iou(box1, box2)
        self.assertAlmostEqual(iou, 0.14, places=2)  # Replace expected IoU value

    def test_calculate_precision_recall_f1(self):
        # Test cases for calculate_precision_recall_f1 function
        # Prepare mock predicted and ground truth bounding boxes
        pred_bboxes = {'image1': [[0, 0, 10, 10], [20, 20, 30, 30]], 'image2': [[5, 5, 15, 15]]}
        gt_bboxes = {'image1': [[0, 0, 10, 10]], 'image2': [[5, 5, 15, 15], [25, 25, 35, 35]]}
        
        # Call calculate_precision_recall_f1
        precision, recall, f1_score = calculate_precision_recall_f1(pred_bboxes, gt_bboxes)
        # Add assertions to check precision, recall, and F1-score
        self.assertGreater(precision, 0.5)
        self.assertGreater(recall, 0.6)
        self.assertGreater(f1_score, 0.6)

    def test_calculate_ap(self):
        # Test cases for calculate_ap function
        # Prepare mock predicted and ground truth bounding boxes
        pred_bboxes = {'image1': [[0, 0, 10, 10, 0.99], [20, 20, 30, 30, 0.98]], 'image2': [[5, 5, 15, 15, 0.97]]}
        gt_bboxes = {'image1': [[0, 0, 10, 10]], 'image2': [[5, 5, 15, 15], [25, 25, 35, 35]]}
        
        # Call calculate_ap
        iou_threshold = 0.5
        ap = calculate_ap(pred_bboxes, gt_bboxes, iou_threshold)
        self.assertGreaterEqual(ap, 0.0)

if __name__ == '__main__':
    unittest.main()
