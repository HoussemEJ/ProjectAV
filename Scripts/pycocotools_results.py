import sys
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def evaluate_coco(gt_file, pred_file):
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(pred_file)

    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <path_to_ground_truth_annotations> <path_to_predictions>")
        sys.exit(1)
    
    gt_file = sys.argv[1]
    pred_file = sys.argv[2]

    evaluate_coco(gt_file, pred_file)
