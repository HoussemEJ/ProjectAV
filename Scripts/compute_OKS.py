import json
import numpy as np
import os
import sys

def load_ground_truth(gt_path):
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    gt_keypoints = {}
    annotations_info = {}
    for ann in gt_data['annotations']:
        keypoints = np.array(ann['keypoints']).reshape(-1, 3)
        gt_keypoints[ann['image_id']] = keypoints
        annotations_info[ann['image_id']] = {'area': ann['area'], 'bbox': ann['bbox']}
    return gt_keypoints, annotations_info

def calculate_oks(gt_kpts, pred_kpts, sigma, area):
    squared_distances = np.sum((gt_kpts[:, :2] - pred_kpts[:, :2]) ** 2, axis=1)
    visibility = gt_kpts[:, 2] > 0
    scale = np.sqrt(area) if area > 0 else 1
    oks = np.sum(np.exp(-squared_distances / (2 * (sigma ** 2) * (scale ** 2))) * visibility) / np.sum(visibility)
    return oks

def main(ground_truth_path, predictions_path, output_file_path):
    gt_keypoints, annotations_info = load_ground_truth(ground_truth_path)
    prediction_files = [f for f in os.listdir(predictions_path) if 'keypoints' in f]
    sigma = np.array([0.025, 0.025, 0.079, 0.079, 0.072, 0.062, 0.072, 0.062])
    total_oks = 0
    non_zero_oks_count = 0
    frame_count = 0

    with open(output_file_path, 'w') as output_file:
        for pred_file in prediction_files:
            frame_id = int(pred_file.split('_')[1]) + 1
            if frame_id not in annotations_info:
                print(f"Error: Frame ID {frame_id} not found in ground truth.")
                continue
            area = annotations_info[frame_id]['area']
            with open(os.path.join(predictions_path, pred_file), 'r') as f:
                predictions = json.load(f)
            
            best_oks = 0
            best_pred = None
            for person in predictions:
                pred_kpts = np.array(person['keypoints']).reshape(-1, 3)
                oks = calculate_oks(gt_keypoints[frame_id], pred_kpts, sigma, area)
                oks_rounded = round(oks, 4)  # Round the OKS value to four decimal places
                if oks_rounded > best_oks:
                    best_oks = oks_rounded
                    best_pred = person
            if best_oks > 0:
                non_zero_oks_count += 1
            
            output_file.write(f"Frame {frame_id}: Best OKS = {best_oks:.4f} for person_id = {best_pred['person_id'] if best_pred else 'N/A'}\n")
            total_oks += best_oks
            frame_count += 1
            
        print(non_zero_oks_count)
        avg_oks = total_oks / frame_count if frame_count > 0 else 0
        avg_non_zero_oks = total_oks / non_zero_oks_count if non_zero_oks_count > 0 else 0
        non_zero_ratio = non_zero_oks_count / frame_count if frame_count > 0 else 0
        output_file.write(f"\nTotal OKS for all frames: {total_oks}\n")
        output_file.write(f"Average OKS for all frames: {avg_oks}\n")
        output_file.write(f"Average non-zero OKS for all frames: {avg_non_zero_oks}\n")
        output_file.write(f"Non-zero OKS count / Total frame count ratio: {non_zero_ratio:.4f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <path_to_ground_truth.json> <path_to_predictions_directory> <output_file_path>")
        sys.exit(1)
    ground_truth_path = sys.argv[1]
    predictions_path = sys.argv[2]
    output_file_path = sys.argv[3]
    main(ground_truth_path, predictions_path, output_file_path)

