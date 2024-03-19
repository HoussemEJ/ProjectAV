import re
import sys
import os

def extract_oks_scores(file_path):
    oks_scores = []
    with open(file_path, 'r') as file:
        for line in file:
            match = re.search(r"Best OKS = ([\d.]+)", line)
            if match:
                oks_score = float(match.group(1))
                oks_scores.append(oks_score)
    return oks_scores

def calculate_ap(oks_scores, thresholds):
    ap_results = {}
    for threshold in thresholds:
        TP = sum(1 for score in oks_scores if score >= threshold)
        FP = sum(1 for score in oks_scores if score > 0 and score < threshold)
        FN = sum(1 for score in oks_scores if score == 0)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        ap_results[threshold] = precision  # AP calculation based on precision at threshold

    return ap_results

def process_file(input_file_path, output_dir, thresholds):
    file_name = os.path.basename(input_file_path)
    output_file_path = os.path.join(output_dir, file_name)
    
    oks_scores = extract_oks_scores(input_file_path)
    ap_results = calculate_ap(oks_scores, thresholds)

    AP = sum(ap_results.values()) / len(ap_results)
    AP50 = ap_results[0.50]
    AP75 = ap_results[0.75]
    mAP = (AP + AP50 + AP75) / 3

    with open(output_file_path, 'w') as file:
        file.write(f"AP: {AP:.4f}\n")
        file.write(f"AP50: {AP50:.4f}\n")
        file.write(f"AP75: {AP75:.4f}\n")
        file.write(f"mAP: {mAP:.4f}\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <input_dir> <output_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    thresholds = [0.50 + 0.05*i for i in range(10)]  # [0.50, 0.55, ..., 0.95]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.txt'):
            input_file_path = os.path.join(input_dir, file_name)
            process_file(input_file_path, output_dir, thresholds)
            print(f"Processed {file_name}")
