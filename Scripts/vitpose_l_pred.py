import argparse
import os
import subprocess
import time

def main(img_root, out_img_root):
    counter = 0
    image_files = [f for f in os.listdir(img_root) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sorted_image_files = sorted(image_files)
    total_images = len(sorted_image_files)
    
    start_time = time.time()  # Track the overall start time
    times = []  # List to store processing times for each image
    
    for filename in sorted_image_files:
        img_start_time = time.time()  # Start time for this image
        
        # Increment the counter and print progress
        counter += 1
        print(f"Processing image {counter} of {total_images}...")

        # Construct and execute the command
        command = [
                'python3', 'demo/top_down_img_demo_with_mmdet.py',
                'demo/mmdetection_cfg/yolov3_d53_320_273e_coco.py',
                'https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth',
                'configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/ViTPose_large_coco_256x192.py',
                'vitpose-l.pth',
                '--img-root', img_root,
                '--img', filename,
                '--out-img-root', out_img_root
        ]
        
        # Run the command without printing its output
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        
        # Calculate the elapsed time for this image
        img_elapsed_time = time.time() - img_start_time
        times.append(img_elapsed_time)
        
        # Calculate average time per image and estimate remaining time
        avg_time_per_img = sum(times) / len(times)
        remaining_time = avg_time_per_img * (total_images - counter)
        
        # Print the progress and estimated remaining time
        print(f"Completed {counter}/{total_images}. Estimated time remaining: {remaining_time/60:.2f} minutes.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images in a directory with a specified command.")
    parser.add_argument("img_root", type=str, help="Directory containing images to process.")
    parser.add_argument("out_img_root", type=str, help="Directory where processed images will be saved.")
    
    args = parser.parse_args()
    
    main(args.img_root, args.out_img_root)
