import cv2
import os
import argparse

def create_video_from_images(folder_path, output_video_path, fps=10):
    # Get all image file names sorted
    images = sorted([img for img in os.listdir(folder_path) if img.endswith(".jpg") or img.endswith(".png")])
    if not images:
        print(f"No images found in the folder {folder_path}.")
        return
    
    # Get the path of the first image to extract the frame dimensions
    frame = cv2.imread(os.path.join(folder_path, images[0]))
    height, width, layers = frame.shape
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID' if you prefer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for image in images:
        img_path = os.path.join(folder_path, image)
        frame = cv2.imread(img_path)
        out.write(frame)  # Write out the frame
    
    out.release()  # Release everything when job is finished
    print(f"Video saved as {output_video_path}.")

def main():
    parser = argparse.ArgumentParser(description="Create a video from a folder of images.")
    parser.add_argument("folder_path", type=str, help="The path to the folder containing images.")
    parser.add_argument("output_video_path", type=str, help="The path where the output video should be saved.")
    
    args = parser.parse_args()
    
    # Create video from images in the specified folder
    create_video_from_images(args.folder_path, args.output_video_path)

if __name__ == "__main__":
    main()
