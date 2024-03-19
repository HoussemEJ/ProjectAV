import cv2
import os
import argparse

def extract_frames(video_path):
    # Check if the video file exists
    if not os.path.exists(video_path):
        print(f"The file {video_path} does not exist.")
        return
    
    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)
    
    # Get the original FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Original FPS: {fps}")
    
    # Extract the video name without extension and create a directory
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.getcwd(), video_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_count = 0
    while True:
        # Read frame
        ret, frame = cap.read()
        # If frame is read correctly ret is True
        if not ret:
            break
        
        # Save frame to disk
        output_frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(output_frame_path, frame)
        print(f"Saved {output_frame_path}")
        
        frame_count += 1
    
    # When everything done, release the capture
    cap.release()
    print(f"Frames extracted and saved to {output_dir}.")

def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Extract frames from a video and save them to a folder.")
    parser.add_argument("video_path", type=str, help="The path to the video file from which to extract frames.")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Extract frames using the provided video path
    extract_frames(args.video_path)

if __name__ == "__main__":
    main()
