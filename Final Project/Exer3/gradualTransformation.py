import cv2
import numpy as np
import math

def rotate_video(input_path, output_path, final_angle=90, preserve_content=False):
    """
    Rotate a video gradually from 0 to final_angle degrees.
    
    Args:
        input_path (str): Path to input video file
        output_path (str): Path to save output video
        final_angle (float): Final rotation angle in degrees
        preserve_content (bool): If True, scale the frame to ensure no content is clipped
    """
    # Step 1: Open input video and get properties
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Step 2: Final rotation angle defined in function parameter
    
    # Step 3: Center of rotation
    center = (width / 2, height / 2)
    
    # Prepare output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Step 4: Loop through each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Step 5a: Calculate rotation angle for current frame
        if total_frames > 1:
            current_angle = final_angle * (frame_count / (total_frames - 1))
        else:
            current_angle = 0
        
        # For the challenge: Calculate scaling factor to ensure no content is clipped
        if preserve_content:
            # Calculate the scaling factor needed
            # When a rectangle is rotated, its bounding box becomes larger
            # The scale factor is 1/max(abs(cos(angle)) + abs(sin(angle)))
            angle_rad = math.radians(current_angle)
            scale_factor = 1.0 / max(
                abs(math.cos(angle_rad)) + abs(math.sin(angle_rad)) * height / width,
                abs(math.cos(angle_rad)) * height / width + abs(math.sin(angle_rad))
            )
        else:
            scale_factor = 1.0
        
        # Step 5b: Construct rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, current_angle, scale_factor)
        
        # Step 5c: Apply rotation to the frame
        rotated_frame = cv2.warpAffine(frame, rotation_matrix, (width, height))
        
        # Write the rotated frame to output
        out.write(rotated_frame)
        
        frame_count += 1
        
        # Optional: Display progress
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Step 7: Release resources
    cap.release()
    out.release()
    print(f"Video rotation complete. Output saved to {output_path}")

# Example usage
if __name__ == "__main__":
    # Example with a sample video file
    input_video = "Final_Exercise.mp4"  # Replace with your video file path
    
    # Regular rotation (may clip corners)
    output_video = "rotated_output.mp4"
    rotate_video(input_video, output_video, final_angle=360)
    
    # Challenge solution: Scale down to preserve all content
    output_video_preserved = "rotated_preserved_output.mp4"
    rotate_video(input_video, output_video_preserved, final_angle=360, preserve_content=True)