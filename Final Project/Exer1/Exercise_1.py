import cv2
import numpy as np
import math

def process_video_linear_contrast(input_path, output_path, min_alpha=0.8, max_alpha=1.5):
    """
    Process a video by converting it to grayscale and applying temporally varying contrast.
    
    Args:
        input_path: Path to the input video file
        output_path: Path to save the output video
        min_alpha: Starting contrast factor (lower value = lower contrast)
        max_alpha: Ending contrast factor (higher value = higher contrast)
    """
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' depending on your system
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate progress (0 to 1)
        progress = frame_index / total_frames if total_frames > 0 else 0
        
        # Interpolate contrast factor
        alpha = min_alpha + (max_alpha - min_alpha) * progress
        
        # Calculate mean pixel value
        mean = np.mean(gray)
        
        # Apply contrast adjustment: new_pixel = alpha * (pixel - mean) + mean
        adjusted = alpha * (gray.astype(np.float32) - mean) + mean
        
        # Clip values to valid range and convert back to uint8
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        # Write the processed frame
        out.write(adjusted)
        
        frame_index += 1
        
        # Optional: Display progress
        if frame_index % 30 == 0:
            print(f"Processing: {frame_index}/{total_frames} frames - {int(progress*100)}%")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Processing complete. Output saved to {output_path}")


def process_video_pulsating_contrast(input_path, output_path, min_alpha=0.8, max_alpha=1.5, cycles=3):
    """
    Process a video with pulsating contrast following a sine wave pattern.
    
    Args:
        input_path: Path to the input video file
        output_path: Path to save the output video
        min_alpha: Minimum contrast factor
        max_alpha: Maximum contrast factor
        cycles: Number of complete sine wave cycles over the video duration
    """
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' depending on your system
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    frame_index = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate progress (0 to 1)
        progress = frame_index / total_frames if total_frames > 0 else 0
        
        # Calculate pulsating contrast using sine wave
        # Oscillate between min_alpha and max_alpha
        # The sine function returns values between -1 and 1, so we adjust to our range
        alpha_range = max_alpha - min_alpha
        alpha_mid = min_alpha + alpha_range / 2
        alpha = alpha_mid + (alpha_range / 2) * math.sin(2 * math.pi * cycles * progress)
        
        # Calculate mean pixel value
        mean = np.mean(gray)
        
        # Apply contrast adjustment: new_pixel = alpha * (pixel - mean) + mean
        adjusted = alpha * (gray.astype(np.float32) - mean) + mean
        
        # Clip values to valid range and convert back to uint8
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        # Write the processed frame
        out.write(adjusted)
        
        frame_index += 1
        
        # Optional: Display progress
        if frame_index % 30 == 0:
            print(f"Processing: {frame_index}/{total_frames} frames - {int(progress*100)}%")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Processing complete. Output saved to {output_path}")


def process_video_with_clahe(input_path, output_path, min_clip_limit=1.0, max_clip_limit=4.0):
    """
    Process a video using adaptive histogram equalization (CLAHE) with changing clip limits.
    
    Args:
        input_path: Path to the input video file
        output_path: Path to save the output video
        min_clip_limit: Starting CLAHE clip limit
        max_clip_limit: Ending CLAHE clip limit
    """
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' depending on your system
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)
    
    # Create CLAHE object (we'll update its parameters for each frame)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    frame_index = 0
    prev_clip_limit = min_clip_limit  # For temporal smoothing
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate progress (0 to 1)
        progress = frame_index / total_frames if total_frames > 0 else 0
        
        # Interpolate CLAHE clip limit with temporal smoothing
        target_clip_limit = min_clip_limit + (max_clip_limit - min_clip_limit) * progress
        # Apply temporal smoothing (exponential moving average)
        smoothing_factor = 0.1  # Lower means more smoothing
        clip_limit = smoothing_factor * target_clip_limit + (1 - smoothing_factor) * prev_clip_limit
        prev_clip_limit = clip_limit
        
        # Update CLAHE parameters
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        
        # Apply CLAHE to the grayscale image
        adjusted = clahe.apply(gray)
        
        # Write the processed frame
        out.write(adjusted)
        
        frame_index += 1
        
        # Optional: Display progress
        if frame_index % 30 == 0:
            print(f"Processing: {frame_index}/{total_frames} frames - {int(progress*100)}%")
    
    # Release resources
    cap.release()
    out.release()
    print(f"Processing complete. Output saved to {output_path}")


# Example usage
if __name__ == "__main__":
    input_video = "Final_Exercise.mp4"  # Replace with your input video path
    
    # Example 1: Linear contrast adjustment
    process_video_linear_contrast(input_video, "output_linear.mp4")
    
    # Example 2: Pulsating contrast (challenge solution)
    process_video_pulsating_contrast(input_video, "output_pulsating.mp4", cycles=5)
    
    # Example 3: CLAHE with temporal adaptation (advanced alternative)
    process_video_with_clahe(input_video, "output_clahe.mp4")