import cv2
import numpy as np

def apply_moving_filter(input_path, output_path, kernel_size=21, blur_region_width=100, use_gaussian=False, smooth_transition=False):
    """
    Apply a moving blur filter across video frames horizontally.
    
    Parameters:
    -----------
    input_path : str
        Path to the input video file
    output_path : str
        Path to save the output video
    kernel_size : int
        Size of the blur kernel (e.g., 21 for 21x21 blur)
    blur_region_width : int
        Width of the region where blur is applied
    use_gaussian : bool
        If True, uses Gaussian blur instead of box blur
    smooth_transition : bool
        If True, creates a smooth transition at the edges of the blur region
    """
    # 1. Open input video and prepare output writer
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    
    # 2 & 3. Kernel size and frame width are already obtained
    
    # 4. Loop through each frame
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 5a. Calculate horizontal center position of blur effect
        cx = int((width / total_frames) * frame_count)
        
        # 5b. Create a copy of original frame
        output_frame = frame.copy()
        
        # 5c. Apply blur to the entire original frame
        if use_gaussian:
            blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
        else:
            blurred_frame = cv2.blur(frame, (kernel_size, kernel_size))
        
        # 5d. Define the region for blur effect
        x_start = max(0, int(cx - blur_region_width / 2))
        x_end = min(width, int(cx + blur_region_width / 2))
        
        # 5e. Copy blurred pixels to output frame
        if not smooth_transition:
            # Hard edge transition
            output_frame[:, x_start:x_end] = blurred_frame[:, x_start:x_end]
        else:
            # Smooth transition with alpha blending at edges
            transition_width = min(30, blur_region_width // 4)  # Width of transition area
            
            # Core region with full blur
            core_start = x_start + transition_width
            core_end = x_end - transition_width
            
            if core_start < core_end:
                output_frame[:, core_start:core_end] = blurred_frame[:, core_start:core_end]
            
            # Left transition region
            if x_start < core_start:
                for x in range(x_start, min(core_start, width)):
                    # Calculate alpha (0 at edge, 1 at core)
                    alpha = (x - x_start) / transition_width
                    output_frame[:, x] = cv2.addWeighted(frame[:, x], 1 - alpha, blurred_frame[:, x], alpha, 0)
            
            # Right transition region
            if core_end < x_end:
                for x in range(core_end, min(x_end, width)):
                    # Calculate alpha (1 at core, 0 at edge)
                    alpha = 1 - (x - core_end) / transition_width
                    output_frame[:, x] = cv2.addWeighted(frame[:, x], 1 - alpha, blurred_frame[:, x], alpha, 0)
        
        # Write the output frame
        out.write(output_frame)
        frame_count += 1
        
        # Optional: Display progress
        if frame_count % 10 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")
    
    # Release resources
    cap.release()
    out.release()
    print("Processing complete!")

if __name__ == "__main__":
    # Example usage
    input_video = "Final_Exercise.mp4"
    output_video = "output_video_with_moving_blur.mp4"
    
    # Basic usage (with box blur)
    apply_moving_filter(input_video, output_video)
    
    # Or with Gaussian blur and smooth transitions
    # apply_moving_filter(input_video, output_video, kernel_size=21, 
    #                     blur_region_width=150, use_gaussian=True, smooth_transition=True)