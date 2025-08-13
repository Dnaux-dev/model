import cv2
import time

def test_video_file():
    """Test if the video file can be opened and processed"""
    video_source = "scene 2.mp4"
    
    print(f"Testing video file: {video_source}")
    
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"❌ Error: Could not open video source: {video_source}")
        return False
    
    print(f"✅ Successfully opened video source: {video_source}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"📹 Video Properties:")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {frame_count}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Duration: {frame_count/fps:.2f} seconds")
    
    # Test reading a few frames
    frame_num = 0
    while frame_num < 10:  # Test first 10 frames
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Error reading frame {frame_num}")
            break
        
        print(f"✅ Frame {frame_num}: {frame.shape}")
        frame_num += 1
    
    cap.release()
    print(f"✅ Video file test completed successfully!")
    return True

if __name__ == "__main__":
    test_video_file() 