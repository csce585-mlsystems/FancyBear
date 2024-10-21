import cv2
import numpy as np
import sys
import os
from capture_video import capture_video
from preprocessing_video import preprocess_frames
from segmentation_video import segment_frame_deeplabv3

#sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main(video_source = 0):
    capture = capture_video(video_source)
    if capture is None:
        return
    
    # Create background subtractor
    #back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    
    while True:
        #print("Running While Loop")
        ret, frame = capture.read()
        if not ret:
            break
        
        #Preprocess the current frame
        preprocessed_frame, edges = preprocess_frames(frame)

        #Segment Swimmer from Background using DeepLabV3
        swimmer_mask = segment_frame_deeplabv3(frame)

        #Apply the segmentation mask to the original frame
        #masked_frame = apply_mask_to_frame(frame, swimmer_mask)

        #Show original, preprocessed, and segmented frames
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Preprocessed Frame', preprocessed_frame)
        cv2.imshow('Swimmer Segmentation', swimmer_mask)

        #Press 'q' to exit
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main("data\TrialRunSwimmingVideo_1.mp4") #provide video source
