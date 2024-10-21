import cv2

#Video capture
def capture_video(video_source = 0):
    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        print("Error: Cannot Open Video Source.")
        return None
    return capture