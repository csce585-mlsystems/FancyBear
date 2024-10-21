import cv2
import numpy as np

#Preprocessing: Denoise, Contrast Enhancement, and Color Correction
def preprocess_frames(frame):
    #1. Denoising
    denoised_frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    #2. Color Correction (White Balance Adjustment)
    lab = cv2.cvtColor(denoised_frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    corrected_frame = cv2.merge((l, a, b))
    color_corrected_frame = cv2.cvtColor(corrected_frame, cv2.COLOR_LAB2BGR)

    #3. Covert to grayscale after color correction
    gray_frame = cv2.cvtColor(color_corrected_frame, cv2.COLOR_BGR2GRAY)
    
    #4. Histogram Equalization (Enhances Contrast)
    equalized_frame = cv2.cvtColor(color_corrected_frame, cv2.COLOR_BGR2GRAY)
    
    #5. Contrast Enhancement using CLAHE
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    #enhanced_frame = clahe.apply(equalized_frame)

    #6. Sharpening the frame after CLAHE
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_frame = cv2.filter2D(equalized_frame, -1, sharpen_kernel)

    # Edge Dectection and Segmentation Preprocessing
    edges = cv2.Canny(sharpened_frame, 100, 200)

    return sharpened_frame, edges