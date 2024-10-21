import torch
import cv2
import numpy as np
from torchvision import models, transforms

#Load the pre-trained DeepLabV3 model
model = models.segmentation.deeplabv3_resnet101(pretrained=True)
model.eval() #Set the model to evaluation mode

# Define image preprocessing transformations
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((512,512)), #Resize frame to the input size required by DeepLabV3
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def segment_frame_deeplabv3(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Preprocess the frame
    input_tensor = preprocess(rgb_frame)
    input_batch = input_tensor.unsqueeze(0) 

    #Run the model the input frame
    with torch.no_grad():
        output = model(input_batch)['out'][0] # Output is a tensor of shape [C, H, W]

    # Get the segmentation mask by taking the argmax of the output
    output_predictions = output.argmax(0).byte().cpu().numpy()
    
    # Apply thresholding to the segmentation mask
    binary_mask = threshold_segmentation_mask(output_predictions, swimmer_class=15, swimmer_value=128)

    #Resize the mask to match the original frame's dimensions
    original_height, original_width = frame.shape[:2]

    #Print dimensions for debugging
    #print(f"Original Frame Size: {original_width}x{original_height}")
    #print(f"Mask Size Before Resizing: {mask.shape}")

    #Resize the mask to match the original frame's dimensions
    resized_mask = cv2.resize(binary_mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)

    return resized_mask

def threshold_segmentation_mask(mask, swimmer_class=15, swimmer_value=128):
    #Create a binary mask, set swimmer pixels to 'swimmer_value' and everything else to 0 (black)
    binary_mask = np.zeros_like(mask, dtype=np.uint8)
    binary_mask[mask == swimmer_class] = swimmer_value
    return binary_mask

def post_process_mask(mask):
    kernel = np.ones((5,5), np.uint8) # Adjust the kernel size as needed

    #Remove small noise by performing an 'opening' (erosion followed by dilation)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Fill small holes by performing a 'closing' (dilation followed by erosion)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask