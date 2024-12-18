import cv2
import numpy as np
import torch

def prepare_image(image):
    # Resize image to 640x640 and normalize
    resized = cv2.resize(image, (640, 640))  # Resize to the model's input size
    img = resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB and HWC to CHW
    img = np.ascontiguousarray(img)  # Ensure contiguous memory layout
    img_tensor = torch.from_numpy(img).float()  # Convert to PyTorch tensor
    img_tensor /= 255.0  # Normalize to [0, 1]
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor
