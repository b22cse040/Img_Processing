import numpy as np
import cv2

def form_negatives(file_path: str) -> np.ndarray:
    img = cv2.imread(file_path)
    if img is None:
        print("File Not found")
        return np.ndarray([0])
    
    img = img.astype(np.uint8)

    img_negative = 255 - img
    return img_negative

def log_transform(file_path: str) -> np.ndarray:
    img = cv2.imread(file_path)
    if img is None:
        print("File Not found")
        return np.ndarray([0])
    
    img = img.astype(np.float32)
    max_intensity = np.max(img)
    c = 255 / np.log2(1 + max_intensity)
    log_transformed_img = c * np.log2(1 + img)
    log_transformed_img = log_transformed_img.astype(np.uint8)
    return log_transformed_img

def power_transform(file_path: str, c: int = 255, gamma: float = 2.0) -> np.ndarray:
    '''
    file_path: path to image
    c: constant to multiply
    gamma: power of intensity
    '''
    img = cv2.imread(file_path)
    if img is None:
        print("File Not found")
        return np.ndarray([0])
    
    img = img.astype(np.float32)
    pow_img = c * np.power(img / 255, gamma)
    pow_img = pow_img.astype(np.uint8)
    return pow_img


if __name__ == "__main__":
    file_path = 'src\\test_images\\test2.jpg'
    # neg_img = form_negatives(file_path)
    # cv2.imshow("Negative Image: ",neg_img)

    # log_img = log_transform(file_path=file_path)
    # cv2.imshow("Log Image: ", log_img)
    img = cv2.imread(file_path)
    pow_img = power_transform(file_path=file_path)
    cv2.imshow("Original", img)
    cv2.imshow("Power", pow_img)
    cv2.waitKey(0)