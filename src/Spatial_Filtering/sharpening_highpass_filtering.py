import numpy as np
import cv2

def highpass_filter(img: np.ndarray, filter: int) -> np.ndarray:
    '''
    img: input image of the form np.ndarray
    filter: center value of the filter (-8/-4/4/+8)

    Uses second derivative to find change in intensity much better than 
    first derivative.
    '''

    # abs(filter) == 8 indicates diagonal edges as well
    if filter not in [-8, -4, 4, 8]: 
        print("Enter correct filter value, correct filter values: [-8, -4, 4, 8]")
        return np.ndarray([0])

    if filter == -8:
        kernel = np.array([
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1]
        ], dtype=np.float32)

    if filter == 8:
        kernel = np.array([
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

    if filter == 4:
        kernel = np.array([
            [0, -1, 0],
            [-1, +4, -1],
            [0, -1, 0]
        ], dtype=np.float32)

    if filter == -4:
        kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
    
    highpass_img = cv2.filter2D(img, -1, kernel)
    highpass_img = highpass_img.astype(np.uint8)
    return highpass_img

if __name__ == "__main__":
    file_path = 'src\\test_images\\test1.jpg'

    img = cv2.imread(file_path)
    if img is None:
        print("File Not Found")
        
    cv2.imshow("Original", img)

    edges = highpass_filter(img, -8)
    cv2.imshow("Highpass", edges)
    cv2.waitKey(0)