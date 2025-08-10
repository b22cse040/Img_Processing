import numpy as np
import cv2

def geometric_transform_filter(theta: float = 0.0, cx: float = 1.0, cy: float = 1.0, tx: float = 0.0, ty: float = 0.0, sv: float = 0.0, sh: float = 0.0):

    theta = np.deg2rad(theta)

    # Scale 
    Scale_filter = np.array([
        [cx, 0, 0],
        [0, cy, 0],
        [0, 0, 1],
    ])

    Rot_filter = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    Translation_filter = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

    Shear_filter = np.array([
        [1, sv, 0],
        [sh, 1, 0],
        [0, 0, 1]
    ])

    M = Translation_filter @ Rot_filter @ Shear_filter @ Scale_filter
    return M

def geometric_transform(
        img: np.ndarray, theta: float =0.0,
        cx = 1.0, cy = 1.0, tx = 0.0, ty = 0.0, sv = 0.0, sh = 0.0
) -> np.ndarray:
    
    M = geometric_transform_filter(theta, cx, cy, tx, ty, sv, sh)
    M_inv = np.linalg.inv(M)

    h, w = img.shape[:2]
    output = np.zeros_like(img)

    for y in range(h):
        for x in range(w):
            src_x, src_y, _ = M_inv @ np.array([x, y, 1])
            if 0 <= src_x < w and 0 <= src_y < h:
                output[y, x] = img[int(src_y), int(src_x)]

    return output

if __name__ == "__main__":
    file_path = 'src\\test_images\\test2.jpg'
    img = cv2.imread(file_path)

    cv2.imshow("Original", img)

    affine_transformed_img = geometric_transform(img, 30, 1.05, 1.05, 10, 10)
    cv2.imshow("Transformed", affine_transformed_img)
    cv2.waitKey(0)