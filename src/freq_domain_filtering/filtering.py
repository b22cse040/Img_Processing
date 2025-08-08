import numpy as np
import cv2

def filtering_freq(f_x_y: np.ndarray, h_x_y: np.array) -> np.ndarray:
    '''
    f_x_y: input image f(x, y) in the spatial domain
    h_x_y: filter h(x, y) in the spatial domain

    Returns: Output image g(x, y) in the spatial domain in the form of np.ndarray
    g(x, y) output may be high-pass or low-pass depending on the filter.
    '''
    M, N = f_x_y.shape
    P, Q = 2*M, 2*N

    m, n = h_x_y.shape

    ## Padding both image and kernel to size PxQ 
    ## (Padding only bottom and right potrions)
    fp_x_y = cv2.copyMakeBorder(
        f_x_y,
        top=0, bottom=M,
        left=0, right=N,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    hp_x_y = cv2.copyMakeBorder(
        h_x_y,
        top=0, bottom=P-m,
        left=0, right=Q-n,
        borderType=cv2.BORDER_CONSTANT,
        value=0
    )

    ## Centering image and filter for DFT->
    fp_x_y = fp_x_y.astype(np.float32)
    hp_x_y = hp_x_y.astype(np.float32)

    for x in range(P):
        for y in range(Q):
            fp_x_y[x, y] *= (-1) ** (x + y)
            hp_x_y[x, y] *= (-1) ** (x + y)

    ## Converting into frequency domain
    F_u_v = np.fft.fft2(fp_x_y)
    H_u_v = np.fft.fft2(hp_x_y)

    ## Finding the output in freq domain
    # Since, convolution in spatial Domain = multiplication in freq. domain ->
    G_u_v = F_u_v * H_u_v

    ## Converting G_u_v to spatial domain by only taking the real value
    gp_x_y = np.fft.ifft2(G_u_v)

    for x in range(P):
        for y in range(Q):
            gp_x_y[x, y] *= (-1) ** (x + y)

    ## Cropping the output to get the final output
    g_x_y = gp_x_y[:M, :N]

    g_x_y = np.abs(g_x_y)
    g_x_y = g_x_y / np.max(g_x_y) * 255
    g_x_y = g_x_y.astype(np.uint8)

    return g_x_y

    
if __name__ == "__main__":
    file_path = 'src\\test_images\\test2.jpg'
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # high_pass_filter = np.array([
    #     [0, -1, 0],
    #     [-1, 4, -1],
    #     [0, -1, 0]
    # ], dtype=np.float32) 

    low_pass_filter = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ], dtype=np.float32)

    edges = filtering_freq(img, low_pass_filter)
    cv2.imshow("Original", img)
    cv2.imshow("Final", edges)
    cv2.waitKey(0)