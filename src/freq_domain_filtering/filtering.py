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
    gp_x_y = np.fft.ifft2(G_u_v).real

    for x in range(P):
        for y in range(Q):
            gp_x_y[x, y] *= (-1) ** (x + y)

    ## Cropping the output to get the final output
    g_x_y = gp_x_y[:M, :N]

    g_x_y = np.abs(g_x_y)
    g_x_y = g_x_y / np.max(g_x_y) * 255
    g_x_y = g_x_y.astype(np.uint8)

    return g_x_y

def low_pass_filter(img: np.ndarray, D0: int, filter_type: str, n: int = 1) -> np.ndarray:
    '''
    img: input image of the form np.ndarray
    D0: cutoff-frequency
    filter_type: 
        "ILPF" : ideal low-pass filter
        "GLPF" : Gaussian low-pass filter
        "BLPF" : Butterworth low-pass gilter

    n: optional param for when filter_type == "BLPF"
    '''

    if filter_type not in ["ILPF", "GLPF", "BLPF"]: 
        print("Invalid filter type: Eligible filter_type are ILPF, GLPF and BLPF")
        return np.array([0])
    
    def ideal_low_pass_filter(P: int, Q: int, D0: int) -> np.ndarray:
        ''' Generate Ideal Low-Pass Filter Mask. '''
        U, V = np.meshgrid(np.arange(Q), np.arange(P))
        D = np.sqrt((U - Q//2) ** 2 + (V - P//2) ** 2)
        H = np.zeros((P, Q), dtype=np.float32)
        H[D <= D0] = 1
        return H 
    
    def gaussian_low_pass_filter(P: int, Q: int, D0: int) -> np.ndarray:
        ''' Generate Gaussian Low-Pass Filter Mask. '''
        U, V = np.meshgrid(np.arange(Q), np.arange(P))
        D_squared = (U - Q//2) ** 2 + (V - P//2) ** 2
        H = np.exp(-D_squared / (2 * (D0 ** 2)))
        return H
    
    def butterworth_low_pass_filter(P: int, Q: int, D0: int, n: int = 1) -> np.ndarray:
        ''' Generate Butterworth Low-Pass Filter Mask. Order is n'''
        U, V = np.meshgrid(np.arange(Q), np.arange(P))
        D_squared = (U - Q//2) ** 2 + (V - P//2) ** 2
        D = np.sqrt(D_squared)
        H = 1 / (1 + (D / D0) ** (2 * n))
        return H

    
    M, N = img.shape
    P, Q = 2*M, 2*N
    fp = np.zeros((P, Q), dtype=np.float32)
    fp[:M, :N] = img

    for u in range(P):
        for v in range(Q):
            fp[u, v] *= (-1) ** (u + v)

    F = np.fft.fft2(fp)

    if filter_type == "ILPF": H = ideal_low_pass_filter(P, Q, D0)
    elif filter_type == "GLPF": H = gaussian_low_pass_filter(P, Q, D0)
    else: H = butterworth_low_pass_filter(P, Q, D0, n)

    G = F * H

    gp = np.fft.ifft2(G).real
    for x in range(P):
        for y in range(Q):
            gp[x, y] *= (-1) ** (x + y)

    g_final = gp[:M, :N]
    g_final = np.abs(g_final)
    g_final = g_final / np.max(g_final) * 255
    g_final = g_final.astype(np.uint8)
    return g_final
    

def enhance_image(img: np.ndarray, filter: np.array) -> np.ndarray:
    '''
    Note: Filter itself should be aa highpass filter to enhance.
    '''
    sharp_components = filtering_freq(img, filter)
    enhanced_img = cv2.add(img, sharp_components)
    return enhanced_img

    
if __name__ == "__main__":
    file_path = 'src\\test_images\\test2.jpg'
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

    # high_pass_filter = np.array([
    #     [0, -1, 0],
    #     [-1, 4, -1],
    #     [0, -1, 0]
    # ], dtype=np.float32) 

    # low_pass_filter = np.array([
    #     [1, 1, 1],
    #     [1, 1, 1],
    #     [1, 1, 1]
    # ], dtype=np.float32) / 9

    # edges = filtering_freq(img, low_pass_filter)
    # enhanced_image = enhance_image(img, high_pass_filter)
    print(img.shape)
    blurred_img = low_pass_filter(img, D0=150, filter_type="BLPF", n = 1)
    cv2.imshow("Original", img)
    cv2.imshow("Final", blurred_img)
    cv2.waitKey(0)