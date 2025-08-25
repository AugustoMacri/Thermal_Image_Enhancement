import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.exposure import match_histograms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# =====================
# Configuração inicial
# =====================
image_paths = [
    'dataset/garden/00000027.png',
    'dataset/quadrocopter2/00000165.png',
    'dataset/street/00000017.png'
]
images = [cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in image_paths]

def display_images(images, titles, cols=3, cmap='gray', figsize=(15, 8)):
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=figsize)
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap=cmap)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# =====================
# Transformações globais
# =====================
def log_transform(image):
    c = 255 / np.log(1 + np.max(image))
    return (c * np.log(1 + image)).astype(np.uint8)

def histogram_equalization(image):
    return cv2.equalizeHist(image)

# Aplicações globais
log_images = [log_transform(img) for img in images]
equalized_images = [histogram_equalization(img) for img in images]
log_equalized_images = [histogram_equalization(img) for img in log_images]

# Histogram Matching (usando a 1ª imagem como referência)
reference = images[0]
matched_images = [match_histograms(img, reference, channel_axis=None).astype(np.uint8) 
                  for img in log_equalized_images]

# =====================
# Processamento em blocos
# =====================
def block_process(image, block_size=32, func=histogram_equalization):
    h, w = image.shape
    out = np.zeros_like(image)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            i_end = min(i + block_size, h)
            j_end = min(j + block_size, w)
            block = image[i:i_end, j:j_end]
            if block.size > 1:
                out[i:i_end, j:j_end] = func(block)
            else:
                out[i:i_end, j:j_end] = block
    return out

block_eq = [block_process(img, block_size=32, func=histogram_equalization) for img in images]
block_log = [block_process(img, block_size=32, func=log_transform) for img in images]

# =====================
# CLAHE (Adaptive)
# =====================
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_images = [clahe.apply(img) for img in images]

# =====================
# Métricas (PSNR / SSIM)
# =====================
for name, original, processed in zip(
    ["Garden", "Quadrocopter", "Street"], 
    images, 
    log_equalized_images
):
    psnr = peak_signal_noise_ratio(original, processed)
    ssim = structural_similarity(original, processed)
    print(f"{name} -> PSNR: {psnr:.2f}, SSIM: {ssim:.3f}")

# =====================
# Exibição dos resultados
# =====================
titles = [
    "Original", 
    "Log", 
    "Equalized", 
    "Log + Equalized", 
    "Histogram Matching",
    "Block Equalization", 
    "Block Log", 
    "CLAHE"
]

for idx, img_set in enumerate(zip(images, log_images, equalized_images, 
                                  log_equalized_images, matched_images,
                                  block_eq, block_log, clahe_images)):
    display_images(list(img_set), titles, cols=4, figsize=(16, 12))
