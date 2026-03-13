import numpy as np


def get_dominant_color_info(img, t=5):
    if t < 1:
        raise ValueError("threshold must be positive")
    p = img.reshape(-1)
    c = [np.sum(p == i) for i in range(256)]
    m, b = 0, 0
    for i in range(256):
        if c[i]:
            s = sum(c[max(0, i - t + 1) : min(256, i + t)])
            if s > m:
                m, b = s, i
    return np.uint8(b), float(m / len(p) * 100)
