from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_binary_image(filename,):
    im = Image.open(filename)
    cols, rows = im.size
    px = im.load()
    data = np.zeros(cols*rows)

    for c in range(cols):
        for r in range(rows):
            data[c+r*cols] = 1 if px[c,r] < 127 else -1
    return dict(data=data, cols=cols, rows=rows)

def render_image(s, rows, cols):
    bitmap = np.zeros((rows,cols))
    for c in range(cols):
        for r in range(rows):
            bitmap[r,c] = 0 if s[c+r*cols] == 1 else 255
    
    im = plt.imshow(bitmap, cmap='gray', vmin=0, vmax=255)
    plt.draw()
    plt.pause(1e-9)
    return im, bitmap

def render_pixel(im, bitmap, pixel, col, row):
    bitmap[row, col] = 0 if pixel == 1 else 255

    im.set_data(bitmap)
    plt.draw()
    plt.pause(1e-12)
    return im, bitmap 

