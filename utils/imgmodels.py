import numpy as np
from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
import cv2 as cv
import time
from skimage.draw import line_aa


def rescale(img):
    img = np.array(img)
    scaler = ZScaleInterval()
    limits = scaler.get_limits(img)

    #print(img<limits[1])
    img[img>limits[1]] = limits[1]
    img[img<limits[0]] = limits[0]

    img = (img - limits[0]) / (limits[1] - limits[0])
    return img

def draw_line_cv(width, height, cx, cy, length, angle, value):
    img = np.zeros((width, height))
    p1 = np.array([cx, cy]) + np.array([length/2*np.sin(angle), length/2*np.cos(angle)])
    p2 = np.array([cx, cy]) - np.array([length/2*np.sin(angle), length/2*np.cos(angle)])
    res = cv.line(img, p1.astype(int), p2.astype(int), color=value*250, thickness=2)
    result = cv.GaussianBlur(res, (3,3), 0, borderType=cv.BORDER_REPLICATE)/250.
    return result

def draw_line_sk(width, height, cx, cy, length, angle, value):
    img = np.zeros((width, height))
    p1 = np.array([cx, cy]) + np.array([length/2*np.sin(angle), length/2*np.cos(angle)])
    p2 = np.array([cx, cy]) - np.array([length/2*np.sin(angle), length/2*np.cos(angle)])
    rr, cc, val = line_aa(*p1, *p2)
    #result = cv.GaussianBlur(res, (3,3), 0, borderType=cv.BORDER_REPLICATE)/250.
    img[rr,cc] = val
    return img



def draw_line_np(image, value=1):
    x0 = 1
    y0=1
    x1=30
    y1=20
    # Calculate the number of points needed
    num_points = max(abs(x1 - x0), abs(y1 - y0)) + 1

    # Generate coordinates along the line
    x_coords = np.linspace(x0, x1, num_points, dtype=int)
    y_coords = np.linspace(y0, y1, num_points, dtype=int)
    # Set the line pixels
    image[x_coords, y_coords] = value

    return image

def distance_based_line_endpoints(cx, cy, length, angle):
    x0, y0 = np.array([cx, cy]) + np.array([length/2.*np.sin(angle), length/2.*np.cos(angle)])
    x1, y1 = np.array([cx, cy]) - np.array([length/2.*np.sin(angle), length/2.*np.cos(angle)])
    return [x0, y0, x1, y1]


def distance_based_line(width, height, cx, cy, length, angle, value):
    linewidth=1.
    x0, y0 = np.array([cx, cy]) + np.array([length/2.*np.sin(angle), length/2.*np.cos(angle)])
    x1, y1 = np.array([cx, cy]) - np.array([length/2.*np.sin(angle), length/2.*np.cos(angle)])


    """Create line based on distance from line equation"""
    rows, cols = np.mgrid[0:width, 0:height]

    # Line equation: ax + by + c = 0
    a = y1 - y0
    b = x0 - x1
    c = x1 * y0 - x0 * y1

    # Distance from each pixel to line
    dist = np.abs(a * cols + b * rows + c) / np.sqrt(a**2 + b**2)

    # Create line with anti-aliasing based on distance
    line_mask = np.exp(-dist**2 / (2 * (linewidth/2)**2))

    # Limit to actual line segment
    t = ((cols - x0) * (x1 - x0) + (rows - y0) * (y1 - y0)) / ((x1 - x0)**2 + (y1 - y0)**2)
    segment_mask = (t >= 0) & (t <= 1)

    result = cv.GaussianBlur(line_mask * segment_mask*value, (3,3), 0, borderType=cv.BORDER_REPLICATE)
#    result = cv.GaussianBlur(result, (11,11), 0, borderType=cv.BORDER_REPLICATE)
#    result[result>0.35] = 0.5
    return result

if __name__ == "__main__":
    size = (32, 32)
    img = np.zeros(size)
    starttime = time.time()
    res = draw_line_cv(32, 32, 15, 15, 5, 0.2, 1)
    print(time.time()-starttime)
    starttime = time.time_ns()
    print(time.time_ns()-starttime)
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    axs[0][0].imshow(img)
    axs[0][1].imshow(res)


#    image = distance_based_line((32, 32), 15, 15, 30, 30, width=1.0)
#    axs[1][0].imshow(image)

    image = distance_based_line(32, 32, 15, 15, 5, 0.2, 0)
#    image = distance_based_line((32, 32), 15, 15, 30.5, 30, width=1.0)
    axs[1][1].imshow(image)


    plt.show()
    
