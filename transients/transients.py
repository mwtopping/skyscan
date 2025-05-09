import numpy as np
import requests
import matplotlib.patches as patches
import cv2 as cv
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

import torch

import sys
sys.path.append("../utils/")
from utils import plot_one
from find_stars import load_image

sys.path.append("./training/")
from model import get_device, load_model

def callback(x):
    print(x)

def iterative_stats(image, n=3):
    std = np.max(image)
    med = np.median(image)
    for ii in range(10):
        med = np.median(image[image<(med+n*std)])
        std = np.std(image[image<(med+n*std)])
    
    return med, std



def skeleton(image):
    is_done = False
#cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    skeleton = np.zeros_like(image)

    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    ii = 0

    while not is_done:
        print("Starting iteration ", ii)
        temp = cv.morphologyEx(image, cv.MORPH_OPEN, element)
        temp = cv.bitwise_not(temp)
        temp = cv.bitwise_and(image, temp, temp)
        skeleton = cv.bitwise_or(temp, skeleton)
        image = cv.erode(image, element)


#        max = cv.minMaxLoc(image, 0) 
        max = np.max(image)
        if max == 0:
            is_done = True
        ii += 1
        

    #bool done;
    #do
    #{
    #  cv::morphologyEx(img, temp, cv::MORPH_OPEN, element);
    #  cv::bitwise_not(temp, temp);
    #  cv::bitwise_and(img, temp, temp);
    #  cv::bitwise_or(skel, temp, skel);
    #  cv::erode(img, img, element);
    # 
    #  double max;
    #  cv::minMaxLoc(img, 0, &max);
    #  done = (max == 0);
    #} while (!done);
    return skeleton


def draw_aabb(line, ax, **kwargs):

    padding = 20
    rect = patches.Rectangle((min(line[0], line[2])-padding, 
                              min(line[1], line[3])-padding), 
                             np.abs(line[2]-line[0])+2*padding,
                             np.abs(line[3]-line[1])+2*padding, linewidth=2, facecolor='none', **kwargs)

        #for x1,y1,x2,y2 in line:
    ax.add_patch(rect)


    
def find_lines(image, start=0, model=None, device=None):


    # ensure img is in uint8
    image = np.nan_to_num(image, posinf=0, neginf=0)

    image = cv.GaussianBlur(img, (5, 5), 0)


    image -= np.min(image)


    image = 255*image / np.max(image)
    image = image.astype(np.uint8)


    ax = plot_one(image)

#    print(np.median(image), np.std(image))
    med, std = iterative_stats(image)
    (thresh, im_bw) = cv.threshold(image, med+3*std, 255, cv.THRESH_BINARY)
#    plot_one(thresh)
#    print(thresh)
#    plot_one(im_bw)
    sk = skeleton(im_bw)
    #plot_one(sk)




#    canny = cv.Canny(sk, 85, 255) 
#
#    cv.namedWindow('image') # make a window with name 'image'
#    cv.createTrackbar('L', 'image', 0, 255, callback) #lower threshold trackbar for window 'image
#    cv.createTrackbar('U', 'image', 0, 255, callback) #upper threshold trackbar for window 'image
#
#    while(1):
#        numpy_horizontal_concat = np.concatenate((img, canny), axis=1) # to display image side by side
#        cv.imshow('image', numpy_horizontal_concat)
#        k = cv.waitKey(1) & 0xFF
#        if k == 27: #escape key
#            break
#        l = cv.getTrackbarPos('L', 'image')
#        u = cv.getTrackbarPos('U', 'image')
#
#        canny = cv.Canny(sk, l, u)
#
#    cv.destroyAllWindows()
#    plot_one(edges)
#
#
    rho = 1.5  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 20  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(sk, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)


    if lines is None:
        return start


    if  len(lines) > 0:
        #fig2, axs2 = plt.subplots(len(lines))
        pass
    else:
        return start


    for ii, line in enumerate(lines):
#        for x1,y1,x2,y2 in line:
#            print(x1, y1, x2, y2)
#            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
#            ax.plot([x1, x2], [y1, y2], color='black')
        cutout, is_satellite = get_image_cutout(image, line[0], model=model, device=device)
        if is_satellite:
            draw_aabb(line[0], ax, edgecolor="limegreen")
            # further processing
            box_width = 0.5 * (line[0][2] - line[0][0])
            box_height= 0.5 * (line[0][3] - line[0][1])
            cx = line[0][0]+box_width
            cy = line[0][1]+box_height
            data = {'RA':0.0, 'DEC':0.0, 'x_pix':cx, 'y_pix':cy}
            headers = {"Content-Type": "application/json"}

            r = requests.post("http://localhost:8080",  json=data, headers=headers)

            print(f"Status: {r.status_code}")
            print(f"Response: {r.text}")



        else:
            draw_aabb(line[0], ax, edgecolor="red")
#        axs2[ii].imshow(cutout, origin='lower')
#        if cutout is not None:
#            cv.imwrite(f"./training/raw/{start}.png", 256*cutout)
#            start += 1


#    return edges
    return start



def get_image_cutout(img, aabb, size=(32, 32), model=None, device=None):

    buffer = 8

    x1, y1, x2, y2 = aabb
    width = x2-x1
    height = y2-y1
    cx = int(round(0.5*(x1+x2)))
    cy = int(round(0.5*(y1+y2)))

    largest_size = max((width, height))
    half_size = int(round(0.5*largest_size))

    cutout = img[cy-half_size-buffer:cy+half_size+buffer, cx-half_size-buffer:cx+half_size+buffer]

    print(np.shape(cutout))
    if 0 in list(np.shape(cutout)):
        return None, False

    cutout = cv.resize(cutout, size)

    # scale the values to be from 0-1
    cutout = cutout - np.nanmin(cutout)
    cutout = cutout / np.nanmax(cutout)

    is_satellite = False

    if model:
        #cutout = cutout * 256.0
        #fig, ax = plt.subplots()
        #ax.imshow(cutout)

        cutout = torch.tensor(cutout, dtype=torch.float32)[None,None,:,:]
        cutout = cutout.to(device)

        result = model(cutout)
        print(result)
        result = float(result.cpu())
        result = int(round(result))
        #plt.show()
        if result == 1:
            is_satellite = True


    return cutout, is_satellite




if __name__ == "__main__":
    fnames = []
    for ii in range(10000)[1000:1100]:
        fnames.append(f"/Users/michael/ASICAP/CapObj/2025-04-17_03_46_06Z/2025-04-17-0346_1-CapObj_{ii:04d}.FIT")

    # load the model here
    device = get_device()
    model = load_model("./models/model.pth", device)
    print(model)

    imageno=0
    for fname in fnames:
        img = load_image(fname, preprocess_image=True, border_percent=0.15)
#        plot_one(img)
        print("Starting from ", imageno)
        imageno = find_lines(img, start=imageno, model=model, device=device)
        
#        plot_one(edges)

        plt.show()

