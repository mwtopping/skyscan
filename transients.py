import numpy as np
import matplotlib.patches as patches
import cv2 as cv
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

from find_stars import load_image

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


def draw_aabb(line, ax):

    padding = 20
    rect = patches.Rectangle((min(line[0], line[2])-padding, 
                              min(line[1], line[3])-padding), 
                             np.abs(line[2]-line[0])+2*padding,
                             np.abs(line[3]-line[1])+2*padding, linewidth=2, edgecolor='limegreen', facecolor='none')

        #for x1,y1,x2,y2 in line:
    ax.add_patch(rect)


def plot_one(image):

    scaler = ZScaleInterval()

    # do some preprocessing
    image = np.nan_to_num(image, posinf=0, neginf=0)
    image -= np.min(image)

    limits = scaler.get_limits(image)

    fig, ax = plt.subplots()
    print(limits)
    ax.imshow(image, 
              aspect='auto', 
              origin='lower', 
              vmin=limits[0], vmax=limits[1]+0.1,
              cmap='Greys')

    return ax
    
def find_lines(image):

    # ensure img is in uint8
    image = np.nan_to_num(image, posinf=0, neginf=0)

    image = cv.GaussianBlur(img, (5, 5), 0)


    image -= np.min(image)


    image = 255*image / np.max(image)
    image = image.astype(np.uint8)


    ax = plot_one(image)

    print(np.median(image), np.std(image))
    med, std = iterative_stats(image)
    (thresh, im_bw) = cv.threshold(image, med+3*std, 255, cv.THRESH_BINARY)
#    plot_one(thresh)
    print(thresh)
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
    threshold = 25  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 20  # minimum number of pixels making up a line
    max_line_gap = 5  # maximum gap in pixels between connectable line segments

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv.HoughLinesP(sk, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)

    for line in lines:
        print("Line: ", line)
        for x1,y1,x2,y2 in line:
            print(x1, y1, x2, y2)
#            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
#            ax.plot([x1, x2], [y1, y2], color='black')
        print(line)
        draw_aabb(line[0], ax)

#    return edges
    return None

if __name__ == "__main__":
    fnames = []
    for ii in range(10000)[995:1120]:
        fnames.append(f"/Users/michael/ASICAP/CapObj/2025-04-17_03_46_06Z/2025-04-17-0346_1-CapObj_{ii:04d}.FIT")


    for fname in fnames:
        img = load_image(fname)
#        plot_one(img)
        edges = find_lines(img)
#        plot_one(edges)

        plt.show()

