import numpy as np
from skimage.morphology import skeletonize
from tqdm import tqdm
from io import BytesIO
from datetime import datetime
import psycopg2
from glob import glob
import os
import requests
import matplotlib.patches as patches
import cv2 as cv
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

from skyfield.api import load, wgs84
from skyfield.iokit import parse_tle_file


import torch

import sys
sys.path.append("../utils/")
from utils import plot_one
from find_stars import load_image, create_solved_image
from read_image import read_solved_image

sys.path.append("../orbits/")
from satellites import get_nearby_satellites

sys.path.append("./training/")
from model import get_device, load_model


def parse_expstart(timestr):
    date = timestr.split('T')[0].split('-')
    clock = timestr.split('T')[1].split(':')

    year = int(date[0])
    month= int(date[1])
    day  = int(date[2])

    hour= int(clock[0])
    min= int(clock[1])
    sec= float(clock[2])

    return year, month, day, hour, min, sec

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
    skeleton = np.zeros_like(image)

    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    ii = 0

    while not is_done:
        #print("Starting iteration ", ii)
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
        
    return skeleton


def draw_aabb(line, ax, edgecolor='white', text=None):

    padding = 20
    rect = patches.Rectangle((min(line[0], line[2])-padding, 
                              min(line[1], line[3])-padding), 
                             np.abs(line[2]-line[0])+2*padding,
                             np.abs(line[3]-line[1])+2*padding, linewidth=2, facecolor='none', edgecolor=edgecolor)

        #for x1,y1,x2,y2 in line:
    ax.add_patch(rect)
    if text is not None:
        ax.text(min(line[0], line[2])-padding, min(line[1], line[3])-padding+ np.abs(line[3]-line[1])+2*padding, 
        text, color='black', fontsize=12)



def normalize_image(image):
    image -= np.min(image)
    image = 255*image / np.max(image)
    return image


def find_closest(ra, dec, ra2, dec2, satellite_coords):

    best_satellite = None
    min_dist = np.inf

    for satellite in satellite_coords:
        coord = satellite_coords[satellite]

        dra = (ra-coord[0])*np.cos(0.5*(dec+dec2)*3.14/180)
        ddec = (dec-coord[1])
        dra2 = (ra2-coord[2])*np.cos(0.5*(dec+dec2)*3.14/180)
        ddec2 = (dec2-coord[3])
        total_offset = np.sqrt(dra**2 + ddec**2) + np.sqrt(dra2**2 + ddec2**2)

        dra = (ra-coord[2])*np.cos(0.5*(dec+dec2)*3.14/180)
        ddec = (dec-coord[3])
        dra2 = (ra2-coord[0])*np.cos(0.5*(dec+dec2)*3.14/180)
        ddec2 = (dec2-coord[1])
        total_offset_rev = np.sqrt(dra**2 + ddec**2) + np.sqrt(dra2**2 + ddec2**2)

        print(total_offset, total_offset_rev)
        min_offset = min(total_offset, total_offset_rev)
        if min_offset < min_dist:
            min_dist = min_offset
            best_satellite = satellite
    return min_dist, best_satellite

    
def find_lines(image, wcs, header, sats, satellite_coords, start=0, model=None, device=None, plotting=False, startimg=None):

    image = image.astype(np.float32)
    if startimg is not None:
        diffimg = image-startimg
    else:
        diffimg = image

    exptime = header["EXPTIME"]
    expstart = header["DATE-OBS"]

#    image -= np.min(image)

#    image = 255*image / np.max(image)
    # ensure img is in uint8
    image = np.nan_to_num(image, posinf=0, neginf=0)
    diffimg = np.nan_to_num(diffimg, posinf=0, neginf=0)

#    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
#    axs[0].imshow(image)



    image16 = image.copy()
#    image = cv.GaussianBlur(image, (3, 3), 0)
#    image16 = cv.GaussianBlur(image16, (3, 3), 0)
    diffimg = cv.GaussianBlur(diffimg, (3, 3), 0)

#    axs[1].imshow(image)
#    plt.show()


#    image = normalize_image(image)
    diffimg = normalize_image(diffimg)
    med = np.median(diffimg)
    diffimg_abs = diffimg.copy() - med
    diffimg_abs = np.abs(diffimg_abs)
    diffimg_abs += med

    image16 = normalize_image(image16)

    diffimg = diffimg.astype(np.uint8)
    diffimg_abs = diffimg_abs.astype(np.uint8)


#    print(np.median(image), np.std(image))
    med, std = iterative_stats(diffimg_abs)
    (thresh, im_bw) = cv.threshold(diffimg_abs, med+2.0*std, 255, cv.THRESH_BINARY)
#    ax2 = plot_one(im_bw)

#    print(thresh)
#    plot_one(im_bw)
    #sk = skeleton(im_bw)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE ,(3,3))
    im_bw = cv.morphologyEx(im_bw, cv.MORPH_OPEN, kernel)

    #plot_one(sk)
    if plotting:
        #ax = plot_one(image16)
        ax = plot_one(im_bw)
        ax = plot_one(diffimg)
        #ax = plot_one(sk)

#    res = feature_based_matching(diffimg, "./sat.png")
#    print(res)


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
    rho = 1.0  # distance resolution in pixels of the Hough grid
    theta = 0.5 * np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15  # minimum number of pixels making up a line
    max_line_gap = 3  # maximum gap in pixels between connectable line segments

    lines = cv.HoughLinesP(im_bw, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)


    if lines is None:
        return start

    line_indices = deduplicate(lines)
    line_indices_u = np.unique(line_indices)
    lines = lines[line_indices_u]
    print(lines)

    if  len(lines) > 0:
        if plotting:
            nx = int(np.sqrt(len(lines)))
            ny = int(len(lines) / nx)+1
            fig2, axs2 = plt.subplots(nx, ny)
    else:
        return start

    for ii, line in enumerate(lines):
        x1,y1,x2,y2 = line[0]
#        for x1,y1,x2,y2 in line:
#            print(x1, y1, x2, y2)
#            cv.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
#            ax.plot([x1, x2], [y1, y2], color='black')
        cutout, is_satellite, prob = get_image_cutout(image16, line[0], model=model, device=device)

        subimg = get_subimg(image16, line[0], buffer=16)
        #subimg = skeletonize_cutout(subimg)
#    return leftmost, rightmost, topmost, bottommost
#        x1, x2, yfind_line_endpoints(subimg))
#        subimg = image16[y1:y2, x1:x2] 

        axs2.flatten()[ii].imshow(subimg)

        myline, temp = fit_line(subimg)
        print("MYLINE:", myline)

        #axs2.flatten()[ii].imshow(temp)
        if myline is not None:
            x1,y1,x2,y2 = myline
            axs2.flatten()[ii].plot([x1, x2], [y1, y2], color='red', linewidth=2, alpha=0.6)



        if is_satellite:
            # further processing
            box_width = 0.5 * (line[0][2] - line[0][0])
            box_height= 0.5 * (line[0][3] - line[0][1])
            ra_first, dec_first = wcs.all_pix2world(line[0][0], line[0][1], 0)
            ra_second, dec_second = wcs.all_pix2world(line[0][2], line[0][3], 0)

            min_dist, best_satellite = find_closest(ra_first, dec_first, ra_second, dec_second, satellite_coords)
            print(f"MATCHED SATELLITE {best_satellite} at distance of {min_dist}")

            if plotting:
                #draw_aabb(line[0], ax, edgecolor="limegreen", text=f"{prob:.2f}-{sat_names[best_satellite]}")
                draw_aabb(line[0], ax, edgecolor="limegreen", text=f"{sat_names[best_satellite]}")
                pass


            #print("COORDS: ", ra_first, dec_first)
            #print("COORDS: ", ra_second, dec_second)
            cx = line[0][0]+box_width
            cy = line[0][1]+box_height

            stamp = np.array(cutout.cpu()).squeeze().tolist()
            w, h = np.shape(stamp)

            stamp = rescale(stamp)
            stamp = stamp.tolist()
            # rescale the stamp here
            #plt.figure()
            #plt.imshow(stamp)
            #plt.show()

            ra_center, dec_center = wcs.all_pix2world(cx, cy, 0)
            data = {
                    'RA1':float(ra_first), 
                    'DEC1':float(dec_first), 
                    'RA2':float(ra_second), 
                    'DEC2':float(dec_second), 
                    'EXPTIME':float(exptime),
                    'EXPSTART':expstart,
                    'x_pix':cx, 'y_pix':cy,
                    'image':stamp,
                    'width':w, 'height':h}

            if best_satellite is not None:
                data['satnum'] = best_satellite

            headers = {"Content-Type": "application/json"}
            try:
                r = requests.post("http://localhost:8080/api/submit/",  json=data, headers=headers)

                print(f"Status: {r.status_code}")
                print(f"Response: {r.text}")
            except Exception as e:
                print("UNABLE TO SEND INFO")
                print(e)



        else:
            if plotting:
                if prob > 1e-2:
                    draw_aabb(line[0], ax, edgecolor="red", text=f"{prob:.2f}")
                else:
                    draw_aabb(line[0], ax, edgecolor="cyan", text=f"{prob:.2f}")
            pass
#        axs2[ii].imshow(cutout, origin='lower')
#        if cutout is not None:
#            cv.imwrite(f"./training/raw/{start}.png", 256*cutout)
#            start += 1


#    return edges
    return start

def deduplicate(lines, buffer=8):
    parents = []

    for ii, this_line in enumerate(lines):
        best_index = ii
        best_area = np.abs((this_line[0][0] - this_line[0][2]+2*buffer) * (this_line[0][1] - this_line[0][3]+2*buffer))
        print(this_line[0])
        print(f"STARTING RECT {ii} with area {best_area}")
        for jj, that_line in enumerate(lines):
            that_area = overlaps(this_line[0], that_line[0], buffer=buffer)
            if that_area > best_area:
                best_area = that_area
                best_index = jj
                print("FOUND OVERLAP OF BIGGER RECTANGLE", jj)

        parents.append(best_index)
    return parents

def overlaps(line1, line2, threshold = 0.5, buffer=8):
    ax1,ay1,bx1,by1 = line1
    ax1, bx1 = min(ax1, bx1)-buffer, max(ax1, bx1)+buffer
    ay1, by1 = min(ay1, by1)-buffer, max(ay1, by1)+buffer
    area1 = (bx1-ax1) * (by1-ay1)
    ax2,ay2,bx2,by2 = line2
    ax2, bx2 = min(ax2, bx2)-buffer, max(ax2, bx2)+buffer
    ay2, by2 = min(ay2, by2)-buffer, max(ay2, by2)+buffer
    area2 = (bx2-ax2) * (by2-ay2)


    # rect format: (x1, y1, x2, y2) where (x1,y1) is bottom-left, (x2,y2) is top-right
    x1_max = max(ax1, ax2)  # Left edge of overlap
    y1_max = max(ay1, ay2)  # Bottom edge of overlap
    x2_min = min(bx1, bx2)  # Right edge of overlap
    y2_min = min(by1, by2)  # Top edge of overlap

    # Check if rectangles actually overlap
    if x1_max < x2_min and y1_max < y2_min:
        width = x2_min - x1_max
        height = y2_min - y1_max
        if width * height / area1 > threshold:
            return area2
        print("SMALL OVERLAP")
        return 0
            
    else:
        print("NO OVERLAP")
        return 0  # No overlap

def skeletonize_cutout(img):
    img = img.copy() - np.median(img)
    img[img<0] = 0
    sigma = np.std(img)

    # Threshold to isolate black line
    _, binary = cv.threshold(img, 1.5*sigma, 255, cv.THRESH_BINARY)

    skeleton = skeletonize(binary)
    return skeleton



def fit_line(img):

    img = cv.GaussianBlur(img, (3, 3), 0)
    img = img.copy() - np.median(img)
    img[img<0] = 0
    img[img > 5*np.std(img)] = 5*np.std(img)
    sigma = np.std(img)
    #med, sigma = iterative_stats(img.copy(), n=10)

    found_line = False
   
    for ns in np.linspace(3, 13, 4)[::-1]:
        _, binary = cv.threshold(img, ns*sigma, 255, cv.THRESH_BINARY)
        sk = skeletonize(binary).astype(np.uint8)
        lines = cv.HoughLinesP(sk, 0.7, np.pi/360, threshold=7, 
                        minLineLength=5, maxLineGap=1)

        if lines is not None and len(lines) > 0:
            print(len(lines), lines, "Found line at sigma:", ns)

            return get_longest_line(lines[0]), binary
    return None, binary 

def get_longest_line(lines):
    longest_idx = 0
    longest_length = 0
    for ii, l in enumerate(lines):
        x1, y1, x2, y2 = l
        length = (x1-x2)**2 + (y1-y2)**2
        if length > longest_length:
            longest_length = length
            longest_idx = ii
    return lines[longest_idx]

def get_subimg(image, line, buffer=8):
    x1, y1, x2, y2 = line
    width = x2-x1
    height = y2-y1
    cx = int(round(0.5*(x1+x2)))
    cy = int(round(0.5*(y1+y2)))

    largest_size = max((width, height))
    half_size = int(round(0.5*largest_size))

    cutout = image[cy-half_size-buffer:cy+half_size+buffer, cx-half_size-buffer:cx+half_size+buffer]

    return cutout



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

    #print(np.shape(cutout))
    if 0 in list(np.shape(cutout)):
        return None, False, 0.0

    cutout = cv.resize(cutout, size)

    # scale the values to be from 0-1
    cutout = cutout - np.nanmin(cutout)
    cutout = cutout / np.nanmax(cutout)
    cutout = rescale(cutout)

    is_satellite = False

    if model:
        #cutout = cutout * 256.0
        #fig, ax = plt.subplots()
        #ax.imshow(cutout)

        cutout = torch.tensor(cutout, dtype=torch.float32)[None,None,:,:]
        cutout = cutout.to(device)

        result = model(cutout)
        #print(result)
        prob = float(result.cpu())
        result = float(result.cpu())
        result = int(round(result))
        #plt.show()
        if result == 1:
            is_satellite = True

    if cutout is not None:
        start = 0
        existing_files = sorted(glob("./training/raw/*png"), key=os.path.getmtime)
#        searchedfiles = sorted(glob.glob("*cycle*.log"), key=os.path.getmtime)
        if len(existing_files) > 0:
            print(existing_files[-1])
            last = int(existing_files[-1].split('/')[-1].replace('.png', ''))
            start = last + 1
        img_to_save = np.array(cutout.cpu()).squeeze()
#        img_to_save = rescale(img_to_save)
        print(img_to_save)
#        if prob > 1e-2:
#        cv.imwrite(f"./training/raw/{start}.png", 256*img_to_save)

    return cutout, is_satellite, prob


def rescale(img):
    img = np.array(img)
    scaler = ZScaleInterval()
    limits = scaler.get_limits(img)

    print(img<limits[1])
    img[img>limits[1]] = limits[1]
    img[img<limits[0]] = limits[0]

    img = (img - limits[0]) / (limits[1] - limits[0])
    return img



def get_updated_satellites(timestamp):

    sats = []
    ts = load.timescale()

    conn = psycopg2.connect(
        host="localhost",
        database="skyscan",
        user="michael",
        password=""
    )

    cursor = conn.cursor()

#    getAll_query = """
#    SELECT * FROM ELEMENTS;
#    """
#
#    cursor.execute(getAll_query)
#    rows = cursor.fetchall()
#    t = []
#    for row in rows:
#        id,satnum,name,created_at,updated_at,epoch,line1,line2 = row
#        t.append(epoch)
    
    getRecent_query = """
    SELECT DISTINCT ON (satnum) * from elements
    ORDER BY satnum, ABS(EXTRACT(EPOCH from epoch) - EXTRACT(EPOCH from %s))
            """

#    import matplotlib.dates as mdates
#    fig, ax = plt.subplots()

    times = []
    cursor.execute(getRecent_query, (timestamp,))
    rows = cursor.fetchall()
    for row in tqdm(rows):
        id,satnum,name,created_at,updated_at,epoch,line1,line2 = row
        time_diff = epoch - timestamp
        print(timestamp, epoch, time_diff, time_diff.total_seconds())
        if time_diff.total_seconds() > 2*24*60*60:
            continue
        f = BytesIO(str.encode(f"{name}\n{line1}\n{line2}"))
        sat = list(parse_tle_file(f, ts))[0]
        sats.append(sat)
#        times.append(epoch)
#
#    dates_numeric = mdates.date2num(times)
#    t_numeric = mdates.date2num(t)
#    ax.hist(t_numeric, bins=100)
#    ax.hist(dates_numeric, bins=100)
#    plt.show()

    sat_names = {}
    for sat in sats:
        sat_names[sat.model.satnum] = sat.name

    return sats, sat_names


if __name__ == "__main__":
    DATA_DIR = "/Users/michael/ASICAP/CapObj/2025-04-17_03_46_06Z"
    DATA_DIR = "/Users/michael/ASICAP/CapObj/2025-06-09_04_55_49Z"
    DATA_DIR = "../data/2025-06-17_05_38_56Z"


    fnames = sorted(glob(f"{DATA_DIR}/*FIT"))

#    fnames = []
#    for ii in range(10000)[3528:4371]:
#        #fnames.append(f"{DATA_DIR}/2025-04-17-0346_1-CapObj_{ii:04d}.FIT")
#        fnames.append(f"{DATA_DIR}/2025-06-09-0455_8-CapObj_{ii:04d}.FIT")

    # load the model here
    device = get_device()
    model = load_model("./models/model.pth", device)
    #print(model)

    startfname = fnames[0]
    solved_fname = create_solved_image(startfname, iterations=2)
    startimg, wcs, header = read_solved_image(solved_fname)



    ts = load.timescale()

    timestamp_value = parse_expstart
    y, m, d, hr, minutes, sec = parse_expstart(header["DATE-OBS"])
    timestamp_value = datetime(y, m, d, hr, minutes, int(sec))  # Year, month, day, hour, minute, second
    sats, sat_names = get_updated_satellites(timestamp_value)
    print("loaded satellite models")

#    with load.open('../orbits/data/utc2025apr17_u.dat') as f:
#        sats = list(parse_tle_file(f, ts))



#    sat_names = {}
#    for sat in sats:
#        sat_names[sat.model.satnum] = sat.name


    plotting=True
    imageno=0
    for fname in fnames[1:]:
#        img = load_image(fname, preprocess_image=True, border_percent=0.15)
        solved_fname = create_solved_image(fname, iterations=2)
#        plot_one(img)
        #print("Starting from ", imageno)
        img, wcs, header = read_solved_image(solved_fname)

        coords, pixels = get_nearby_satellites(img, wcs, header, sats, ts, plotting=plotting)
        print(coords)
 
        imageno = find_lines(img, wcs, header, sat_names, coords, start=imageno, model=model, device=device, plotting=plotting, startimg=startimg)
        if startimg is not None:
            startimg = img

       
#        plot_one(edges)
        if plotting:
            plt.show()
