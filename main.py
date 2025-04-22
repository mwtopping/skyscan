import time
from glob import glob
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from tqdm import tqdm
import itertools
import numpy as np
import cv2 as cv

from find_stars import *

from astropy.visualization import ZScaleInterval


def get_total_shift_inv(shift):

    tot_shift_h = [0, 0]
    tot_shift_v = [0, 0]
    if shift[0]>shift[1]:
        tot_shift_v = [0, shift[0]-shift[1]]
    else:
        tot_shift_v = [shift[1]-shift[0], 0]

    if shift[2]>shift[3]:
        tot_shift_h = [0, shift[2]-shift[3]]
    else:
        tot_shift_h = [shift[3]-shift[2], 0]


    return [tot_shift_v[0], tot_shift_v[1],
            tot_shift_h[0], tot_shift_h[1]]

# currently unused
def transform_points(rot, shift, points, recenter=False):

    if recenter:
        mean = np.mean(points, axis=0)
        points -= mean
    else:
        mean = 0


    ones = np.ones(np.shape(points)[0])
    points = np.column_stack((points, ones))

    rotmat = np.matrix([[np.cos(rot),-1*np.sin(rot), shift[0]],
                        [np.sin(rot),np.cos(rot), shift[1]],
                        [0,0,1]])
    outpoints = (rotmat*points.T).T[:,:2]
    if recenter:
        outpoints += mean
    return outpoints 



def get_angles(xs, ys):

    a = np.sqrt((xs[0]-xs[1])**2 + (ys[0]-ys[1])**2)
    b = np.sqrt((xs[1]-xs[2])**2 + (ys[1]-ys[2])**2)
    c = np.sqrt((xs[0]-xs[2])**2 + (ys[0]-ys[2])**2)
    
    a1 = np.arccos((a*a + c*c - b*b) / (2*a*c))#opposite point 0
    a2 = np.arccos((a*a + b*b - c*c) / (2*a*b))#opposite point 1
    a3 = np.arccos((b*b + c*c - a*a) / (2*b*c))#opposite point 2

    angles = np.array([a1, a2, a3])
    sorted_inds = np.argsort(angles)
    sorted_angles = angles[sorted_inds]
    sorted_pos = [xs[sorted_inds], ys[sorted_inds]]

    return sorted_angles, sorted_pos


def get_all_tris(ids):
    perms = list(itertools.combinations(ids, 3))
    return perms



def match_angles(angles1, angles2):
    # get first two angles of each list
    #  angles are sorted, so this is smallest 2 angles
    angles1 = np.array(angles1)[:,:2]
    angles2 = np.array(angles2)[:,:2]

    tree_angles1 = KDTree(angles1)
    tree_angles2 = KDTree(angles2)

    dists, inds = tree_angles2.query(angles1, k=1)

    return dists, inds



def get_all_angles(xs, ys, N=4):
    allperms = set()
    for ii in range(len(xs)):
        dists = (xs-xs[ii])**2 + (ys-ys[ii])**2
        closest = np.argsort(dists).tolist()[:N+1]

        perms = [tuple(sorted(p)) for p in get_all_tris(closest)]
        for p in perms:
            allperms.add(p)

    allperms = list(allperms)
    angles = []
    pos = []
    for p in allperms:
        sorted_angles, sorted_pos = get_angles(xs[list(p)], ys[list(p)])
        angles.append(sorted_angles)
        pos.append(sorted_pos)

    return angles, pos



def get_frame_transform(ref_fname, target_fname, ref_data = None, target_data = None):
    if ref_data is not None:
        xs, ys, img1 = ref_data
    else:
        xs, ys, img1 = get_star_locs(load_image(ref_fname), return_image=True)

    if target_data is not None:
        xs_shift, ys_shift, img2 = target_data
    else:
        xs_shift, ys_shift, img2 = get_star_locs(load_image(target_fname), return_image=True)


    if xs_shift is None:
        return None, None, None, None, None

#    print("COMARING IMAGE SHAPES:", np.shape(img2), np.shape(img1))
#    ref_shape = np.shape(img1)
#    tar_shape = np.shape(img2)
#    shape_diff = np.array(ref_shape)-np.array(tar_shape)
#    print(shape_diff)
#
#    img2 = cv.copyMakeBorder(img2, 0, shape_diff[0], 0, shape_diff[1],
#                             cv.BORDER_CONSTANT)
#

    starttime = time.time()
    angles, pos = get_all_angles(xs, ys)
    angles_shift, pos_shift = get_all_angles(xs_shift, ys_shift)

    dists, inds = match_angles(angles, angles_shift)
    weights = np.pow(np.clip(1-dists*10, 0, 1), 4)

    all_refs = []
    all_targs = []


    for ii, ind in enumerate(inds):
        w = weights[ii][0]
        ind = ind[0]
        p = pos[ii]
        ps = pos_shift[ind]

        ref_point  = [p[0][0] , p[1][0]]
        targ_point = [ps[0][0], ps[1][0]]

        if ref_point not in all_refs:
            all_refs.append(ref_point)
            all_targs.append(targ_point)



    # estimate transformation matrix here
    H, inpts = cv.estimateAffinePartial2D(np.array(all_targs),
                                        np.array(all_refs),
                                       ransacReprojThreshold=0.5)

    translation_xdir = np.sign(H[0][-1])
    translation_ydir = np.sign(H[1][-1])

    translation_x = int(round(np.abs(H[0][-1])))
    translation_y = int(round(np.abs(H[1][-1])))
    rot_angle = np.arcsin(H[1][0])

    # if rotation is positive
    if rot_angle > 0:
        shiftx = translation_x# + int(H[1][0] * np.shape(img2)[0])+1
    else: 
        shiftx = translation_x

    # if rotation is negative
    if rot_angle < 0 or rot_angle > np.pi/2:
        shifty = translation_y# + int(H[1][0] * np.shape(img2)[0])+1
    else:
        shifty = translation_y

    # transform the points here

    temp = img2.copy()
    
    # top bottom left right
    # if shiftx is negative, pad top

    if translation_xdir < 0:
        horizontal_shift = (shiftx, 0)
    else:
        horizontal_shift = (0, shiftx)
    if translation_ydir < 0:
        vertical_shift = (shifty, 0)
    else:
        vertical_shift = (0, shifty)


#    horizontal_shift = (shiftx, 0)
#    vertical_shift = (shifty, 0)

    total_shift = [vertical_shift[0], vertical_shift[1],
                   horizontal_shift[0], horizontal_shift[1]]

    
    img_border = cv.copyMakeBorder(np.ones_like(img2, dtype=float), *total_shift,
                             cv.BORDER_CONSTANT)

    temp = cv.copyMakeBorder(temp, *total_shift,
                             cv.BORDER_CONSTANT)
    outshape = (np.shape(temp)[1], np.shape(temp)[0])

    transformed1 = cv.warpAffine(temp, H, outshape)
    img_border= cv.warpAffine(img_border, H, outshape)
#    fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
#    ax[0].imshow(img1)
#    ax[1].imshow(transformed1)
#    ax[2].imshow(img_border)
#    plt.show()


    return transformed1, img_border, total_shift, xs_shift, ys_shift, img2


if __name__ == "__main__":
    starttime = time.time()
    fnames = ["/Users/michael/ASICAP/CapObj/2025-04-17_03_46_06Z/2025-04-17-0346_1-CapObj_2000.FIT",
              "/Users/michael/ASICAP/CapObj/2025-04-17_03_46_06Z/2025-04-17-0346_1-CapObj_2075.FIT"]
    fnames = []
    for ii in range(10000)[800:1250:15]:
        fnames.append(f"/Users/michael/ASICAP/CapObj/2025-04-17_03_46_06Z/2025-04-17-0346_1-CapObj_{ii:04d}.FIT")

    fnames = []
    for ii in range(10000)[800:900]:
        fnames.append(f"/Users/michael/ASICAP/CapObj/2025-04-17_03_46_06Z/2025-04-17-0346_1-CapObj_{ii:04d}.FIT")


#    fnames = ["/Users/michael/ASICAP/CapObj/2025-04-17_03_46_06Z/2025-04-17-0346_1-CapObj_2000.FIT",
#              "/Users/michael/ASICAP/CapObj/2025-04-17_03_46_06Z/2025-04-17-0346_1-CapObj_2100.FIT",
#              "/Users/michael/ASICAP/CapObj/2025-04-17_03_46_06Z/2025-04-17-0346_1-CapObj_2200.FIT"]




    final_images = None
    ref_fname = fnames[0]
    ref_xs, ref_ys, ref_image = get_star_locs(load_image(ref_fname), return_image=True)
    first_image = ref_image.copy()
    final_image = ref_image.copy().astype(np.float64)
    Nimages = np.ones_like(final_image)
    total_shift = np.array([0,0,0,0])
    total_shift_inv = np.array([0,0,0,0])

    shifts = []


    for target_fname in tqdm(fnames[1:]):
        transformed, img_contribution, shift, tar_xs, tar_ys, tar_image = get_frame_transform(ref_fname, target_fname,
                                          ref_data = (ref_xs, ref_ys, ref_image))
        if transformed is None:
            continue

        if np.sum(shift) > 500:
            print("TOO LARGE OF SHIFT, SKIPPING")
            continue

        if len(shifts) > 5:
            if np.sum(shift) > 5*np.std(shifts):
                print("FRAME SHIFT OUTLIER")
                continue

        shifts.append(np.sum(shift))
        print(shift)
        print(total_shift_inv)
        transformed = cv.copyMakeBorder(transformed, *total_shift_inv, cv.BORDER_CONSTANT)
        img_contribution = cv.copyMakeBorder(img_contribution, *total_shift_inv, cv.BORDER_CONSTANT)

        total_shift += np.array(shift)
        total_shift_inv = np.array([total_shift[1], total_shift[0],
                                    total_shift[3], total_shift[2]])

#        total_shift_inv = get_total_shift_inv(total_shift)

        if final_image is None:
            print("FUCK")
            final_images = transformed
            ref_image = transformed
        else:

#            fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)
#            axs[0].imshow(cv.copyMakeBorder(final_image, *shift, cv.BORDER_CONSTANT), origin='lower')
#            axs[1].imshow(transformed, origin='lower')
#            plt.show()
            final_image = cv.copyMakeBorder(final_image, *shift, cv.BORDER_CONSTANT) + transformed
            Nimages = cv.copyMakeBorder(Nimages, *shift, cv.BORDER_CONSTANT) + img_contribution

        Nimages[Nimages==0] = np.inf
        ref_xs, ref_ys, ref_image = get_star_locs(np.nan_to_num(final_image/Nimages), return_image=True)


    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)


    scaler = ZScaleInterval()
    limits = scaler.get_limits(first_image)

    axs[0].imshow(transformed, vmin=limits[0], vmax=limits[1], cmap='Greys_r', origin='lower')

#    final_image = np.nanmedian(final_images, axis=2)
#    final_image = np.nansum(final_images, axis=2)
#    final_image = final_image-np.min(final_image)
#    print(np.min(final_image), np.max(final_image))
    limits = scaler.get_limits(final_image/Nimages)
    print(limits)
    axs[1].imshow(final_image/Nimages, vmin=limits[0], vmax=limits[1], cmap='Greys_r', origin='lower')
    axs[2].imshow(Nimages, origin='lower')

    plt.show()
    
