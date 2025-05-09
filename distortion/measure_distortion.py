import sys
sys.path.append("..")

from find_stars import load_image

import json

import numpy as np
import cv2 as cv
from glob import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

fnames = glob('./data/orig/*FIT')


fname = fnames[1]


nx = 12
ny = 7

objp = np.zeros((nx*ny,3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2)


objpoints = []
imgpoints = []


for fname in fnames:
    img = load_image(fname, preprocess=False, border_percent=0)
    img = img / 256.0
    img = img.astype(np.uint8)

    ret, corners = cv.findChessboardCorners(img, (nx, ny), None)
    print(ret)
    print(corners)

    if ret:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(img, corners, (11, 11), (-1,-1), criteria) 
        imgpoints.append(corners2)


#    cv.drawChessboardCorners(img, (nx, ny), corners2, ret)
#    cv.imshow("Checkerboard", img)
#    cv.waitKey(0)

h, w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
#    print(ret, mtx, dist, rvecs, tvecs)
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))




print(mtx)
print(dist)
print(newcameramtx)
print(roi)

dst = cv.undistort(img, mtx, dist, None, newcameramtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

params = {"mtx":mtx.tolist(), "dist":dist.tolist(), "newcameramtx":newcameramtx.tolist(), "roi":roi}


with open("distortion_params.json", "w") as f:
    json.dump(params, f)


#cv.imshow("Checkerboard ", np.concatenate((img, dst), axis=1))
#cv.imshow("Checkerboard ", dst)
#cv.waitKey(0)



