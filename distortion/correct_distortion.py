import sys
sys.path.append("..")


import json

import numpy as np
import cv2 as cv


def load_distortion_params(fname):
    with open(fname, "r") as f:
        data = f.read()
        params = json.loads(data)

    mtx = np.array(params["mtx"])
    dist = np.array(params["dist"])
    newcameramtx = np.array(params["newcameramtx"])
    roi = params["roi"]

    return mtx, dist, newcameramtx, roi


##    img = img / 256.0
##    img = img.astype(np.uint8)
#


#
def correct_distortion(img, mtx, dist, newcameramtx, roi):
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst
#
#params = {"mtx":mtx.tolist(), "dist":dist.tolist(), "newcameramtx":newcameramtx.tolist(), "roi":roi}
#
#
#with open("distortion_params.json", "w") as f:
#    json.dump(params, f)
#
#
##cv.imshow("Checkerboard ", np.concatenate((img, dst), axis=1))
#mtx, dist, newcameramtx, roi = load_distortion_params()
#img = load_image(fname, preprocess=False, border_percent=0)
#dst = correct_distortion(img, mtx, dist, newcameramtx, roi)
#
#cv.imshow("Checkerboard fixed", dst)
#cv.imshow("Checkerboard orig", img)
#cv.waitKey(0)
#
#
#
