import time
import cv2 as cv
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import sparse

from astropy.io import fits
from astropy.visualization import ZScaleInterval


def gaussian_2d(coords, amplitude, x0, y0, sigma_x, sigma_y, theta, offset):
    """
    2D Gaussian function
    
    Parameters:
    -----------
    coords : tuple
        (x, y) coordinates where x and y are meshgrid arrays
    amplitude : float
        Height of the gaussian
    x0, y0 : float
        Center position of the gaussian
    sigma_x, sigma_y : float
        Width of the gaussian in x and y directions
    theta : float
        Rotation angle in radians
    offset : float
        Baseline offset
    
    Returns:
    --------
    z : ndarray
        2D Gaussian evaluated at x, y points
    """
    x, y = coords
    
    # Rotation
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    
    # Gaussian function
    z = offset + amplitude * np.exp(
        -(a*((x-x0)**2) + 2*b*(x-x0)*(y-y0) + c*((y-y0)**2))
    )
    
    return z.ravel()

def fit_gaussian_2d(image):
    """
    Fit a 2D Gaussian to the input image
    
    Parameters:
    -----------
    image : ndarray
        Input image
    
    Returns:
    --------
    popt : ndarray
        Optimal parameters (amplitude, x0, y0, sigma_x, sigma_y, theta, offset)
    pcov : ndarray
        Covariance matrix for the parameters
    """
    # Create x and y indices
    y, x = np.indices(image.shape)
    
    # Initial guess for parameters
    height = np.max(image) - np.min(image)
    offset = np.min(image)
    
    # Find the peak position
    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)
    
    # Initial guess for width
    sigma_x = sigma_y = np.sqrt(np.sum((image - offset) * ((x - x_max)**2 + (y - y_max)**2)) / np.sum(image - offset))
    
    # Initial parameters
    initial_guess = [height, x_max, y_max, sigma_x, sigma_y, 0, offset]
    
    # Bounds for the parameters
    # (amplitude, x0, y0, sigma_x, sigma_y, theta, offset)
    lower_bounds = [0, 0, 0, 0, 0, -np.pi/2, -np.inf]
    upper_bounds = [np.inf, image.shape[1], image.shape[0], image.shape[1], image.shape[0], np.pi/2, np.inf]
    bounds = (lower_bounds, upper_bounds)
    
    # Fit the data
    popt, pcov = curve_fit(
        gaussian_2d, 
        (x, y), 
        image.ravel(), 
        p0=initial_guess,
        bounds=bounds
    )
    
    return popt, pcov



def preprocess(img):


    # make sure the img is a float
    img = img.astype(np.float64)

#    fig, ax = plt.subplots(1,3, figsize=(12, 4), sharex=True, sharey=True)
    scaler = ZScaleInterval()
    limits = scaler.get_limits(img)

#    ax[0].imshow(img, origin='lower', vmin=limits[0], vmax=limits[1])
#    print(img)


    # determine background

    shape = np.shape(img)
#    print(shape)
    kernel_size = min(shape)
    if kernel_size % 2 == 0:
        kernel_size += 1

    result = cv.GaussianBlur(img, (kernel_size, kernel_size), 0)

#    ax[1].imshow(result, origin='lower', vmin=limits[0], vmax=limits[1])

    img_bkg = img-result

#    limits = scaler.get_limits(img_bkg)
#    ax[2].imshow(img_bkg, origin='lower', vmin=limits[0], vmax=limits[1])

#    plt.show()

    img_bkg += np.min(img_bkg)

    return img_bkg


def get_noise_level(img, mask=None):
    if mask is None:
        return np.nanstd(img)

    return np.nanstd(img[mask])

def load_image(fname):

    hdu = fits.open(fname)

    image_data = hdu[0].data

    # debayer
    color_image = cv.cvtColor(image_data, cv.COLOR_BayerBG2BGR)

    lum = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)

    lum = preprocess(lum)
    print(np.shape(lum[300:-300, 400:-400]))
    
    return lum[200:-200, 200:-200]

    
def pedastal(img):

    pos_img = img[img!=0]

    return img - np.nanmedian(pos_img), img!=0

def get_star_locs(img, sigma=5, return_image=False, padding=1):
    starttime = time.time()

#    img = fits.getdata(fname)
#    img = load_image(fname)

#    scaler = ZScaleInterval()
#    limits = scaler.get_limits(img)

    # subtract pedastal
    normed_img, mask = pedastal(img)

    noise = get_noise_level(normed_img, mask=mask)

    masked_img = (normed_img).copy()
    masked_img[masked_img < sigma*noise] = 0
    masked_img[masked_img > 0] = 1

    labels_img = cv.threshold(masked_img, 0, 1, cv.THRESH_BINARY)[1]
    num_labels, labels_img = cv.connectedComponents(masked_img.astype(np.int8), connectivity=4)

    stars = {}

    coo = sparse.coo_matrix(labels_img)

    # go through each of the new labels
    for ii in range(1, num_labels):
        this_coo = (coo==ii).tocoo()
        xinds, yinds = this_coo.row, this_coo.col

        if len(xinds) <= 8:
            continue

        left = np.min(xinds)
        bottom = np.min(yinds)
        width = np.max(xinds)-left+1
        height = np.max(yinds)-bottom+1


        cutout = img[left:left+width, bottom:bottom+height]

        if np.any(cutout >= 2**16-1):
            continue

        if left < padding or bottom < padding:
            continue

        stars[ii] = (left-padding,
                     bottom-padding,
                     width+2*padding,
                     height+2*padding)


    num_stars = len(stars)

    finalxs = []
    finalys = []


    for inum, ii in enumerate(stars.keys()):
        left   = stars[ii][0]
        bottom = stars[ii][1]
        width  = stars[ii][2]
        height = stars[ii][3]

        cutout = img[left:left+width, bottom:bottom+height]

            
        try:
            popt, pcov = fit_gaussian_2d(cutout)
        except (RuntimeError, ValueError) as e:
            print("ERROR", e)
            continue
        x = popt[1]
        y = popt[2]

        finalxs.append(float(bottom+x))
        finalys.append(float(left+y))

#    fig, ax = plt.subplots()
#    ax.imshow(img)
#    for x, y in zip(finalxs, finalys):
#        ax.scatter(x, y, color='white', facecolor='none', linewidth=2)
#
#    plt.show()



    if len(finalxs) < 15:
        print("Not enough stars, only found ", len(finalxs))
        return None, None, None

    if return_image:
        return np.array(finalxs), np.array(finalys), img
    else:
        return np.array(finalxs), np.array(finalys), None




if __name__ == "__main__":
    fname = "/Users/michael/ASICAP/CapObj/2025-04-17_03_46_06Z/2025-04-17-0346_1-CapObj_2000.FIT"
#    fname = "/Users/michael/ASICAP/CapObj/2025-04-17_03_46_06Z/2025-04-17-0346_1-CapObj_2000.FIT"
#    xs2, ys2, img2 = get_star_locs('./test_data/0015.fit')

    img = load_image(fname)
    img = preprocess(img)
    xs, ys, img = get_star_locs(img, return_image=True)
    print(img)
    fig, ax = plt.subplots()
    ax.imshow(img)
    for x, y in zip(xs, ys):
        ax.scatter(x, y, color='white', facecolor='none', linewidth=2)

    plt.show()
