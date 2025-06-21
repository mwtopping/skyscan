from glob import glob
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy.io import fits
import cv2 as cv

def fit_and_subtract_plane(image):
    """
    Fit a plane to an image and subtract it off
    """
    # Get image dimensions
    rows, cols = image.shape

    # Create coordinate grids
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Flatten arrays for fitting
    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = image.flatten()

    # Create design matrix for plane fitting (z = ax + by + c)
    A = np.column_stack([x_flat, y_flat, np.ones(len(x_flat))])

    # Solve for plane coefficients using least squares
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z_flat, rcond=None)

    # Generate the fitted plane
    fitted_plane = coeffs[0] * x + coeffs[1] * y + coeffs[2]

    # Subtract the plane from the original image
    corrected_image = image - fitted_plane

    return corrected_image, fitted_plane, coeffs


def fit_polynomial_surface(image, degree=2, mask=None):
    """
    Fit a 2D polynomial surface to an image

    Parameters:
    - image: 2D numpy array
    - degree: polynomial degree (1=plane, 2=quadratic, etc.)
    - mask: optional boolean mask (True = exclude from fitting)
    """
    rows, cols = image.shape
    x, y = np.meshgrid(np.arange(cols), np.arange(rows))

    # Apply mask if provided
    if mask is not None:
        valid_pixels = ~mask
        x_data = x[valid_pixels]
        y_data = y[valid_pixels]
        z_data = image[valid_pixels]
    else:
        x_data = x.flatten()
        y_data = y.flatten()
        z_data = image.flatten()

    # Generate all polynomial terms up to specified degree
    # For degree=2: [1, x, y, x², xy, y²]
    # For degree=3: [1, x, y, x², xy, y², x³, x²y, xy², y³]
    terms = []
    term_names = []

    for total_degree in range(degree + 1):
        for x_power in range(total_degree + 1):
            y_power = total_degree - x_power
            term = (x_data ** x_power) * (y_data ** y_power)
            terms.append(term)

            # Create readable term names for debugging
            if x_power == 0 and y_power == 0:
                term_names.append('1')
            elif x_power == 0:
                term_names.append(f'y^{y_power}' if y_power > 1 else 'y')
            elif y_power == 0:
                term_names.append(f'x^{x_power}' if x_power > 1 else 'x')
            else:
                x_part = f'x^{x_power}' if x_power > 1 else 'x'
                y_part = f'y^{y_power}' if y_power > 1 else 'y'
                term_names.append(f'{x_part}*{y_part}')

    # Create design matrix
    A = np.column_stack(terms)

    # Solve for coefficients
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z_data, rcond=None)

    # Generate fitted surface for entire image
    fitted_surface = np.zeros_like(image, dtype=float)
    term_idx = 0

    for total_degree in range(degree + 1):
        for x_power in range(total_degree + 1):
            y_power = total_degree - x_power
            fitted_surface += coeffs[term_idx] * (x ** x_power) * (y ** y_power)
            term_idx += 1

    # Subtract fitted surface
    corrected_image = image - fitted_surface

    return corrected_image, fitted_surface, coeffs, term_names

if __name__ == "__main__":
    DATA_DIR = "../data/2025-06-17_05_38_56Z"
    fnames = sorted(glob(f"{DATA_DIR}/*FIT"))

    scaler = ZScaleInterval(contrast=0.15)

    hdu = fits.open(fnames[0])
    img_data = hdu[0].data
    color_image = cv.demosaicing(img_data, cv.COLOR_BayerBG2BGR)
    lum = cv.cvtColor(color_image, cv.COLOR_BGR2GRAY)
    fig, axs = plt.subplots(1, 3)

    limits = scaler.get_limits(lum)
    lum_corr, bkg, coefs = fit_and_subtract_plane(lum)

    lum_corr2, bkg2, coeffs, names =  fit_polynomial_surface(lum, degree=3, mask=None)

    axs[0].imshow(lum, vmin=limits[0], vmax=limits[1], cmap="Greys_r", aspect='auto')
    limits = scaler.get_limits(lum_corr)
    axs[1].imshow(lum_corr, vmin=limits[0], vmax=limits[1], cmap="Greys_r", aspect='auto')

    from astropy.stats import SigmaClip
    from photutils.background import Background2D, MedianBackground
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg = Background2D(lum, (50, 50), filter_size=(3, 3),
                       sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

    #lum3 = lum.astype(np.float64) - bkg.background
    #limits = scaler.get_limits(lum3)
    #axs[2].imshow(lum3, vmin=-5000, vmax=20000, cmap="Greys_r", aspect='auto')
    #print(limits)
#   # axs[2].imshow(lum3, cmap="Greys_r", aspect='auto')
    #plt.show()
#    cv.imshow("first", color_image)
#    cv.waitKey(0)
#    height, width, channels = color_image.shape
#    print(height, width)
#    lum /= np.nanmedian(lum)
#    fourcc = cv.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
#    out = cv.VideoWriter("temp.mp4", fourcc, 30.0, (width, height))
    limits = None
    frameno = 0
    for ii in tqdm(range(len(fnames))[::2]):
        #fig, ax = plt.subplots(frameon=False)
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0., 0., 1., 1.])
        fname1 = fnames[ii]
        fname2 = fnames[ii+1]
        hdu1 = fits.open(fname1)
        img_data1 = hdu1[0].data
        hdu2 = fits.open(fname2)
        img_data2 = hdu2[0].data
        hdu1.close()
        hdu2.close()

        color_image1 = cv.demosaicing(img_data1, cv.COLOR_BayerBG2BGR)
        color_image2 = cv.demosaicing(img_data2, cv.COLOR_BayerBG2BGR)
        lum1 = cv.cvtColor(color_image1, cv.COLOR_BGR2GRAY)
        lum2 = cv.cvtColor(color_image2, cv.COLOR_BGR2GRAY)
        lum = 0.5*lum1 + 0.5 * lum2


        #print(np.nanmedian(lum))
        lum /= np.nanmedian(lum)

        bkg = Background2D(lum, (50, 50), filter_size=(3, 3),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

        lum = lum.astype(np.float64) - bkg.background

        if limits is None:
            limits = scaler.get_limits(lum)
        #cv.imshow("next", color_image)
        #cv.waitKey(0)
        #out.write(color_image)
        
        ax.imshow(lum, vmin=-0.4, vmax=2, cmap="Greys_r", aspect='auto')
        ax.set_axis_off()
        fig.savefig(f"./out/{frameno}.png", dpi=200)
        #plt.show()
        plt.close('all')
        frameno += 1
    #out.release()
    #cv.destroyAllWindows()
