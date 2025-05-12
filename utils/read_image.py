from astropy.io import fits
from astropy.wcs import WCS



def read_solved_image(fname):
    hdu = fits.open(fname)
    img_data = hdu[0].data
    wcs = WCS(hdu[0].header)

    return img_data, wcs
