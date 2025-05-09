from astropy.visualization import ZScaleInterval
import matplotlib.pyplot as plt
import numpy as np

def plot_one(image, invert=True):

    scaler = ZScaleInterval()

    # do some preprocessing
    image = np.nan_to_num(image, posinf=0, neginf=0)
    image -= np.min(image)

    limits = scaler.get_limits(image)

    fig, ax = plt.subplots()
    print(limits)
    cmap="Greys_r"
    if invert:
        cmap="Greys"
    
    ax.imshow(image, 
              aspect='auto', 
              origin='lower', 
              vmin=limits[0], vmax=limits[1]+0.1,
              cmap=cmap)

    return ax
