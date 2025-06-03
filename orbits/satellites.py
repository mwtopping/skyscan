import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import os

from dotenv import load_dotenv


import numpy as np

from astropy.visualization import ZScaleInterval

import sys
sys.path.append("../utils/")
from find_stars import load_image, create_solved_image
from read_image import read_solved_image

from skyfield.api import load, wgs84
from skyfield.iokit import parse_tle_file

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


def in_view(ra, dec, wcs):
    footprint = wcs.calc_footprint()
    ras = footprint[:,0]
    decs = footprint[:,1]
    minra = np.min(ras)
    maxra = np.max(ras)
    mindec = np.min(decs)
    maxdec = np.max(decs)

    return (minra < ra < maxra) and (mindec < dec < maxdec)

def check_sat(args):

    sat, ii, myloc, tstart, wcs, shared_results = args
    difference = sat - myloc
    topocentric = difference.at(tstart)
    ra, dec, distance = topocentric.radec(epoch='date')
    if in_view(ra.degrees, dec.degrees, wcs):
        shared_results.append(ii)


#@line_profiler.profile
def get_nearby_satellites(image, wcs, header, sats, ts, plotting=False):


    if plotting:
        fig, ax = plt.subplots()
        scaler = ZScaleInterval()
        limits = scaler.get_limits(image)
        ax.imshow(image, origin='lower', vmin=limits[0], vmax=limits[1])

    print("Reading file")
#    allsats = pd.read_csv('./data/allsats_20250521_1230131364.csv')
#    print(allsats)
#    plt.figure()
#    plt.hist(allsats["PERIGEE"], bins=np.linspace(0, 100000, 1000))
#    plt.show()

    expstart = header["DATE-OBS"]
    exptime = header["EXPTIME"]

    load_dotenv("../.env")
    lat = float(os.getenv("LAT"))
    lon = float(os.getenv("LON"))
    myloc = wgs84.latlon(lat, lon, elevation_m=0)



#    tmp_sats= parse_tle_file(f, ts) 
#    differences = tmp_sats-myloc

    unique_sats = []
    sat_names = []

    #for sat in sats:
    #    if not sat.name in sat_names:
    #        unique_sats.append(sat)


    timestart = time.time()

     

    year, month, day, hour, min, sec = parse_expstart(expstart) 
    tstart = ts.utc(year, month, day, hour, min, sec-3)
    tend = ts.utc(year, month, day, hour, min, sec+float(exptime)-3)


    #ra_first, dec_first = wcs.all_pix2world(line[0][0], line[0][1], 0)

    from astropy.coordinates import SkyCoord
    import astropy.units as u

    travel_distances = []


    #def check_sat(shared_list, sat, tstart, myloc, wcs, index):
    final_list = []

#    with Manager() as manager:
#        shared_results = manager.list()
#
##        def check_sat(sat):
##            difference = sat - myloc
##            topocentric = difference.at(tstart)
##            ra, dec, distance = topocentric.radec(epoch='date')
##            if in_view(ra.degrees, dec.degrees, wcs):
##                shared_results.append(sat)
#        args = [(sat, ii, myloc, tstart, wcs, shared_results) for ii, sat in enumerate(sats)]
#
#        with Pool(processes=4) as pool:
#            pool.map(check_sat, args)
#
#        
#        final_list = list(shared_results)
#
#    print(final_list)
#        s = allsats[allsats["NORAD_CAT_ID"]==satnum]
#        
#        if len(s) > 0:
#            size = s.iloc[0]["RCS_SIZE"]
#            if size != "LARGE":
#                continue
#        else:
#            continue

    #        try:
    #        except:
    #            print(sat)
    #            exit()

#    positions = [sat.at(tstart) for sat in sats]
#    observer_position = myloc.at(tstart)
#    for ii, pos in tqdm(enumerate(positions), total=len(positions)):
#        difference = pos-observer_position
#        ra, dec, distance = difference.radec(epoch="date")

#    differences = sats-myloc

    movements = []
    found_coords = {}
    found_pixels = {}


    for sat in tqdm(sats):
        difference = sat - myloc
        topocentric = difference.at(tstart)
        ra, dec, distance = topocentric.radec(epoch="date")
        #ra, dec, distance = topocentric.radec(ts.J(2025.0))
        movements.append(206265*sat.model.no_kozai/60)
        if 206265*sat.model.no_kozai/60 < 100:
            continue
#        print(sat.epoch.utc_jpl())
#        print(sat.model.satnum)

        size = "LARGE"

#        if wcs.footprint_contains(c):
        if in_view(ra.degrees, dec.degrees, wcs):

#            print("FOUND ONE!")
            topocentric_end = difference.at(tend)
            ra2, dec2, distance = topocentric_end.radec(epoch='date')

            c = SkyCoord(ra=float(ra.degrees)*u.deg, dec=float(dec.degrees)*u.deg)
            c2 = SkyCoord(ra=float(ra2.degrees)*u.deg, dec=float(dec2.degrees)*u.deg)

            x, y = wcs.world_to_pixel(c)
            x2, y2 = wcs.world_to_pixel(c2)
            travel_distance = float(np.sqrt((x-x2)**2 + (y-y2)**2))
            if travel_distance < 1.0:
                continue
    #        ax.scatter(x, y, color='black')
            if plotting:
                if size == "LARGE":
                    color = 'black'
                elif size == "MEDIUM":
                    color='red'
                elif size == "SMALL":
                    color='blue'
                else:
                    color='cyan'

                ax.plot([x,x2], [y,y2], color=color, linewidth=0.8, alpha=0.8)
                ax.text(x, y, sat.name, fontsize=8, alpha=0.8)
            found_pixels[sat.model.satnum] = (float(x), float(y), float(x2), float(y2))
            found_coords[sat.model.satnum] = (float(ra.degrees), float(dec.degrees), float(ra2.degrees), float(dec2.degrees))

#    print(dir(sat))


    #x, y = wcs.world_to_pixel(c)
    #print(x, y)
    #
    #xpix, ypix = wcs.all_world2pix(ra.degrees, dec.degrees, 0)
    #print(xpix, ypix)


    #def check_loc(ra, dec, wcs):
    #footprint = wcs.calc_footprint()
    #ras = footprint[:,0]
    #decs = footprint[:,1]
    #minra = np.min(ras)
    #maxra = np.max(ras)
    #mindec = np.min(decs)
    #maxdec = np.max(decs)
    #
    #    pass
    #fig, ax = plt.subplots()
    #ax.hist(travel_distances, bins=np.linspace(0, 100, 100))
    return found_coords, found_pixels


def main():
    startfname = f"/Users/michael/ASICAP/CapObj/2025-04-17_03_46_06Z/2025-04-17-0346_1-CapObj_1003.FIT"
    solved_fname = create_solved_image(startfname, iterations=2)
    startimg, wcs, header = read_solved_image(solved_fname)


    ts = load.timescale()

    with load.open('./data/utc2025apr17_u.dat') as f:
        sats = list(parse_tle_file(f, ts))

    coords, pixels = get_nearby_satellites(startimg, wcs, header, sats, ts, plotting=True)
    print(coords)
    plt.show()


if __name__ == "__main__":
    main()
