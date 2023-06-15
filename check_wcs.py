import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from astropy import wcs
from astropy.io import fits
from astropy import units as u
from astroquery.gaia import Gaia
from astroquery.mast import Catalogs
from astropy.coordinates import SkyCoord

import sep
import aplpy
import warnings
from contextlib import contextmanager
from astropy.utils.exceptions import AstropyWarning

sep.set_sub_object_limit(1e4)

from matplotlib.font_manager import findSystemFonts
font_family = 'serif'
font_families = findSystemFonts(fontpaths=None, fontext='ttf')
for family in font_families:
    if 'GFSArtemisia' in family:
        font_family = "GFS Artemisia"
        break
        
@contextmanager
def suppress_stdout():
    """Suppresses annoying outputs.

    Useful with astroquery and aplpy packages.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            
def plot_objects(hdu, x, y, radius=20, coords_frame='pixel', title=None):
    """Plots sources on a FITS image.
    
    Parameters
    ----------
    hdu: ~hdu
        Header Data Unit.
    x: array-like or ~SkyCoord
        x-axis coordinates in pixel units or right ascension.
    y: array-like or ~SkyCoord
        y-axis coordinates in pixel units or declination.
    radius: float, default ``20``
        Radius of "aperture" in pixel units or in world units.
    coords_frame: str, default ``pixel``
        Coordinates frame. Either 'pixel' or 'world'.
    title: str, default ``None``
        Title for the plot.
    """
    figure = plt.figure(figsize=(10, 10))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        fig = aplpy.FITSFigure(hdu, figure=figure)

    with suppress_stdout():
        fig.show_grayscale(stretch="arcsinh")

    fig.show_circles(
        x,
        y,
        radius,
        coords_frame=coords_frame,
        linewidth=2,
        edgecolor="r",
    )

    # ticks
    fig.tick_labels.set_font(**{"family": font_family, "size": 18})
    fig.tick_labels.set_xformat("dd.dd")
    fig.tick_labels.set_yformat("dd.dd")
    fig.ticks.set_length(6)

    fig.axis_labels.set_font(**{"family": font_family, "size": 18})

    if title is not None:
        fig.set_title(title, **{"family": font_family, "size": 24})
    fig.set_theme("publication")

    plt.show()
    
def find_gaia_objects(ra, dec, rad=0.15):
    """Finds objects using the Gaia DR3 catalog for the given
    coordinates in a given radius.

    Parameters
    ----------
    ra: float
        Right ascension in degrees.
    dec: float
        Declination in degrees.
    rad: float, default ``0.15``
        Search radius in degrees.

    Returns
    -------
    gaia_coords: SkyCoord object
        Coordinates of the objects found.
    """
    Gaia.MAIN_GAIA_TABLE = "gaiaedr3.gaia_source"
    Gaia.ROW_LIMIT = -1
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.degree, u.degree), frame="icrs")
    width = u.Quantity(rad, u.deg)
    height = u.Quantity(rad, u.deg)
    try:
        with suppress_stdout():
            gaia_cat = Gaia.query_object_async(
                coordinate=coord, width=width, height=height
            )
    except Exception as exc:
        print(exc)
        print("No objects found with Gaia DR3, switching to DR2")
        Gaia.MAIN_GAIA_TABLE = "gaiadr2.gaia_source"
        gaia_cat = Gaia.query_object_async(
            coordinate=coord, width=width, height=height
        )

    gaia_ra = np.array(gaia_cat["ra"].value)
    gaia_dec = np.array(gaia_cat["dec"].value)
    gaia_coords = SkyCoord(
        ra=gaia_ra, dec=gaia_dec, unit=(u.degree, u.degree), frame="icrs"
    )

    return gaia_coords

def cross_match_sources(objects_coords1, objects_coords2, sep_threshold=3):
    """Cross-matches a given set of sources with another set of sources.
    
    Parameters
    ----------
    objects_coords1: ~SkyCoord
        Coordinates of first set of sources.
    objects_coords2: ~SkyCoord
        Coordinates of second set of sources.
    sep_threshold: float, default ``3``
        Maximum separation in arcsec to successfully 
        cross-match a given source with a catalog one.
        
    Returns
    -------
    matching_coords: ~SkyCoord
        Coordinates of successfully cross-matched sources.
    """    
    indeces = []
    for obj_coords in objects_coords1:
        separation = objects_coords2.separation(obj_coords)
        sep_arcsec = separation.to(u.arcsec).value

        if sep_arcsec.min() <= sep_threshold:
            index = np.argmin(sep_arcsec)
            indeces.append(index)

    matching_coords = objects_coords2[indeces]
    if len(matching_coords) < 5:
        warnings.warn((f'Less than 5 sources ({len(matching_coords)}) were found with'
                       f' a separation of {sep_threshold} arcsec or less'))
    
    return matching_coords

def cross_match_catalog(ra, dec, objects_coords, sep_threshold=3):
    """Cross-matches given sources with the Gaia catalog.
    
    Parameters
    ----------
    ra: float
        Right ascension in degrees.
    dec: float
        Declination in degrees.
    objects_coords: ~SkyCoord
        Coordinates of sources.
    sep_threshold: float, default ``3``
        Maximum separation in arcsec to successfully 
        cross-match a given source with a catalog one.
        
    Returns
    -------
    matching_coords: ~SkyCoord
        Coordinates of successfully cross-matched sources.
    """
    gaia_coords = find_gaia_objects(ra, dec)
    matching_coords = cross_match_sources(objects_coords, gaia_coords, 
                                          sep_threshold)
    
    return matching_coords

def extract_sources(file, sources_threshold=15):
    """Extracts source from a FITS image.
    
    Parameters
    ----------
    file: str
        FITS image file.
    sources_threshold: float, default ``3``
        Threshold used to detect source on the given image. 

    Returns
    -------
    objects: ndarray
        Extracted sources.
    img_wcs: ~WCS
        Image's WCS.
    """
    # load data, header and WCS
    hdu = fits.open(file)
    header = hdu[0].header
    data = hdu[0].data
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        img_wcs = wcs.WCS(header, naxis=2)

    # detect sources
    data = data.astype(np.float64)
    bkg = sep.Background(data)
    bkg_rms = bkg.globalrms
    objects = sep.extract(data, sources_threshold, err=bkg_rms)
    
    return objects, img_wcs

def check_wcs(file, ra=None, dec=None, plot=False, 
              sources_threshold=15, sep_threshold=3):
    """Checks if the WCS (astrometry) of an image is correct.
    
    Parameters
    ----------
    file: str
        FITS image file.
    ra: float or str
        Right ascension in degrees or keyword from header.
    dec: float or str
        Declination in degrees or keyword from header.
    plot: bool
        If ``True``, cross-matched sources are plotted.
    sources_threshold: float, default ``3``
        Threshold used to detect source on the given image. 
    sep_threshold: float, default ``3``
        Maximum separation in arcsec to successfully 
        cross-match a given source with a catalog one.
    """
    # detect sources
    objects, img_wcs = extract_sources(file, sources_threshold)
        
    # load coordinates
    if ra is None and dec is None:
        ra, dec = header['RA'], header['DEC']
    elif isinstance(ra, str) and isinstance(dec, str):
        ra, dec = header[ra], header[dec]
    
    # cross-match with Gaia catalog
    objects_coords = img_wcs.pixel_to_world(objects['x'], objects['y'])
    matching_coords = cross_match_catalog(ra, dec, objects_coords, sep_threshold)
    
    if plot is True:
        hdu = fits.open(file)
        matching_coords_pixels = img_wcs.world_to_pixel(matching_coords)
        title = f'Cross-matched sources (sep $\leq$ {sep_threshold} arcsec)'
        plot_objects(hdu, matching_coords_pixels[0], matching_coords_pixels[1], title=title)