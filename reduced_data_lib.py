from astropy.io import fits
from astropy.time import Time,TimeDelta
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord, AltAz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime


def get_spectra(fitsfile):
    return fitsfile[0].data

def get_flags(fitsfile):
    return fitsfile['FLAG_PERCENTAGE'].data,fitsfile['FLAG_PERCENTAGE'].header

def get_S4(fitsfile, timewindow="60s"):
    return fitsfile['S4_%s'%timewindow].data,fitsfile['S4_%s'%timewindow].header

def get_bandpass(fitsfile):
    return fitsfile['BANDPASS'].data,fitsfile['BANDPASS'].header

def get_time_range(header):
    tmst = header['CRVAL1']
    atime = Time(tmst,format='unix')
    crpix = header['CRPIX1']
    nr_times = header['NAXIS1']
    dtime = header['CDELT1']*u.s
    atimes = atime + TimeDelta((np.arange(nr_times)-crpix)*dtime)
    return atimes

def get_obstime_start(header):
    '''get time range from description in header, since somehow there is an error in the CRVAL1'''
    tmst = header['TIME-OBS']
    datest = header['DATE-OBS']
    return Time(datest.replace("/","-")+'T'+tmst)
    
def get_freq_range(header):
    freqst = header['CRVAL2']*u.MHz
    crpix = header['CRPIX2']
    nr_freqs = header['NAXIS2']
    dfreq = header['CDELT2']*u.MHz
    freqs = freqst + (np.arange(nr_freqs)-crpix)*dfreq
    return freqs


def get_pos(header):
    x = header['REF_X']
    y = header['REF_Y']
    z = header['REF_Z']
    return x,y,z
    
def get_azel_range(header,times=None,location=None,sourcepos=None):
    if sourcepos is None:
        ra, dec = header['RA'], header['DEC']
        sourcepos = SkyCoord(ra=ra, dec=dec, unit='deg')
    if location is None:
        x,y,z = get_pos(header)
        location = EarthLocation(x=x, y=y, z=z, unit='m')
    if times is None:
        times = get_time_range(header)
    altaz = sourcepos.transform_to(AltAz(obstime=times,location=location))
    return altaz.az.deg, altaz.alt.deg


def get_source(header):
    return "_".join(header["OBJECT"].strip().split())
    
def open_fits_file(fname):
    hdul = fits.open(fname)
    header = hdul[0].header
    return hdul, header


def plot_data(fname):
    fitsf, header = open_fits_file(fname)
    times = get_time_range(header)
    freqs = get_freq_range(header).value
    flags,_ = get_flags(fitsf)
    bp,_ = get_bandpass(fitsf)
    data = get_spectra(fitsf)
    S4,_ = get_S4(fitsf,'60s')
    
    fig,axs = plt.subplots(4,1,figsize=(20,40))
    flagged_data = np.copy(data)
    flagged_data[flags>0] = np.nan
    cmap = plt.cm.inferno.copy()
    cmap.set_bad((1, 0, 0, 1))
    date_format = mdates.DateFormatter('%H:%M')
    xlims = mdates.date2num(list(map(datetime.datetime.fromtimestamp,[times.unix[0],times.unix[-1]])))
    img = axs[0].imshow(data,origin="lower",interpolation="none",aspect="auto",vmin=0.7,vmax=1.5 , extent = [xlims[0],xlims[1],freqs[0],freqs[-1]],cmap="inferno")
    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='1%', pad=0.05)
    axs[0].set_ylabel("MHz")
    axs[0].xaxis.set_major_formatter(date_format)
    fig.colorbar(img,cax)
    img = axs[1].imshow(flagged_data,origin="lower",interpolation="none",aspect="auto",vmin=0.7,vmax=1.5 , extent = [xlims[0],xlims[1],freqs[0],freqs[-1]],cmap=cmap)
    divider = make_axes_locatable(axs[1])
    cax = divider.append_axes('right', size='1%', pad=0.05)
    fig.colorbar(img,cax)
    axs[1].set_ylabel("MHz")
    axs[1].xaxis.set_major_formatter(date_format)
    img = axs[2].imshow(np.log10(flagged_data*bp),origin="lower",interpolation="none",aspect="auto", extent = [xlims[0],xlims[1],freqs[0],freqs[-1]],cmap=cmap)
    divider = make_axes_locatable(axs[2])
    cax = divider.append_axes('right', size='1%', pad=0.05)
    fig.colorbar(img,cax)
    axs[2].set_ylabel("MHz")
    axs[2].xaxis.set_major_formatter(date_format)
    img = axs[3].imshow(S4,origin="lower",interpolation="none",aspect="auto",vmin=0.,vmax=0.2 , extent = [xlims[0],xlims[1],freqs[0],freqs[-1]],cmap='plasma')
    divider = make_axes_locatable(axs[3])
    cax = divider.append_axes('right', size='1%', pad=0.05)
    fig.colorbar(img,cax)
    axs[3].set_ylabel("MHz")
    axs[3].xaxis.set_major_formatter(date_format)
    plt.show()
