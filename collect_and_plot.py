#!/usr/bin/env python3
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from urllib.request import urlopen,urlretrieve
import requests
import json
from bs4 import BeautifulSoup
from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz,Angle,ITRS
from lofarantpos import db
from scipy import interpolate
import os
from reduced_data_lib import *
import RMextract.getIONEX as ionex
ionex_server = "ftp://gssc.esa.int/gnss/products/ionex/"
ionex_prefix='UQRG'
ionexPath="/home/mevius/IONEX_DATA/"
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
dummytime=Time.now()
def reformat_date(datestr):
    
    month = datestr.split("-")
    month[1] = months.index(month[1])+1
    month = [str(i) for i in month[::-1]]
    return "-".join(month)

# new pp  calculations from TJ Dijkema

def far_to_near(altaz_far, distance, obstime=dummytime):
    """Add distance to AltAz instance"""
    if obstime is None:
        obstime = altaz_far.obstime
    return AltAz(
        location=altaz_far.location,
        obstime=obstime,
        az=altaz_far.az,
        alt=altaz_far.alt,
        distance=distance,
    )


def distance_from_height(obs_altaz, height=300 * u.km):
    """From an AltAz pointing, find the distance to where it crosses a plane height km above Earth"""
    try_distances = np.linspace(height.to_value(), 2*np.max(obs_altaz.secz)*height.to_value()) * height.unit
    if np.any(try_distances<0):
        return [height]
    try_altaz = far_to_near(obs_altaz[np.newaxis], try_distances[:,np.newaxis], obstime=dummytime)
    try_heights = EarthLocation(
        *(try_altaz.transform_to(ITRS(obstime=dummytime)).cartesian.xyz)
    ).height
    distances = []
    for i in range(try_heights.shape[1]):
        height_to_distance = interpolate.interp1d(
            try_heights[:,i].to(u.km), try_distances[:].to(u.km),
            fill_value="extrapolate"
        )
        distances.append( height_to_distance(height) * u.km)

    return distances


def earthloc_at_height(obs_altaz, height=300 * u.km, obstime=dummytime):
    """From an AltAz pointing, find the EarthLocation at which point the line of sight is exactly at a given height (100km)"""
    distance = distance_from_height(obs_altaz, height)
    intersection_altaz = far_to_near(obs_altaz, distance)
    return EarthLocation(
        *(intersection_altaz.transform_to(ITRS(obstime=dummytime)).cartesian.xyz)
    )


mydb=db.LofarAntennaDatabase()
CS001=EarthLocation.from_geocentric(*list(mydb.phase_centres["CS001LBA"]),unit=u.m)
lon,lat =  CS001.lon.value,CS001.lat.value

def get_loc(station):
    return EarthLocation.from_geocentric(*list(mydb.phase_centres[station]),unit=u.m)
def get_loc_from_pos(statpos):
    return EarthLocation.from_geocentric(*statpos,unit=u.m)

def get_stat_name(fname):
    short = fname.split("_LBA_OUTER")[0]
    return short.split("_")[-1]

def download_fits(url,pattern,datapath= "/home/mevius/SpaceWeather/Pithia/data/processed_Data/",overwrite=False):
    print ("opening",url)
    page = urlopen(url)
    soup = BeautifulSoup(page.read(),features="lxml")
    fitsfiles = [  node.get('href') for node in soup.find_all('a') if node.get("href").endswith(".fits") and pattern in node.get("href") ]
    filelist = []
    if not datapath.endswith("/"):
        datapath+="/"
    for fitsfile in fitsfiles:
        if not overwrite and not os.path.isfile(datapath+fitsfile):
            urlretrieve(url+fitsfile, datapath + fitsfile )
        else:
            print("file ",datapath+fitsfile,"exists")
        filelist.append(datapath + fitsfile)
    return filelist

def get_obsid(url):
    page = urlopen(url)
    soup = BeautifulSoup(page.read(),features="lxml")
    return [  node.get('href') for node in soup.find_all('a') if not ".." in node.get('href')][0]

def get_iono_info(fitsfiles, do_tec=False):
    S4time = {}
    Ptime = {}
    bptime = {}
    Ptimes = {}
    atimes = {}
    azel = {}
    azelS4 = {}
    flags = {}
    pp = {}
    pp120 = {}
    ppS4 = {}
    ppS4120 = {}
    tec={}
    freqs = {}
    fitsfiles = sorted(fitsfiles)
    source=''
    dt = -1
    dtS4 = -1
    for fname in fitsfiles:
      try:
        hdul, header =  open_fits_file(fname)
        stat_name = get_stat_name(fname)
        stat_pos = get_loc(stat_name)
        print (stat_name,stat_pos,CS001)
        start_time = get_obstime_start(header)
        tmpsource = get_source(header)
        if not tmpsource==source:
            source = tmpsource
            if not source in S4time.keys():
                S4time[source]=[]
                Ptime[source]=[]
                bptime[source]=[]
                Ptimes[source]=[]
                atimes[source]=[]
                azelS4[source]=[]
                azel[source]=[]
                flags[source]=[]
        freqs[source] = get_freq_range(header)
        S4 = get_S4(hdul,"60S")
        flags[source] += [get_flags(hdul)[0]]
        Ptime[source] += [get_spectra(hdul)]
        Ptimes[source] += [get_time_range(header)]
        bptime[source] += [get_bandpass(hdul)[0]]
        atimes[source] += [get_time_range(S4[1])]
        S4time[source] += [S4[0] ] 
        azelS4[source] += [get_azel_range(header, times = get_time_range(S4[1]), location = stat_pos)]        
        azel[source] += [get_azel_range(header, times = get_time_range(header), location = stat_pos)]        
        if dt==-1:
            dt = start_time.mjd-Ptimes[source][0][0].mjd
        if dtS4==-1:
            dtS4 = start_time.mjd-atimes[source][0][0].mjd
      except:
          print("one of the fitsfiles failed",fname,"skipping")
          continue
    for src in S4time.keys():
        S4time[src] = np.concatenate(S4time[src],axis=1)
        Ptime[src] = np.concatenate(Ptime[src],axis=1)
        flags[src] = np.concatenate(flags[src],axis=1)
        bptime[src] = np.concatenate(bptime[src],axis=1)
        azlist = [i[0] for i in azel[src]]
        az = np.concatenate(azlist)
        ellist = [i[1] for i in azel[src]]
        el = np.concatenate(ellist)
        azlistS4 = [i[0] for i in azelS4[src]]
        azS4 = np.concatenate(azlistS4)
        ellistS4 = [i[1] for i in azelS4[src]]
        elS4 = np.concatenate(ellistS4)
        azel[src] = np.concatenate((az[:,np.newaxis],el[:,np.newaxis]),axis=1)
        azelS4[src] = np.concatenate((azS4[:,np.newaxis],elS4[:,np.newaxis]),axis=1)
        dta = [(i+dtS4).datetime for i in atimes[src]]
        atimes[src] = np.concatenate(dta)
        dtP = [(i+dt).datetime for i in Ptimes[src]]
        Ptimes[src] = np.concatenate(dtP)
        #try:
        if True:
            #pp[src]=earthloc_at_height(AltAz(az=az*u.deg,alt = (np.remainder(180.-el,180.)-90)*u.deg,location = stat_pos), height=350 * u.km, obstime=Time.now())
            ppS4[src]=earthloc_at_height(AltAz(az=azS4*u.deg,alt = elS4*u.deg,location = stat_pos), height=350 * u.km, obstime=Time.now())
            pp[src]=earthloc_at_height(AltAz(az=az*u.deg,alt = el*u.deg,location = stat_pos), height=350 * u.km, obstime=Time.now())
        #except:
        if False:
            print("pp failed,using CS001")
            pp[src]=CS001
            ppS4[src]=CS001
        if True:
        #try:
            #pp120[src]=earthloc_at_height(AltAz(az=az*u.deg,alt = (np.remainder(180.-el,180.)-90)*u.deg,location = stat_pos), height=120 * u.km, obstime=Time.now())
            ppS4120[src]=earthloc_at_height(AltAz(az=azS4*u.deg,alt = elS4*u.deg,location = stat_pos), height=120 * u.km, obstime=Time.now())
            pp120[src]=earthloc_at_height(AltAz(az=az*u.deg,alt = el*u.deg,location = stat_pos), height=120 * u.km, obstime=Time.now())
        if False:
        #except:
            print("pp failed,using CS001")
            pp120[src]=CS001
            ppS4120[src]=CS001
        if do_tec:
            tec[src] = gettec(atimes[src],ppS4[src])
            
    return {'P':Ptime,'times':Ptimes,'S4':S4time,'S4times':atimes,'azel':azel,'azelS4':azelS4,'bp':bptime,'flags':flags,'ppS4':ppS4, 'ppS4_120':ppS4120,'pp':pp, 'pp_120':pp120,'freqs':freqs,'tec':tec}


def gettec(times,pp):
    atimes = Time(times,format = "datetime")
    days = atimes.mjd.astype(int)
    days -= days[0]
    max_days = days[-1]+1
    idx_days = [np.where(days==i)[0] for i in range(max_days)]
    idx_days = [(i[0],i[-1]+1) for i in idx_days]
    vtecs=[]
    for stidx,endidx in idx_days:
        ionexf=ionex.getIONEXfile(time=atimes[stidx].ymdhms,server=ionex_server,prefix=ionex_prefix,outpath=ionexPath)
        tecinfo=ionex.readTEC(ionexf)
        seltimes = atimes[stidx:endidx]
        selpp = pp[stidx:endidx]
        vtecs.append(ionex.getTECinterpol(time=(seltimes.mjd - seltimes.mjd.astype(int))*24,lat=selpp.lat.deg,lon=selpp.lon.deg,tecinfo=tecinfo,apply_earth_rotation=0))
    return np.concatenate(vtecs)


def collect_all_data(url,datapath,outpath,stations = ["CS001LBA"],sources=['cas'],save_fig=True,show_fig=True,store_data=False):
    obsid = get_obsid(url)
    mypath = url + "/"+obsid
    if not "https" in mypath:
        print("no link found, skipping",splitted[0])
        return
    print ("retreiving data from",mypath)
    procesnr = mypath.split("/")[-1]
    if not len(procesnr):
        procesnr = mypath.split("/")[-2]
    print ("process",procesnr)
    for stat in stations:
        fitsfiles = download_fits(mypath+"/fits_files/",pattern=stat,datapath=datapath)
        if not fitsfiles:
            print ("station",stat,"not existing")
            continue
        info = get_iono_info(fitsfiles)
        for mysrc in sources:
            src = [i for i in info['P'].keys() if mysrc.lower() in i.lower()][0]
            fl = info['flags'][src]
            starttimes = info['times'][src]

            freqselect = np.mean(fl,axis=1)<0.05
            if not np.any(freqselect):
                freqselect  = np.mean(fl,axis=1)<0.05
                if not np.any(freqselect):
                    if np.median(fl)<1:
                        freqselect = [np.argmin(np.sum(fl,axis=1))]

                    else:
                        print("all data flagged, skipping")
                        continue
            procesdate = starttimes[0].strftime("%Y_%h_%d")
            if show_fig or save_fig:
                fig1,ax1 = plt.subplots()
                fig2,ax2 = plt.subplots()
                ax1.plot(info['times'][src],info['P'][src][freqselect].T)
                ax1.get_xaxis().set_major_formatter(mdates.DateFormatter('%d %b %H:00'))
                ax1.set_ylim(0.5,2)
                for label in ax1.get_xticklabels(which='major'):
                    label.set(rotation=30, horizontalalignment='right')
    

                fig1.suptitle(starttimes[0].strftime("%d %h %Y"))
                fig1.tight_layout()
                ax2.plot(info['S4times'][src],info['S4'][src][freqselect].T)
                ax2.get_xaxis().set_major_formatter(mdates.DateFormatter('%d %b %H:00'))
                ax2.set_ylim(0.,0.3)
                for label in ax2.get_xticklabels(which='major'):
                    label.set(rotation=30, horizontalalignment='right')
                fig2.tight_layout()

                fig2.suptitle(starttimes[0].strftime("%d %h %Y"))
                st = starttimes[0].strftime("%Y/%h/%d %H:%M:%S")
                et = starttimes[-1].strftime("%Y/%h/%d %H:%M:%S")

            lonst = str(info['pp'][src][0].lon.deg)
            lone = str(info['pp'][src][-1].lon.deg)
            latst = str(info['pp'][src][0].lat.deg)
            late = str(info['pp'][src][-1].lat.deg)
            if fig1 and save_fig:
                fig1.savefig(outpath+"/P_%s_%s_%s_%s.png"%(procesdate,procesnr,stat,src))
                fig2.savefig(outpath+"/S4_%s_%s_%s_%s.png"%(procesdate,procesnr,stat,src))
            if fig1 and show_fig:
                plt.show()
            if fig1:
                plt.close(fig1)
                plt.close(fig2)
            if store_data:
                np.savez(outpath+"/P_%s_%s_%s_%s.npz"%(procesdate,procesnr,stat,src),
                         lon=info['pp'][src].lon.deg,
                         lat=info['pp'][src].lat.deg,
                         lon_120=info['pp_120'][src].lon.deg,
                         lat_120=info['pp_120'][src].lat.deg,
                         lonS4=info['ppS4'][src].lon.deg,
                         latS4=info['ppS4'][src].lat.deg,
                         lonS4_120=info['ppS4_120'][src].lon.deg,
                         latS4_120=info['ppS4_120'][src].lat.deg,
                         power=info['P'][src][freqselect],
                         times=info['times'][src],
                         S4=info['S4'][src][freqselect],
                         S4times=info['S4times'][src],
                         freqs = info['freqs'][src][freqselect],
                         azel = info['azel'][src],
                         azelS4 = info['azelS4'][src]
                         )

def collect_flare_info(url,datapath,outpath,stations = ["CS001LBA"],sources=['cas'],save_fig=True,show_fig=True,store_data=False):
    fig1=None
    obsid = get_obsid(url)
    mypath = url + "/"+obsid
    if not "https" in mypath:
        print("no link found, skipping",splitted[0])
        return
    print ("retreiving data from",mypath)
    procesnr = mypath.split("/")[-1]
    if not len(procesnr):
        procesnr = mypath.split("/")[-2]
    print ("process",procesnr)
    for stat in stations:
        fitsfiles = download_fits(mypath+"/fits_files/",pattern=stat,datapath=datapath)
        if not fitsfiles:
            print ("station",stat,"not existing")
            continue

        info = get_iono_info(fitsfiles,do_tec=True)
        for i in info['P'].keys():
            print(i)
        for mysrc in sources:
            src = [i for i in info['P'].keys() if mysrc.lower() in i.lower()]
            if not len(src):
                print("source",mysrc,"not existing")
                continue
            src = src[0]
            
            fl = info['flags'][src]
            starttimes = info['times'][src]

            freqselect = np.mean(fl,axis=1)<0.005
            if not np.any(freqselect):
                freqselect  = np.mean(fl,axis=1)<0.01
                if not np.any(freqselect):
                    if np.median(fl)<1:
                        freqselect = [np.argmin(np.sum(fl,axis=1))]

                    else:
                        print("all data flagged, skipping")
                        continue
            procesdate = starttimes[0].strftime("%Y_%h_%d")
            if show_fig or save_fig:
                fig1,ax1 = plt.subplots()
                fig2,ax2 = plt.subplots()
                ax1.plot(info['times'][src],info['P'][src][freqselect].T)
                ax1.get_xaxis().set_major_formatter(mdates.DateFormatter('%d %b %H:00'))
                ax1.set_ylim(0.5,2)
                for label in ax1.get_xticklabels(which='major'):
                    label.set(rotation=30, horizontalalignment='right')
    

                fig1.suptitle(starttimes[0].strftime("%d %h %Y"))
                fig1.tight_layout()
                ax2.plot(info['S4times'][src],info['S4'][src][freqselect].T)
                ax2.get_xaxis().set_major_formatter(mdates.DateFormatter('%d %b %H:00'))
                ax2.set_ylim(0.,0.3)
                for label in ax2.get_xticklabels(which='major'):
                    label.set(rotation=30, horizontalalignment='right')
                fig2.tight_layout()

                fig2.suptitle(starttimes[0].strftime("%d %h %Y"))
                st = starttimes[0].strftime("%Y/%h/%d %H:%M:%S")
                et = starttimes[-1].strftime("%Y/%h/%d %H:%M:%S")

            lonst = str(info['pp'][src][0].lon.deg)
            lone = str(info['pp'][src][-1].lon.deg)
            latst = str(info['pp'][src][0].lat.deg)
            late = str(info['pp'][src][-1].lat.deg)
            if fig1 and save_fig:
                fig1.savefig(outpath+"/P_%s_%s_%s_%s.png"%(procesdate,procesnr,stat,src))
                fig2.savefig(outpath+"/S4_%s_%s_%s_%s.png"%(procesdate,procesnr,stat,src))
            if fig1 and show_fig:
                plt.show()
            if fig1:
                plt.close(fig1)
                plt.close(fig2)
            if store_data:
                np.savez(outpath+"/P_%s_%s_%s_%s.npz"%(procesdate,procesnr,stat,src),
                         lonS4=info['ppS4'][src].lon.deg,
                         latS4=info['ppS4'][src].lat.deg,
                         lonS4_120=info['ppS4_120'][src].lon.deg,
                         latS4_120=info['ppS4_120'][src].lat.deg,
                         power=info['P'][src][freqselect],
                         times=info['times'][src],
                         S4=info['S4'][src][freqselect],
                         S4times=info['S4times'][src],
                         freqs = info['freqs'][src][freqselect],
                         azelS4 = info['azelS4'][src],
                         tec = info['tec'][src]
                         )
   
def main(csv_file,separator=",",relative_path="Mclass",save_fig=True,show_fig=False,skip_exist=False):
    newf = open(csv_file.replace(".csv","_extra.csv"),"w")
    old_proces = ""
    fig1 = fig2 = None
    with open(csv_file) as myf:
         
         for line in myf:
            #if '3094' in line:
            #    continue
           
           #try: 

            if line.strip() and not line.strip()[0]=="#":
                splitted = line.strip().replace("\"","").split(separator)
                mypath = splitted[-1]
                if not "https" in mypath:
                    print("no link found, skipping",splitted[0])
                    continue
                print ("retreiving data from",mypath)
                datapath = "processed_Data/"+relative_path+"/"
                outpath = datapath +"images/"+mypath.split("/")[-1]
                if skip_exist and os.path.isdir(outpath):
                    continue
                os.makedirs(outpath, exist_ok=True)
                mypath = "/".join(mypath.split("/")[:-1])
                collect_flare_info(mypath,datapath,outpath,
                                   stations = ["CS001","DE602","DE603","DE604","DE605","FR606","IE6","LV6","PL610","PL611","PL612","UK6","SE6","DE609"],
                                   sources=['cas','cyg'],save_fig=save_fig,show_fig=show_fig,store_data=True)
