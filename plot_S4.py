from reduced_data_lib import *
import glob
import numpy as np

stations = ['LV6','SE6','DE609','CS001','UK6','IE6']

fnamelist = [sorted(glob.glob("/home/mevius/SpaceWeather/Pithia/data/processed_Data/2015_01_14/*%s*CasA*fits"%i)) for i in stations]

S4time = []
atimes = []
for CS001 in fnamelist:
 S4time.append([])
 atimes.append([])
 for fname in CS001:
    hdul, header =  open_fits_file(fname)
    S4 = get_S4(hdul,"180S")
    atimes[-1] += [get_time_range(S4[1])]
    S4time[-1] += [S4[0] ] 
S4list = [np.concatenate(i,axis=1) for i in S4time]
dt = [i.datetime for i in atimes[0]]
times = np.concatenate(dt)
for S4 in S4list: 
    plt.plot(times,S4[180].T)
plt.legend(stations)
plt.show()
