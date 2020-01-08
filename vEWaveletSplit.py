import numpy as np
import netCDF4 as nc4
import gc
import psutil
import sys
#from SHfilter import SHfilter
from math import pi
import xarray as xr
from Z_calc import calculategeoh,calc_dp
from wavelet_decomp import compute_scale_transport_levels as wavelet_transport
from wavelet_decomp import data_from_NC
import joblib as jb
# Reading year and month from command line inputs
year = sys.argv[1]
month = sys.argv[2]
Nscales = 5

# Assigning paths to variables
pathp = "/gpfs/scratch/ms/no/fath"
path = pathp + "/" + year + "/" + month
paths = pathp +"/EnergySplit_W2"

fV = path + "/V.nc"
fLQ = path + "/LQ.nc"
fVerr = path + "/Verr." + year + "." + month + ".nc"

# Reading data
Vf = nc4.Dataset(fV,'r')
LQf = nc4.Dataset(fLQ,'r')
Verrf = nc4.Dataset(fVerr,'r')

g = 9.80665
ai = Tf0['hyai']
bi = Tf0['hybi']
am = Tf0['hyam']
bm = Tf0['hybm']
p0 = 100000
ps = np.exp(LnSP['LNSP'])
Z0 = Z0f['Z']
Verr = Verrf['Verr']

Nlat = len(Tf0['lat'])
Nlon = len(Tf0['lon'])
Ntime = len(Tf0['time'])
Nlev = len(am)

lon = Tf0['lon']

#Allocating wave-arrays
vQtot = xr.DataArray(np.zeros((Nscales+2,Ntime,Nlat)),dims = ['Scale','time','lat'],coords = [np.arange(0,Nscales+2),np.arange(0,Ntime),Tf0['lat'][:]])
vQtot.name = 'vQtot'

# Time loop
for itime in range(Ntime):
    print('itime: ' + str(itime))
    gc.collect()
    V = data_from_NC(Vf,itime,'V')
    Q = data_from_NC(LQf,itime,'Q')
    gc.collect()

    dP = calc_dp(ps[itime,0,:,:],ai,bi)
    gc.collect()

    for ilev in range(Nlev):
        V[ilev,:,:] = V[ilev,:,:] - Verr[itime,:,:]

    gc.collect()

    V *= dP

    gc.collect()
    vEtot[:,itime,:] = wavelet_transport(V,E,np.linspace(0,1,len(Tf0['lon'])),numscales = Nscales)/g
    vQtot[:,itime,:] = wavelet_transport(V,Q,np.linspace(0,1,len(Tf0['lon'])),numscales = Nscales)/g
    gc.collect()


print('Month loop done')
vQtot.to_netcdf(paths + "/Waves/vQtot." + year + "." + month + ".NS"+str(Nscales) + ".nc")
print('DONE')
