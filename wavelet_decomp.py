import numpy as np
from joblib import Parallel, delayed
import multiprocessing as mp
from numba import jit,vectorize
import gc

@vectorize()
def haar(x):
    """
    Definition of a step-function on a unit interval x in [0,1].
    """
    if x>=0 and x<=0.5:
        return 1
    elif x > 0.5 and x<=1:
        return -1
    else:
        return 0

@jit(cache = True)
def d_haar(x,j,k):
    """
    Definition of the Haar wavelet. Takes scale j and translation parameter
    k as inputs, in addition to x. Outputs the Haar wavelet corresponding to
    j and k for each value in x-vector.
    """
    return 2**(j/2)*haar(2**j*x - k)
@jit(cache = True)
def decomp(f,x,jmax):
    """
    Method for decomposing an arbitrary field f (input) with the Haar wavelet
    decomposition for all scales up to scale jmax. Returns the wavelet
    decomposition of field f for all k and j upto jmax.
    """
    j_array = np.arange(0,jmax +1,1)
    c_array = []
    for j in j_array:
        kmax = 2**j
        temp_array = []
        for k in range(kmax):
            temp_array.append(np.mean(f*d_haar(x,j,k)))
        if temp_array == []:
            temp_array = [0]
        c_array.append(np.array(temp_array))
    return np.array(c_array)
@jit(cache = True,parallel = True)
def comp(c,x):
    """
    Method for recomposing a field from the Haar wavelet decomposition c.
    Returns the recomposed field f on the domain x.
    """
    j_array = np.arange(0,len(c),1)
    f = np.zeros(x.shape)
    for j,ck in zip(j_array,c):
        k_array = np.arange(0,len(ck),1)
        for ck_a,k in zip(ck,k_array):
            f+= ck_a*d_haar(x, j, k)
    return f


@jit(cache = True)
def loop_scale_spectrum(c1,c2):
    """
    Jit-ed loop used in the computation of the (cross) scale spectrum of
    the wavelet decompositions c1 and c2. Returns the spectrum corresponding
    to each of the scales incorporated in c1 and c2
    """
    c = []
    for d in c1*c2:
        c.append(np.sum(d))
    return c

def compute_scale_spectrum(data1,data2,x, numscales = 5):
    """
    Method for computing the cross scalespectrum of a dataset with respect to
    the discrete wavelet series expansion.
    Data input has to be a 1D array. Returns numscales+2 scale coefficients.
    x is the spatial extent where the data points are defined.
    """
    #Removing means
    m1 = np.mean(data1)
    m2 = np.mean(data2)
    d1 = data1 - m1
    d2 = data2 - m2
    #Computation of coefficients
    c_1 = decomp(d1,x,numscales)
    c_2 = decomp(d2,x,numscales)

    cross_scalogram = np.array(loop_scale_spectrum(c_1,c_2))
    cross_scalogram = np.concatenate([[m1*m2],cross_scalogram])
    return cross_scalogram

def compute_scale_spectrum_latlon(data1,data2,x,numscales = 5):
    """
    Executes compute_scale_spectrum() for each latitude on a latlon gridded
    dataset.
    Returns a cross scalogram for each latitude in a Nlat X numscales+1
    2D array. Perform on data where 1st dimension is latitude and second
    dimension is longitude.
    """
    # Initialization of array for scalograms
    s = np.full((numscales + 2,data1.shape[0]),-999,np.double)
    Nlat = data1.shape[0]
    for i in range(Nlat):
        s[:,i] = compute_scale_spectrum(data1[i,:],data2[i,:],x,numscales)
        gc.collect()
    gc.collect()
    return s

#@jit(cache = True)
def compute_scale_transport_levels(data1,data2,x,numscales = 5):
    """
    Execution of scalespectrum for each level in a data set.
    Returns the sum of the levels.
    Input data shaped (Nlev, Nlat,Nlon). Outputdata shaped (numscales +2, Nlat)
    """
    nproc = mp.cpu_count()
    Nlev = data1.shape[0]
    parallel = Parallel(n_jobs = -1,backend = 'loky')(delayed(compute_scale_spectrum_latlon)(data1[i,:,:],data2[i,:,:],x,numscales) for i in range(Nlev))
    sl = np.sum(np.array(parallel),axis = 0)
    gc.collect()
    return sl

#@processify
def data_from_NC(ncf,time,var):
    if var == 'Ex':
        return np.array(ncf['Q'][time,:,:,:])
    elif var == 'Tv':
        return np.array(ncf['T'][time,:,:,:])
    elif var == 'V':
        return np.array(ncf['V'][time,:,:,:])
    elif var == 'Q':
        return np.array(ncf['Q'][time,:,:,:])
