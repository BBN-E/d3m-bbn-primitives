#import gzip
import numpy as np
#import h5py
#import os
import scipy
import scipy.io
import scipy.linalg as spl

def col(v):
  return v.reshape((v.size, 1))

def row(v):
  return v.reshape((1, v.size))

#################################################################################
#################################################################################
#def load_v_from_h5(file_name, dataset_name = 'v'):
#    f = h5py.File(file_name, 'r', driver='core')
#    return np.array(f[dataset_name])


################################################################################
################################################################################
def estimate_i(Nt, Ft, v, VtV=None, I=None, out=None):
    v_dim   = v.shape[1]
    n_gauss = Nt.shape[1]

    # compute VtV if necessary
    if VtV is None:
        VtV = compute_VtV(v, n_gauss)

    # Allocate space for out i-vec if necessary
    if out is None: 
        out = np.empty((1, v_dim), dtype=v.dtype)

    # Construct eye if necessary
    if I is None:
        I = np.eye(v_dim, dtype=v.dtype)

    b   = np.dot(Ft, v).T
    L   = np.dot(Nt, VtV).reshape(v_dim, v_dim) + I
    out = spl.solve(L, b)

    return out


################################################################################
################################################################################
def estimate_i_and_invL(Nt, Ft, v, VtV=None, I=None, out=None):
    v_dim   = v.shape[1]
    n_gauss = Nt.shape[1]

    # compute VtV if necessary
    if VtV is None:
        VtV = compute_VtV(v, n_gauss)

    # Allocate space for out i-vec if necessary
    if out is None: 
        out = np.empty((1, v_dim), dtype=v.dtype)

    # Construct eye if necessary
    if I is None:
        I = np.eye(v_dim, dtype=v.dtype)

    b   = np.dot(Ft, v).T
    L   = np.dot(Nt, VtV).reshape(v_dim, v_dim) + I
    out = spl.solve(L, b)
    invL = spl.inv(L);

    return out, invL



################################################################################
################################################################################
def compute_VtV(v, n_gauss, out=None):

    v_dim   = v.shape[1]
    f_dim   = int(v.shape[0] / n_gauss)

    # Allocate space if necessary
    if out is None: 
        out = np.empty((n_gauss, v_dim, v_dim), dtype=v.dtype)
    else:
        out = out.reshape((n_gauss, v_dim, v_dim))

    # reshape v to 
    v3d = v.reshape((n_gauss, f_dim, v_dim));

    for i in range(n_gauss):
        v_part = v3d[i,:,:]
        np.dot(v_part.T, v_part, out=out[i,:,:])

    out = out.reshape((n_gauss, v_dim*v_dim))
    return out


################################################################################
################################################################################
def E_step(Nt, Ft, v, VtV=None, I=None, A=None, C=None):
    n_data,n_gauss  = Nt.shape
    v_dim           = v.shape[1]

    # compute VtV if necessary
    if VtV is None:
        VtV = compute_VtV(v, n_gauss)

    # Construct eye if necessary
    if I is None:
        I = np.eye(v_dim, dtype=v.dtype)

    if A is None:
        A = np.zeros((n_gauss,v_dim,v_dim), dtype=np.float64)

    if C is None:
        C = np.zeros(v.shape, dtype=np.float64)

    Ftv = np.dot(Ft, v)

    # accumulate over data
    for i in range(n_data):
        Nti   = row(Nt[i,:])
        Fi    = col(Ft[i,:])
        Ftvi  = row(Ftv[i,:])

        L     = np.dot(Nti, VtV).reshape(v_dim, v_dim) + I
        invL  = spl.inv(L)

        it    = np.dot(Ftvi, invL)
        invL += np.dot(it.T,it)

        # update A over Gaussians
        for c in range(n_gauss):
            A[c,:,:] += invL * Nti[0,c]

        C += np.dot(Fi, it)
    
    return A, C


################################################################################
################################################################################
def E_step_with_MD(Nt, Ft, v, VtV=None, I=None, A=None, C=None, 
                    Amd=None, Cmd=None, Nmd=None):

    n_data,n_gauss  = Nt.shape
    v_dim           = v.shape[1]

    # compute VtV if necessary
    if VtV is None:
        VtV = compute_VtV(v, n_gauss)

    # Construct eye if necessary
    if I is None:
        I = np.eye(v_dim, dtype=v.dtype)

    if A is None:
        A = np.zeros((n_gauss,v_dim,v_dim), dtype=np.float64)

    if C is None:
        C = np.zeros(v.shape, dtype=np.float64)

    if Amd is None:
        Amd = np.zeros((v_dim,v_dim), dtype=np.float64)

    if Cmd is None:
        Cmd = np.zeros((1,v_dim), dtype=np.float64)

    if Nmd is None:
        Nmd = np.zeros((1), dtype=np.float64)

    Ftv = np.dot(Ft, v)

    # accumulate over data
    for i in range(n_data):
        Nti   = row(Nt[i,:])
        Fi    = col(Ft[i,:])
        Ftvi  = row(Ftv[i,:])

        L     = np.dot(Nti, VtV).reshape(v_dim, v_dim) + I
        invL  = spl.inv(L)

        it    = np.dot(Ftvi, invL)
        invL += np.dot(it.T,it)

        # update A over Gaussians
        for c in range(n_gauss):
            A[c,:,:] += invL * Nti[0,c]

        C   += np.dot(Fi, it)
        Amd += invL
        Cmd += it
        Nmd += n_data

    return A, C, Amd, Cmd, Nmd


################################################################################
################################################################################
def M_step(A, C, out=None, dtype=None):
    n_gauss = A.shape[0]
    v_dim   = A.shape[1]
    f_dim   = int(C.shape[0] / n_gauss)
    #print f_dim, v_dim

    if out is None:
        if dtype is None:
            dtype = np.float32
        out = np.empty(C.shape,dtype=dtype)

    elif (dtype is not None) and (dtype != out.dtype):
        raise TypeError("Out matrix is of different type than desired")

    C   = C.reshape((n_gauss,f_dim,v_dim))
    out = out.reshape((n_gauss,f_dim,v_dim))

    for c in range(n_gauss):
        out[c,:,:] = spl.solve(A[c,:,:], C[c,:,:].T).T

    return out


################################################################################
################################################################################
def M_step_MD(Amd, Cmd, Nmd, v):
    Amd /= Nmd
    Cmd /= Nmd
    Amd -= np.dot(Cmd.T, Cmd)
    r    = spl.cholesky(Amd)
    v    = np.dot(v, r)

    return v


################################################################################
################################################################################
def normalize_stats(n, f, ubmMeans, covNorm):
    """ Normalize statistics using UBM params
    """
    numG = ubmMeans.shape[0]
    dimF = ubmMeans.shape[1]

    f0   = f - ubmMeans*np.kron(np.ones((dimF,1),dtype=n.dtype),n).transpose()
    if covNorm.ndim == 2:
      f0=f0*covNorm
    else:
      for ii in range(numG):
        f0[ii,:]=f0[ii,:].dot(covNorm[ii])

    return n,f0


