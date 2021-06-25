###################### ACT-R + PYTHON TEMPLATE #######################
#   Author: Cher Yang
#   Date: 6.1.2021
# This script provides dmd analysis functions
#
# Bugs: 
#
# TODO: Calcualte Modes
# 
# Requirement: 
#
#
#
###################### ####################### #######################

import os, glob
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
from sklearn.preprocessing import StandardScaler, normalize

import lasso_func




class DMD(object):

    """
    Class to compute dynamic mode decomposition of dynamical system.
    """

    def __init__(self, n_modes=None, es=None, tr=0.720):

        """
        Initialization method.

        Parameters:
        - - - - -
        n_modes: int
            number of DMD modes to keep
        es: float
            fraction of energy of system to keep
        tr: float
            sampling frequency (repetition time)
            default set to TR of HCP data
        """

        self.n_modes = n_modes
        self.es = es
        self.tr = tr

    def fit(self, X):

        """
        Fit the DMD model.

        Parameters:
        - - - - -
        X: float, array
            matrix of time series / samples
        """

        n = self.n_modes
        p = self.es
        rT = self.tr

        X -= np.mean(X, axis=1)[:, None]
        C = X[:, :-1]
        Cp = X[:, 1:]

        [U, S, V] = np.linalg.svd(C, full_matrices=False)

        # If power is defined, but number of modes is not
        # estimate the number of modes form data
        if not n and p:
            n = np.where((np.cumsum(S)/np.sum(S)) >= p)[0][0]+1
            print('Keeping {:} modes to capture {:} of energy.'.format(n, p))

        # Otherwise, number of modes = number of samples
        if n == None:
            n = X.shape[0]
        
        Ut = U[:, :n]
        Sinv = np.diag(1./S[:n])
        Vt = V[:n].T

        # compute reduced-dimensional representation of A-matrix
        Ap = (Ut.T).dot(Cp.dot(Vt.dot(Sinv)))

        # weight Ap by singular values so that modes reflect explained variance
        Ah = np.diag(S[:n]**-0.5).dot(Ap.dot(np.diag(S[:n]**0.5)))

        # compute eigendecomposition of weighted A matrix
        [w, v] = np.linalg.eig(Ah)
        v = np.diag(S[:n]**0.5).dot(v)

        # compute DMD modes from eigenvectors
        # using this approach, modes are not normalized -- norm gives power
        # of mode in data
        Phi = Cp.dot(Vt.dot(Sinv.dot(v)))
        power = np.real(np.sum(Phi*Phi.conj(), 0))

        # using h to convert complex eigenvalues into corresponding
        # oscillation frequencies
        freq = np.angle(w)/(2*np.pi*rT)

        self.phi_ = Phi
        self.power_ = power
        self.freq_ = freq

"""
Methods to compute group-DMD modes and projections of single-subject time series onto these modes.

#####
#####
Example usage with usage for HCP data, with four resting-state acquisitions for single subject.
Compute group-DMD modes by aggregating all four resting-state runs.


# compute group modes
ts = [r1, r2, r3, r4]                                   # r{1,2,3,4} (N x 1200)
[X, Y] = concat_timeseries(T=ts)                        # X, Y are (N x 4799)
[modes, power, frequency] = group_DMD(X, Y, tr=0.72)    # run DMD


# project single time-series onto group-modes
C = linear_operator(r1, modes)                          # compute C linear operator
weights = betas(r1, C)                                  # compute linear weights


# compute eigenvalues
D_mat = eigenvalues(C, weights)                         # D is a diagonal matrix of eigenvalues


"""

def group_DMD(C, Cp, tr=0.72, n=None, p=None):
    
    """
    Compute group-DMD modes and eigenvalues.
    
    DMD formulation is that described by:
        
        Schmid et al (https://hal-polytechnique.archives-ouvertes.fr/hal-01020654/document)
        
        Brunton et al (https://arxiv.org/abs/1409.5496)
    
    Parameters:
    - - - - -
    C: float, array
        time-series matrix of from t=0 to t=(T-1)
    Cp: float, array
        time-series matrix of from t=1 to t=T
    
    tr: float
        sampling frequency (timestep between measurements, for calculating frequencies in Hz)
        if tr is unset, uses default of tr=(14*60+33)/1200 for HCP data
    n: int
        number of modes to compute
    p: float
        fraction of energy of system to keep
        
    Returns:
    - - - -
    Phi: float, array
        DMD modes
    power: float, array
        power of each mode
    freq: float, array
        eigenvalues of each mode
    """
    
    [U, S, V] = np.linalg.svd(C, full_matrices=False)

    # If power is defined, but number of modes is not
    # estimate the number of modes form data
    if not n and p:
        n = np.where((np.cumsum(S)/np.sum(S)) >= p)[0][0]+1
        print('Keeping {:} modes to capture {:} of energy.'.format(n, p))

    # Otherwise, number of modes = number of samples
    if n == None:
        n = C.shape[0]

    Ut = U[:, :n]
    Sinv = np.diag(1./S[:n])
    Vt = V[:n].T

    # compute reduced-dimensional representation of A-matrix
    Ap = (Ut.T).dot(Cp.dot(Vt.dot(Sinv)))

    # weight Ap by singular values so that modes reflect explained variance
    Ah = np.diag(S[:n]**-0.5).dot(Ap.dot(np.diag(S[:n]**0.5)))

    # compute eigendecomposition of weighted A matrix
    [w, v] = np.linalg.eig(Ah)
    v = np.diag(S[:n]**0.5).dot(v)

    # compute DMD modes from eigenvectors
    # using this approach, modes are not normalized -- norm gives power
    # of mode in data
    Phi = Cp.dot(Vt.dot(Sinv.dot(v)))
    power = np.real(np.sum(Phi*Phi.conj(), 0))

    # using h to convert complex eigenvalues into corresponding
    # oscillation frequencies
    freq = np.angle(w)/(2*np.pi*tr)
    
    return [Phi, power, freq]


def linear_operator(X, modes):
    
    """
    Compute C matrix for projecting single-subject time-series onto DMD modes.
    See Casorso et al. for more details (https://www.sciencedirect.com/science/article/pii/S1053811919301922)
    
    Parameters:
    - - - - -
    X: float, array, (N x T)
        single-subject time series matrix
    S: float, array, (N x K)
        group DMD modes
    
    Returns:
    - - - -
    C: float, array, (K x K)
        matrix of partial derivative solutions to linear operator
        C-matrix from Carsorso et al.
    """
    
    # get shape of data and number of DMD modes
    [n, p] = modes.shape

    # initialize U and V matrices
    u = modes; v = modes.T
    C = np.zeros((p, p))

    # compute squared-norm of each DMD mode
    unorm = u.T.dot(u) # (K x K)
    # 2-norm of each DMD mode
    norms = np.sqrt(np.diag(unorm))
    
    # solve for diagonal of linear operator
    xv = X.T.dot(v.T) # (T x K)
    xv_weighted = xv*norms
    
    D = xv_weighted.T.dot(xv_weighted) # (K x K)
    C = np.diag(2*np.diag(D), k = 0)
    
    # solve for off-diagonal elements of linear operator
    # off diagonal elements are sums of transposable elements
    # so C is symmetric
    for k in np.arange(p):
        for j in np.arange(p):

            if k != j:

                unorm1 = u[:,j].T.dot(u[:, k])
                unorm2 = u[:, k].T.dot(u[:, j])

                xvk = X.T.dot(v.T[:, k])[:, None]
                xvj = X.T.dot(v.T[:, j])[:, None]

                C[k, j] = np.diag(unorm1*xvk.dot(xvj.T)).sum() + np.diag(unorm2*xvj.dot(xvk.T)).sum()
            
            #z = 2*unorm[k, j] * xv[:, k].dot(xv[:, j])
            # C[j, k] = z
            # C[k, j] = z

    return C


def betas(X, modes):
    
    """
    Compute linear weights.
    See Casorso et al. for more details (https://www.sciencedirect.com/science/article/pii/S1053811919301922)
    
    Parameters:
    - - - - -
    X: float, array, (N x T)
        single-subject time series matrix
    S: float, array, (N x K)
        group DMD modes
        
    Returns:
    - - - -
    weights: float, array
        linear weights for operator matrix
        beta-matrix from Casorso et al.
    """
    
    # get shape of data and number of DMD modes
    [n, p] = modes.shape
    weights = np.zeros((p,))
    
    # initialize U and V matrices
    u = modes
    v = modes.T
    # compute squared-norm of each DMD mode
    xu = X[:, 1:].T.dot(u)
    xv = v.dot(X[:, :-1])

    for k in np.arange(p):
        
        temp_xu = xu[:, k][:, None]
        temp_xv = xv[k, :][None, :]
        
        temp = temp_xu.dot(temp_xv)
        weights[k] = np.diag(temp).sum()

    return weights


def eigenvalues(operator, weights):
    
    """
    Compute the subject-specific eigenvalues.
    See Casorso et al. for more details (https://www.sciencedirect.com/science/article/pii/S1053811919301922)
    
    Parameters:
    - - - - -
    operator: float, array
        matrix of partial derivative solutions to linear operator
        C-matrix from Casorso et al.
    weights: float, array
        linear weights for operator matrix
        beta-matrix from Casorso et al.
        
    Returns:
    - - - -
    D: float, array
        diagonal matrix of subject-specific eigenvalues, informed by group-DMD modes
        D-matrix in Casorso et al.
    """
    
    D = np.linalg.inv(operator).dot(weights)
    
    return D

    def concat_timeseries(T=[]):
        
        """
        Generate concatenated time-series matrices for group-DMD.
        """

        X = construct_x(T=T)
        Y = construct_y(T=T)

        return [X, Y]


##########

# Methods for concatenating time series within and across subjects

##########

def concat_timeseries(T=[]):
    """
    Generate concatenated time-series matrices for group-DMD.
    """

    X = construct_x(T=T)
    Y = construct_y(T=T)

    return [X, Y]


def construct_x(T=[]):
    """
    Generate X-matrix of time series for group-DMD.
    """

    times = []
    for t in T:
        t -= np.mean(t, axis=1)[:, None]
        times.append(t[:, :-1])
    times = np.column_stack(times)

    return times


def construct_y(T=[]):
    """
    Generate Y-matrix of time series for group-DMD.
    """

    times = []
    for t in T:
        t -= np.mean(t, axis=1)[:, None]
        times.append(t[:, 1:])
    times = np.column_stack(times)

    return times

###################### COMPUTE DMD #######################
def save_all_dmd(connctivitymatrix_dir='./connectivity_matrix/REST1', dmd_dir='./dmd_results/REST1/ses-01'):
    """
    """
    # create a dmd results dir
    try: os.mkdir(dmd_dir)
    except: pass

    # ieterate through all subjects
    subj_dirs=glob.glob(connctivitymatrix_dir+'/sub-*')
    subj_dirs.sort()
    for subj_dir in subj_dirs:
        subj_id=int(subj_dir.split('-')[-1])

        # check if already processed
        if os.path.exists(dmd_dir+'/{:}.h5'.format('sub-'+str(subj_id))):
            print('{:} Already processed! Moving on...'.format(subj_id))
            continue

        # load subject's time series data
        subj_timeseries=pd.read_csv('{:}/ses-01/raw_timeseries.txt'.format(subj_dir)).T.values
        
        # compute single subject's dmd
        subj_dmd=compute_subj_dmd(subj_timeseries, subj_id)
        save_dict_to_hdf5(subj_dmd, dmd_dir+'/{:}.h5'.format('sub-'+str(subj_id)))
    return

def compute_subj_dmd(subj_timeseries, HCPID, nwindow=32, nslide=4, nmodes=8):
    frame0s=np.arange(0,1200,nslide)
    frame0s=frame0s[(frame0s+nwindow)<=1200]

    Phi=[]
    FT=[]
    PT=[]
    ux=[]
    jx=[]
    jno=[]
    for j, frame0, in enumerate(frame0s):
        curr_X=subj_timeseries[:,frame0:(frame0+nwindow)]

        # perform DMD on window of data, calculated 'nmodes' modes
        curr_dmd=DMD(n_modes=nmodes)
        curr_dmd.fit(curr_X)

        # save data
        # only keep positive frequencies (drop conjugates)
        phik=curr_dmd.phi_[:,curr_dmd.freq_>=0]
        ft=curr_dmd.freq_[curr_dmd.freq_>=0]
        pt=curr_dmd.power_[curr_dmd.freq_>=0]
        
        Phi.append(phik)
        FT.append(ft)
        PT.append(pt)

        #record:
        #subject number (ux),
        #window number (jx),
        #mode number within window (jno)
        ux.append(HCPID*np.ones((len(ft),)))
        jx.append(j*np.ones((len(ft),)))
        jno.append(np.arange(len(ft)))
        
    Phi=np.absolute(np.hstack(Phi))
    Phase=np.angle(np.hstack(Phi))      
    #stack frequencies, powers, etc.
    freq=np.hstack(FT)                  #frequency
    power=np.hstack(PT)                 # power
    ux=np.hstack(ux)                    # subject number
    jx=np.hstack(jx)                    # window number
    jno=np.hstack(jno)                  # mode number

    # dict res
    res = {
        'Phi':Phi,
        'Phase':Phase,
        'freq':freq,
        'power':power,
        'ux':ux,
        'jx':jx,
        'jno':jno,
        #'nmodes':str(nmodes),
        #'nframes':str(nwindow)
    }
    return res

def load_all_dmd(dmd_dir='./dmd_results/REST1/ses-01', mode_format='list'):
    """
    """
    dmd_subjs=glob.glob(dmd_dir+'/sub-*.h5')
    dmd_subjs.sort()
    dmd_df=[]
    for dmd_subj in dmd_subjs:
        dmd_dict = load_dict_from_hdf5(dmd_subj)
        del dmd_dict['Phase']
        
        # a list of 264 numbers represents one mode
        if mode_format=='list':
            dmd_dict['Phi']=dmd_dict['Phi'].T.tolist()
            curr_df=pd.DataFrame(dmd_dict)
            curr_df.columns=['modes', 'freq', 'mode_index', 'window_index', 'power', 'HCPID']
            dmd_df.append(curr_df)
        # each column represents one mode for one ROI
        else:
            mode_df=pd.DataFrame(dmd_dict['Phi'].T, columns=range(1,265))
            del dmd_dict['Phi']
            rest_df=pd.DataFrame(dmd_dict)
            rest_df.columns=['freq', 'mode', 'window', 'power', 'HCPID']
            dmd_df.append(pd.concat([rest_df, mode_df], axis=1))
    return pd.concat(dmd_df, axis=0)

def compute_dmd_corr(dmd_dir='./dmd_results/REST1/ses-01'):
    dmd_subjs=glob.glob(dmd_dir+'/sub-*.h5') # 178
    dmd_subjs.sort()

    dmd_corr_list=[]
    for sub in dmd_subjs:
        subj_dmd = load_dict_from_hdf5(sub)
        subj_mode = subj_dmd['Phi'] #264 * 1573
        subj_mode_scaled = StandardScaler().fit_transform(subj_mode) 
        subj_mode_normalized = pd.DataFrame(normalize(subj_mode_scaled))
        subj_mode_corr_mat = subj_mode_normalized.T.corr()
        subj_mode_corr_vec = lasso_func.matrix2vector(subj_mode_corr_mat)
        dmd_corr_list.append(subj_mode_corr_vec)
    dmd_corr_df = pd.DataFrame(dmd_corr_list, columns=lasso_func.vectorize_mat_name(range(1, 265), range(1,265)))
    dmd_corr_df['HCPID'] = [sub.split('-')[-1].split('.')[0]+'_fnca' for sub in dmd_subjs]
    # dmd_corr_df.to_csv('./bin/REST1_ses-01_g_dmd_corr.csv', index=False)
    return dmd_corr_df



###################### SAVE DATA #######################

def save_dict_to_hdf5(dic, filename):
    """
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes, int, list)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            #ans[key] = item.value
            ans[key] = item[()]
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

