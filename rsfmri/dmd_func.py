import numpy as np

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



#############################################################################