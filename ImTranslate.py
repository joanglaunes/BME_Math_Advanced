def ImTranslate(h,tau):
    M,N = h.shape
    ind0 = np.concatenate((range(tau[0],M),range(tau[0])))
    ind1 = np.concatenate((range(tau[1],N),range(tau[1])))
    h_tau = h[ind0,:][:,ind1]
    return h_tau
