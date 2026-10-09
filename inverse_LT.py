def stehfest_coeff_w_k(m,k, dps=50): #actual Stehfest coefficients for the inverse Laplace transform to get original function (aka pdf)
    with mp.workdps(dps):
        sum = mp.mpf(0)
        for j in range(math.floor((k+1)/2),np.min([m,k])+1):
            sum = sum + mp.power(j,m+1)*mp.binomial(m,j)*mp.binomial(2*j,j)*mp.binomial(j,k-j)/mp.factorial(m)
        return np.float64(np.power(-1,m+k)*sum)

def stehfest_coeffs_pdf(m,dps=50):
    wk = np.empty(2*m)
    for i in range(2*m):
        wk[i] = stehfest_coeff_w_k(m,i+1,dps=dps)
    return wk

def stehfest_coeff_v_k(m,k,dps=50):  #"Stehfest coefficients" for the inverse Laplace transform to get an integral of the original function (aka cdf)) 
    with mp.workdps(dps):
        sum = mp.mpf(0)
        for j in range(math.floor((k+1)/2),np.min([m,k])+1):
            sum = sum + mp.power(j,m+1)*mp.binomial(m,j)*mp.binomial(2*j,j)*mp.binomial(j,k-j)/mp.factorial(m)
        return np.float64(np.power(-1,m+k)*sum/k)

def stehfest_coeffs_cdf(m,nw=True,dps=50):
    wk = np.empty(2*m)
    for i in range(2*m):
        wk[i] = stehfest_coeff_v_k(m,i+1,dps=dps)
    if nw: wk /= np.sum(wk)
    return wk

def get_inverse_pdf_uvF(m=9):
    wk = stehfest_coeffs_pdf(m)
    nodes = (1 + np.arange(2*m))
    def get_pdf(LTF, T, v):
        T = np.asarray(T)
        if T.size == 0:
            return np.empty_like(T)
        nT = T.size
        steps = np.log(2) / T  # shape (nT,)
        S = np.outer(steps, nodes)  # shape (nT, 2*m)
        inp = S.ravel()  # 1D array length nT*2*m
        out = np.asarray(LTF(inp, v))
        if out.size != nT * (2*m):
            raise ValueError(f"LTF returned array of size {out.size}, expected {nT*(2*m)}")
        out = out.reshape(nT, 2*m)
        vals = out.dot(wk)  # shape (nT,)
        res = steps * vals
        return res
    return get_pdf

def get_inverse_pdf_uF(m=9):
    wk = stehfest_coeffs_pdf(m)
    nodes = (1 + np.arange(2*m))
    def get_pdf(LTF, T):
        T = np.asarray(T)
        if T.size == 0:
            return np.empty_like(T)
        nT = T.size
        steps = np.log(2) / T
        S = np.outer(steps, nodes)
        inp = S.ravel()
        out = np.asarray(LTF(inp))
        if out.size != nT * (2*m):
            raise ValueError(f"LTF returned array of size {out.size}, expected {nT*(2*m)}")
        out = out.reshape(nT, 2*m)
        vals = out.dot(wk)
        res = steps * vals
        return res
    return get_pdf

def get_inverse_cdf_uvF(m=9,nw=True):
    wk = stehfest_coeffs_cdf(m,nw=nw)
    nodes = (1 + np.arange(2*m))
    def get_cdf(LTF, T, v):
        T = np.asarray(T)
        if T.size == 0:
            return np.empty_like(T)
        nT = T.size
        steps = np.log(2) / T
        S = np.outer(steps, nodes)
        inp = S.ravel()
        out = np.asarray(LTF(inp, v))
        if out.size != nT * (2*m):
            raise ValueError(f"LTF returned array of size {out.size}, expected {nT*(2*m)}")
        out = out.reshape(nT, 2*m)
        res = out.dot(wk)
        return res
    return get_cdf

def get_inverse_cdf_uF(m=9,nw=True):
    wk = stehfest_coeffs_cdf(m,nw=nw)
    nodes = (1 + np.arange(2*m))
    def get_cdf(LTF, T):
        T = np.asarray(T)
        if T.size == 0:
            return np.empty_like(T)
        nT = T.size
        steps = np.log(2) / T
        S = np.outer(steps, nodes)
        inp = S.ravel()
        out = np.asarray(LTF(inp))
        if out.size != nT * (2*m):
            raise ValueError(f"LTF returned array of size {out.size}, expected {nT*(2*m)}")
        out = out.reshape(nT, 2*m)
        res = out.dot(wk)
        return res
    return get_cdf