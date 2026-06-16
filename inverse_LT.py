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
    nodes=(1+np.arange(2*m)) 
    def get_pdf(LTF,T,v):
        res=np.empty_like(T)
        for i,t in enumerate(T):
            step=np.log(2)/t
            res[i]=step*np.dot(wk,LTF(step*nodes,v))
        return res    
    return get_pdf

def get_inverse_pdf_uF(m=9):
    wk = stehfest_coeffs_pdf(m)
    nodes=(1+np.arange(2*m)) 
    def get_pdf(LTF,T,v):
        res=np.empty_like(T)
        for i,t in enumerate(T):
            step=np.log(2)/t
            res[i]=step*np.dot(wk,LTF(step*nodes))
        return res    
    return get_pdf

def get_inverse_cdf_uvF(m=9,nw=True):
    wk=stehfest_coeffs_cdf(m,nw=nw)
    nodes=(1+np.arange(2*m)) 
    def get_cdf(LTF,T,v):
        res=np.empty_like(T)
        for i,t in enumerate(T):
            step=np.log(2)/t
            res[i]=np.dot(wk,LTF(step*nodes,v))
        return res  
    return get_cdf

def get_inverse_cdf_uF(m=9,nw=True):
    wk=stehfest_coeffs_cdf(m,nw=nw)
    nodes=(1+np.arange(2*m)) 
    def get_cdf(LTF,T):
        res=np.empty_like(T)
        for i,t in enumerate(T):
            step=np.log(2)/t
            res[i]=np.dot(wk,LTF(step*nodes))
        return res  
    return get_cdf
