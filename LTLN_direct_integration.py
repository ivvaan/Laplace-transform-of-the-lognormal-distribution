def hermite_gauss(n,toll=1.e-19):
    ti, wi = np.polynomial.hermite.hermgauss(n)
    toll = toll * np.max(wi)
    k=np.argmax(wi[:n//2]>toll)
    if k!=0:
        ti, wi =ti[k:-k], wi[k:-k]
    return ti,wi
    
def laplace_by_integration_base(ti,wi,u,v,phi = 0):
    sigma2=np.sqrt(2.*v)
    c=np.real(scsp.lambertw(np.atleast_1d(u)*v)) + phi #-np.real(scsp.lambertw(u*v))/sigma2 - shift giving min exp_term at ti=0
    exp_term = np.outer(np.real(c),np.exp(sigma2  * ti)/v) - np.outer(c,2./sigma2*ti)
    # Расчет по формуле Gauss-Hermite
    vals = np.dot(np.exp(-exp_term),wi) 
    L = np.exp(-c**2/(2.*v))*vals.reshape(c.shape)/np.sqrt(np.pi)
    return L

def LTLN_DI_real_uvF(n=35,toll=sys.float_info.epsilon/10.):
    ti, wi = hermite_gauss(n,toll/n)
    def laplace_fn(u,v):
        return laplace_by_integration_base(ti,wi,u,v)
    return laplace_fn


def LTLN_DI_real_uF(v, n=35,toll=sys.float_info.epsilon/10.):
    ti, wi = hermite_gauss(n,toll/n)
    def laplace_fn(u):
        return laplace_by_integration_base(ti,wi,u,v)
    return laplace_fn

def LTLN_DI_imag_uvF(n=35,toll=sys.float_info.epsilon/10.):
    ti, wi = hermite_gauss(n,toll/n)
    def laplace_fn(u,v):
        return laplace_by_integration_base(ti,wi,u,v,1j*np.pi/2)
    return laplace_fn

def LTLN_DI_imag_uF(v, n=35,toll=sys.float_info.epsilon/10.):
    ti, wi = hermite_gauss(n,toll/n)
    def laplace_fn(u):
        return laplace_by_integration_base(ti,wi,u,v,1j*np.pi/2)
    return laplace_fn

def LTLN_DI_neg_uvF(n=35,toll=sys.float_info.epsilon/10.):
    ti, wi = hermite_gauss(n,toll/n)
    def laplace_fn(u,v):
        return laplace_by_integration_base(ti,wi,u,v,1j*np.pi)
    return laplace_fn

def LTLN_DI_neg_uF(v, n=35,toll=sys.float_info.epsilon/10.):
    ti, wi = hermite_gauss(n,toll/n)
    def laplace_fn(u):
        return laplace_by_integration_base(ti,wi,u,v,1j*np.pi)
    return laplace_fn

def LTLN_DI_complex_uvF(n=35,toll=sys.float_info.epsilon/10.):
    ti, wi = hermite_gauss(n,toll/n)
    def laplace_fn(u,v):
        return laplace_by_integration_base(ti,wi,np.abs(u),v,1j*np.angle(u))
    return laplace_fn

def LTLN_DI_complex_uF(v, n=35,toll=sys.float_info.epsilon/10.):
    ti, wi = hermite_gauss(n,toll/n)
    def laplace_fn(u):
        return laplace_by_integration_base(ti,wi,np.abs(u),v,1j*np.angle(u))
    return laplace_fn
    
