def hermite_gauss_scaled(n=36, toll=1.e-20): #scaled for exp(-x^2/2) and sum(w_i)=1
    xi, wi = np.polynomial.hermite.hermgauss(n)
    # Terms with very small weights will be lost due to numerical rounding errors,
    # so we exclude them to improve numerical stability.
    toll = toll * np.max(wi)
    k = np.argmax(wi[:n//2] > toll)
    if k != 0:
        xi, wi = xi[k:-k], wi[k:-k]
    # 1.4142135623730951 = sqrt(2)
    # 0.56418958354775628 = 1/sqrt(pi)
    return xi*1.4142135623730951, wi*0.56418958354775628 
    
def takahasi_mori_stdnormal(n=120,rng=1.4): #1.4 - is big beautiful constant (w_i too small after that)
    n=(n+1)//2
    step=rng/n 
    x_nodes=np.arange(0.5-n,n,1.)*step #symmetric integration points without 0
    x_i=np.empty_like(x_nodes)
    w_i=np.empty_like(x_nodes)
    with mp.workdps(34):
        dstep, h_pi = mp.mpf(step), mp.mpf(0.5)*mp.pi
        for i,node in enumerate(x_nodes):
            dt_node=mp.mpf(node) 
            sinh_t=h_pi*mp.sinh(dt_node)
            x=mp.sinh(sinh_t)
            dx=h_pi*dstep*mp.cosh(sinh_t)*mp.cosh(dt_node)
            x_i[i],  w_i[i] =x,mp.exp(-x**2/2.)/mp.sqrt(2.*mp.pi)*dx #intergating with std normal weights
       
    return x_i,w_i

def takahasi_mori_stdnormal_lh(n=60,rng=1.4): #1.4 - is big beautiful constant (w_i too small after that)
    step=rng/n 
    x_nodes=-np.arange(0.5,n,1.)*step #symmetric integration points without 0
    x_i=np.empty_like(x_nodes)
    w_i=np.empty_like(x_nodes)
    with mp.workdps(34):
        dstep, h_pi = mp.mpf(step), mp.mpf(0.5)*mp.pi
        for i,node in enumerate(x_nodes):
            dt_node=mp.mpf(node) 
            sinh_t=h_pi*mp.sinh(dt_node)
            x=mp.sinh(sinh_t)
            dx=h_pi*dstep*mp.cosh(sinh_t)*mp.cosh(dt_node)
            x_i[i],  w_i[i] =x,mp.mpf(2)*mp.exp(-x**2/2.)/mp.sqrt(2.*mp.pi)*dx #intergating with std normal weights
    
    return x_i,w_i  

def scale_mul(x,w,v):
    s=1./np.sqrt(v)
    return x*s,w*s
    
def scale_div(x,w,v):
    s=np.sqrt(v)
    return x*s,w*s

def LTLN_DI_real_uvF(n = 36, h_g=True, toll=sys.float_info.epsilon/10. ):
    '''
    Computes the Laplace transform at the point u * exp(i * phi),
    using precomputed Hermite–Gauss nodes (xi) and weights (wi).
    
    Arguments:
    - xi, wi: Hermite–Gauss nodes and weights for std normal (scaled on sqrt(2) and 1/sqrt(pi)) or Takahasi-Mori weights for std normal
    - u: real-valued base magnitude of the argument
    - v: log-variance parameter of the lognormal distribution
    - i_phi: phase of the complex argument (defaults to 0 for real arguments)
    '''    
    if h_g and n < 110:
        xi, wi = hermite_gauss_scaled(n,toll)
    else:
        xi, wi = takahasi_mori_stdnormal(n)
    def LT(u, v):
        x = xi*np.sqrt(v) # v-depentent part of the scaling 
        a = np.atleast_1d(np.real(scsp.lambertw(u * v))/v)
        exp_term = np.outer(a, x - np.exp(x))
        vals = np.dot(np.exp(exp_term), wi)
        return np.exp((-.5*v)*a**2) * vals.reshape(a.shape)
    return LT

def LTLN_DI_HA_real_uvF(n=36, h_g=True, toll=sys.float_info.epsilon/10. ): #higher accuracy for big u 
    '''
    Computes nodes and weights for std normal \exp(-x^2/2)/\sqrt(2\pi) integration
    then construct function to get the Laplace transform of log-normal variable
    '''
    if h_g and n < 110: #hermite gauss tested only up to 100 nodes and crashed after 180
        si, ws = hermite_gauss_scaled(n,toll)
    else:
        si, ws = takahasi_mori_stdnormal(n)    
    half_n=len(si)//2
    q_n=half_n//2
    fi, wf=si[:half_n], 2.*ws[:half_n] #half of nodes we must double wieghts

    def LTLN(u,v):
        '''
        Computes the Laplace transform at the point u,
        using precomputed Hermite–Gauss or Takahasi-Mori nodes  and weights .
        
        Arguments:
        - nodes and weights scaled on sqrt(2) and 1/sqrt(pi)
        - u: real-valued base magnitude of the argument
        - v: log-variance parameter of the lognormal distribution
        '''
        a = np.atleast_1d(np.real(scsp.lambertw(u*v))/v)
        σ = np.sqrt(v)
        
        def f_domain():
            a_ = a.flatten()[:,None]
            q=a_+(1j/σ)*fi[None,:]
            g=np.real(np.exp(-np.log(a_)*q)*scsp.gamma(q))
            scale=0.3989422804014327/σ # 0.3989422804014327 = 1/sqrt(2 pi)
            return np.dot(g, wf*scale ), np.abs(g[:,q_n])*scale
            
        def s_domain():
            x = si*σ # v-depentent part of the scaling 
            # The exponential term in the integrand, vectorized over all x
            exp_term = np.exp(np.outer(a, x - np.exp(x)))
            # Gauss–Hermite quadrature integration
            return np.dot(exp_term, ws),exp_term[:,q_n]
            
        f_res,f_test=f_domain()
        s_res,s_test=s_domain()
        sharpness, eps=10., 1e-20
        z = (f_test - s_test) /np.maximum(f_test + s_test, eps)
        sigmoid = 1. / (1. + np.exp( -sharpness * z))
        vals = sigmoid * f_res + (1. - sigmoid) * s_res #to make domain change smooth
        #vals = np.where(f_test>s_test,f_res,s_res)
        return np.exp((-.5*v)*a**2) * vals.reshape(a.shape)  
    return LTLN


