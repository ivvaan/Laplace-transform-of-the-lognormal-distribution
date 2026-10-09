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
    x_nodes=np.linspace(-rng,rng,n)  #symmetric integration points without 0
    step=x_nodes[1]-x_nodes[0]
    x_i=np.empty_like(x_nodes)
    w_i=np.empty_like(x_nodes)
    with mp.workdps(34):
        dstep, h_pi = mp.mpf(step)/n, mp.mpf(0.5)*mp.pi
        for i,node in enumerate(x_nodes):
            dt_node=mp.mpf(node) 
            sinh_t=h_pi*mp.sinh(dt_node)
            x=mp.sinh(sinh_t)
            dx=h_pi*dstep*mp.cosh(sinh_t)*mp.cosh(dt_node)
            x_i[i],  w_i[i] =x,mp.exp(-x**2/2.)/mp.sqrt(2.*mp.pi)*dx #intergating with std normal weights
       
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

def LTLN_DI_HA_real_uvF(n=37, h_g=True, toll=sys.float_info.epsilon/10.): #higher accuracy for big u 
    '''
    Computes nodes and weights for std normal \exp(-x^2/2)/\sqrt(2\pi) integration
    then construct function to get the Laplace transform of log-normal variable
    '''
    n=(n//2) * 2 + 1
    if h_g and n < 120: #hermite gauss tested only up to 100 nodes and crashed after 180
        si, ws = hermite_gauss_scaled(n,toll)
    else:
        si, ws = takahasi_mori_stdnormal(n)  
    n=len(si)

 
    def LTLN(u,v):
        '''
        Computes the Laplace transform at the point u,
        using precomputed Hermite–Gauss or Takahasi-Mori nodes  and weights .
        
        Arguments:
        - nodes and weights scaled on sqrt(2) and 1/sqrt(pi)
        - u: real-valued base magnitude of the argument
        - v: log-variance parameter of the lognormal distribution
        '''
        u=np.atleast_1d(u)
        a = np.real(scsp.lambertw(u*v))/v
        σ = np.sqrt(v)
        
        def f_domain():
            a_ = a.flatten()[:,None]
            q=a_+(1j/σ)*si[None,:]
            g=np.real(np.exp(-np.log(a_)*q)*scsp.gamma(q))
            scale=0.3989422804014327/σ # 0.3989422804014327 = 1/sqrt(2 pi)
            return np.dot(g, ws*scale ), g[:,n//2]*scale
            
        def s_domain():
            x = si*σ # v-depentent part of the scaling 
            # The exponential term in the integrand, vectorized over all x
            exp_term = np.exp(np.outer(a, x - np.exp(x)))
            # Gauss–Hermite quadrature integration
            return np.dot(exp_term, ws),exp_term[:,n//2]

        fv,fm=f_domain()
        sv,sm=s_domain()
        is_f=1.25*fm<sm
        vals=np.where(is_f,fv,sv)
        return np.exp((-.5*v)*a**2) * vals.reshape(a.shape)  
    return LTLN

r"""
L(u) = E[exp(-u X)],  X ~ LogNormal(0, v)

Гибрид двух гауссово-эрмитовых квадратур с переключением по u:

  u <  u_switch : пространственная ветвь (интеграл по y с ламбертовым сдвигом)
  u >= u_switch : частотная ветвь -- контур Меллина-Барнса, поставленный
                  в СЕДЛОВУЮ точку

ЧАСТОТНАЯ ВЕТВЬ
---------------
    L(u) = (1/2pi) \int Gamma(s) exp(s^2 v/2 - s*zeta) dt,  s = c+it, zeta = ln u

Значение не зависит от c при c > 0 (контур правее полюсов Gamma в 0,-1,-2,...).
Оптимальный контур -- через седло подынтегральной функции:

    psi(c) + c v = zeta = ln u          (=  ln a + v a,  a = W(uv)/v )

Оба вида записи тождественны: из a = W(uv)/v следует a*exp(va) = u, то есть
ln a + va = ln u. Так как psi(c) -> -inf при c -> 0+ и монотонно растёт,
корень существует, единствен и всегда положителен -- вычеты не нужны.

Отличие от замкнутой седловой формулы (LTLN_approx_SP): там показатель
раскладывается до второго порядка и гауссов интеграл берётся аналитически,
что даёт 4-9% ошибки, поскольку подынтегральная функция не гауссова. Здесь
седло используется ТОЛЬКО для постановки контура, а интеграл берётся
численно -- ошибка падает до уровня квадратуры.

Две детали, обе существенные:
  * контур ставится в c из седлового уравнения, а НЕ в c = a; при малых a
    разница велика (при a = 1e-3 седловое c ~ 0.15, то есть в 150 раз дальше);
  * ширина квадратуры берётся из полной кривизны b2 = (psi'(c) + v)/2, а не
    из одного лишь v: вклад Gamma заметен при умеренных u и исчезает при
    больших, где седло уходит вправо и psi'(c) -> 0.

ПОЧЕМУ ПОРОГ ПО u, А НЕ СИГМОИДА
--------------------------------
При u -> 0 седло уходит к нулю, psi'(c) ~ 1/c^2, ширина сжимается до ~c, и
гауссова квадратура теряет хвосты Gamma, спадающие как exp(-pi|t|/2). Поэтому
частотная ветвь там непригодна, а пространственная, наоборот, даёт машинную
точность. Ветви перекрываются с огромным запасом, так что жёсткий порог не
создаёт заметного излома: при u_switch = 2 обе стороны дают 1e-12..1e-14.
Это важно для обращения Лапласа -- метод Стехфеста при m = 9 имеет
знакопеременные веса до 1e15 и работает только если ошибка L(u) ГЛАДКАЯ по u;
плавное сигмоидное смешивание двух схем с разной структурой ошибки как раз
негладкость и создаёт, раздувая её на порядки.

ТОЧНОСТЬ (эталон -- mpmath, dps = 50; v = 1.5, 3, 4.5)
относительная ошибка не хуже ~3e-14 на всём диапазоне u от 1e-10 до 1e20,
на большей части -- уровень 1e-15..1e-16.
"""


def _hermite_gauss(n, toll=1e-18):
    xi, wi = np.polynomial.hermite.hermgauss(n)
    tol = toll*np.max(wi)
    k = np.argmax(wi[:n//2] > tol)
    if k != 0:
        xi, wi = xi[k:-k], wi[k:-k]
    return xi, wi


def _saddle_c(u, v, a, n_iter=6):
    """корень psi(c) + c v = ln u, ньютоном от хорошей затравки"""
    tgt = np.log(u)
    c = a + 1.0/(7.5 + 25.0*(a - 0.85*np.sqrt(a)))
    for _ in range(n_iter):
        c = c - (scsp.psi(c) + v*c - tgt)/(scsp.polygamma(1, c) + v)
    return c


def LTLN_DI_CL_real_uvF(n_space=141, n_freq=91, u_switch=10.):
    """Возвращает f(u, v) -> L(u; 0, v). u скаляр или массив.

    n_space  -- узлов Гаусса-Эрмита в пространственной ветви
    n_freq   -- узлов в частотной ветви
    """
    xs, ws = hermite_gauss_scaled(n_space, 1e-18/n_space)
    xf, wf = _hermite_gauss(n_freq)

    def L(u, v):
        u = np.atleast_1d(np.asarray(u, dtype=float))
        sh = u.shape
        uf_ = u.ravel()
        out = np.ones(uf_.size)

        pos = uf_ > 0
        if not np.any(pos):
            return out.reshape(sh)
        up = uf_[pos]
        a = np.real(scsp.lambertw(up*v))/v
        res = np.empty(up.size)

        lo = up < u_switch
        if np.any(lo):                       # пространственная ветвь
            x = xs*np.sqrt(v)
            al = a[lo]
            res[lo] = np.exp(-0.5*v*al**2)*np.dot(
                np.exp(np.outer(al, x) - np.outer(al, np.exp(x))), ws)

        hi = ~lo
        if np.any(hi):                       # частотная ветвь, контур в седле
            uh = up[hi] 
            c = _saddle_c(uh, v, a[hi])
            b2 = (scsp.polygamma(1, c) + v)/2.0
            t = xf[None, :]/np.sqrt(b2)[:, None]
            s = c[:, None] + 1j*t
            lg = (scsp.loggamma(s) + (s**2)*v/2.0
                  - s*np.log(uh)[:, None] + b2[:, None]*t**2)
            m = lg.real.max(axis=1, keepdims=True)
            res[hi] = (np.real(np.exp(lg - m) @ wf)
                       / (2*np.pi*np.sqrt(b2))*np.exp(m.ravel()))

        out[pos] = res
        return out.reshape(sh)
    return L

