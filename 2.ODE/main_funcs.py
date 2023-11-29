#basic functions 
def LaplTrLNIntegrFuncFactory(step=5.5/512.0,n_steps=300):
    bound=(n_steps-1.0)/2.0
    t_nodes=np.arange(-bound*step,(bound+0.5)*step,step)[:,None]#symmetric integration points without 0
    sinh_t=0.5*np.pi*np.sinh(t_nodes)
    sinhsinh_t=np.sinh(sinh_t)
    sinhsinh_t2=sinhsinh_t**2
    dt_mult=0.5*np.pi*(step/np.sqrt(np.pi))*np.cosh(sinh_t)*np.cosh(t_nodes)
    max_exp=int(np.finfo(np.float64).maxexp*np.log(2.))            

    def LaplTrLN(u,v):#direct integration
        '''
        fast version of direct integration
        '''
        s=np.exp(np.sqrt(2.0*v)*sinhsinh_t)*np.atleast_1d(u)[None,:]+sinhsinh_t2
        res=np.sum(np.exp(-s)*dt_mult,axis=0)
        return res
    return LaplTrLN

LaplTrLNFastD=LaplTrLNIntegrFuncFactory()

def LaplTrLNODE2(u,v0):
    V0=1.0/v0
    def ode(V,t):return V0*np.exp(0.5/V)/V
    y=odeint(ode,V0,u).T
    return np.exp(V0-y[0])

sp.init_printing()
μ,ξ,w,d,k,t,x,y,p,m,v,q,r,s,z = sp.symbols('μ,ξ,w,d,k,t,x,y,p,m,v,q,r,s,z')

def prepend_latex(letex_str,sympy_expr): 
    return Math(letex_str+sp.latex(sympy_expr))

def RM(MGF,i):
    '''
    symbolic function returns i-th (start from 1) raw moment: E[Y^n] given Moment-generating function MGF 
    '''
    return sp.diff(MGF,z,i).subs(z,0) 

def RMgenerator(MGF): 
    '''
    returns symbolic function returning i-th (start from 1) raw moment: E[Y^n]
    '''
    return lambda i:sp.diff(MGF,z,i).subs(z,0) 

def EM(MGF,i): 
    '''
    symbolic function returns i-th (start from 1) exp moment: E[e^Y*Y^i]
    '''
    return sp.diff(MGF,z,i).subs(z,1)

def EMgenerator(MGF): 
    return lambda i:EM(MGF,i)

def CM(rm,i): 
    '''
    symbolic function returns i-th (start from 1) central moment: E[(Y-E[Y])^n]
    if i==1 returns 1st raw moment (1st central moment always 0) 
    rm - raw moment generator
    '''
    if i==1: return rm(1)
    x = sp.symbols('u_s')
    expr=sp.expand((x-rm(1))**i)
    for n in range(i,0,-1):
        expr=expr.subs(x**n,rm(n))
    return sp.simplify(expr)

def CMgenerator(rm):
    '''
    returns the function generating central moments for raw moments generator - rm
    '''
    return lambda i:CM(rm,i)

def DRM(MGF,i):
    '''
    symbolic function returns differential of i-th (start from 1) raw moment
     '''
    if i==0: return -EM(MGF,0)
    return sp.simplify(RM(MGF,i)*EM(MGF,0)-EM(MGF,i))

def DRMgenerator(MGF):  
    return lambda i:DRM(MGF,i)

dU=sp.symbols('dU')

def RMSeries(MGF,i):
    '''
    series dm_i(U+dU) up to first term
    '''
    return sp.expand(RM(MGF,i)+DRM(MGF,i)*dU) 

def RMSgenerator(MGF): 
    return lambda i:RMSeries(MGF,i)

def DCM(MGF,i):
    '''
    symbolic function returns differential of i-th central moment
    '''
    if i==0:
        return -EM(MGF,0)  
    
    expr=CM(RMSgenerator(MGF),i)
    return sp.simplify(sp.diff(expr,dU,1).subs(dU,0))

def DCMgenerator(MGF):
    return lambda i:DCM(MGF,i)