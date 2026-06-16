def CDF_accuracy_plot(v: float,func_list, inv_deg=9, spread:float = 3.,accuracy_only: bool =False):
    s0=np.sqrt(v)
    T=np.logspace(-spread*s0,spread*s0,129,base=np.e)
    title=f'v={v:.3f} σ={s0:.3f} inversion degree is {inv_deg} '
    get_cdf_uv=get_inverse_cdf_uvF(inv_deg,True)
    get_cdf_u=get_inverse_cdf_uF(inv_deg,True)


    N=len(func_list)
    width = lambda i: 1.*(N-i)+0.5
    RCDF=scst.lognorm.cdf(T,s0)
    CDFs=[]
    for (_,F) in func_list:
        CDFs.append(get_cdf_uv(F,T,v) if F.__code__.co_argcount==2 else get_cdf_u(F,T))
#    fig, ax = plt.subplots(dpi=120)
    if not accuracy_only:
        plt.title(title)
        for i,(descr,_) in enumerate(func_list):
            CDF = CDFs[i]
            plt.plot(T,CDF, linewidth=width(i), label=descr)        
        plt.plot(T,RCDF, linewidth=width(N), label=f'Actual CDF')
        plt.xlabel('x')
        plt.ylabel('cdf')
        plt.xscale('log')
        plt.legend(loc='lower right')
        plt.show()
    
    plt.title('relative value to exact '+title)
    for i,(descr,_) in enumerate(func_list):
        CDF = CDFs[i]
        diff=np.abs(CDF-RCDF)/(1.-RCDF+1.e-12)+np.abs(CDF-RCDF)/(RCDF+1.e-12)
        head=np.median(diff[:15])
        tail=np.median(diff[-16:-1])
        worse=np.max(diff)
        avr=np.mean(diff)
        plt.plot(T,diff, linewidth=width(i), label=descr+f' h:{head:.2g} t:{tail:.2g} w:{worse:.2g} a:{avr:.2g}')        
    
    plt.xlabel('x')
    plt.legend(loc='lower right')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

#v=1.25
#CDF_accuracy_plot(v,(
#    ('DI B',False,LTLN_DI_real_uF(v)),
#    ('HA B',True,LTLN_DI_real_uvF()),
#    ('HA C',True,LTLN_DI_real_uvF(48)),
#),9,4,True)

def LT_plot(v: float,func_list, spread:float = 3.,accuracy_only: bool =False):
    s0=np.sqrt(v)
    T=np.logspace(-spread*s0,spread*s0,129,base=np.e)
    title=f'σ={s0:.3f} x-range (logarithmic scale) is ±{spread}σ '

    N=len(func_list)
    width = lambda i: 1.*(N-i)+0.5
    curves=[]
    for (_,F) in func_list:
        curves.append(F(T,v) if F.__code__.co_argcount==2 else F(T))
#    fig, ax = plt.subplots(dpi=120)
    base=None
    if not accuracy_only:
        plt.title(title)
        for i,(descr,_) in enumerate(func_list):
            Curve = curves[i]
            if i==0:
                base=Curve
                continue
            plt.plot(T,Curve, linewidth=width(i-1), label=descr)  
        plt.plot(T,base, linewidth=width(N-1), label=func_list[0][0]+' (base)')
            
        plt.xlabel('u')
        plt.ylabel('LT')
        plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='lower right')
        plt.show()
    
    plt.title('relative diff to base \n'+title)
    for i,(descr,_) in enumerate(func_list):
        if(i==0):
            continue
        Curve = curves[i]
        diff=np.abs(Curve-base)/(1.-base+1.e-12)+np.abs(Curve-base)/(base+1.e-12)
        head=np.median(diff[:15])
        tail=np.median(diff[-16:-1])
        worse=np.max(diff)
        avr=np.mean(diff)
        plt.plot(T,diff, linewidth=width(i), label=descr+f' h:{head:.2g} t:{tail:.2g} w:{worse:.2g} a:{avr:.2g}')        
    
    plt.xlabel('u')
    plt.legend(loc='lower right')
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
