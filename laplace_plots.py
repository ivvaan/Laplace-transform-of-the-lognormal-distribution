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


def LT_plot(v: float,func_list, spread:float = 3., npoints:int=129,accuracy_only: bool =False, 
            linear: bool =False, print_more:bool = False):
    s0=np.sqrt(v)
    if linear:
        T=np.linspace(0,np.pow(10,spread),npoints)
    else:
        T=np.logspace(-spread,spread,npoints,base=10)
    title=f'σ={s0:.3f}'

    N=len(func_list)
    width = lambda i: 1.*(N-i)+0.5
    curves=[]
    for (_,F) in func_list:
        curves.append(F(T,v) if F.__code__.co_argcount==2 else F(T))
    base=curves[0]
    log_base=np.log(base[linear:])
    if not accuracy_only:
        plt.title(title)
        for i,(descr,_) in enumerate(func_list):
            if i==0:
                continue
            plt.plot(T,curves[i], linewidth=width(i-1), label=descr)  
        plt.plot(T,base, linewidth=width(N-1), label=func_list[0][0]+' (base)')
            
        plt.xlabel('u')
        plt.ylabel('LT')
        if not linear:
            plt.xscale('log')
        plt.yscale('log')
        plt.legend(loc='lower right')
        plt.show()
    
    plt.title('relative logarithmic accuracy\n'+title)
    for i,(descr,_) in enumerate(func_list):
        if(i==0):
            continue
        log_curve=np.log(curves[i][linear:])
        if linear:
            diff=log_curve/log_base - 1
        else:
            diff=np.abs(log_curve/log_base - 1)
        more_info='';
        if print_more:
            head=np.median(diff[:15])
            tail=np.median(diff[-16:-1])
            worse=np.max(diff)
            avr=np.mean(diff)
            more_info=f' h:{head:.2g} t:{tail:.2g} w:{worse:.2g} a:{avr:.2g}'
        plt.plot(T[linear:],diff, linewidth=width(i), label=descr+more_info)        
    
    plt.xlabel('u')
    plt.legend(loc='lower right')
    if not linear:
        plt.ylabel('|ln(curve)/ln(base) - 1|')
        plt.yscale('log')
        plt.xscale('log')
    else:
        plt.ylabel('ln(curve)/ln(base) - 1')
    plt.savefig('comp.png',dpi=300)
    plt.show()

