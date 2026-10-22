"""Recalentamiento 1D de Pennes: MDF implicitas (Robin orden 1 y 2) y MVF.

Requisitos: numpy, matplotlib (solo graficos). Ejecutar:
    python comparar_mdf_mvf.py

Modelo basado en Rojczyk et al. (2015). Reimplementacion independiente:
no es el codigo MATLAB ni la comparacion ANSYS de los autores.
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
L, TA, TINF, H = 0.005, 37.0, 23.0, 10.0  # m, C, C, W/(m2 K)
RHO, CT, K = 1085., 3680., 0.47
RHOB, CB = 1060., 3617.
A = RHO * CT
DT, TCOOL, THEAT = 0.05, 15.0, 50.0

class Tridiagonal:
    """Factored constant tridiagonal system; lower/upper length n-1."""
    def __init__(self, lower, diag, upper):
        self.l = np.asarray(lower, dtype=float)
        self.u = np.asarray(upper, dtype=float)
        self.diag = np.asarray(diag, dtype=float)
        n = len(diag)
        assert len(self.l) == len(self.u) == n-1
        self.d = np.empty(n)
        self.c = np.zeros(n)
        self.d[0] = self.diag[0]
        for i in range(n - 1):
            self.c[i] = self.u[i]/self.d[i]
            self.d[i+1] = self.diag[i+1] - self.l[i]*self.c[i]
        if np.any(np.abs(self.d) < 1e-16):
            raise ValueError('Singular linear system')

    def solve(self, b):
        b = np.asarray(b,dtype=float)
        y=np.empty_like(b)
        y[0]=b[0]/self.d[0]
        for i in range(1,len(b)):
            y[i]=(b[i]-self.l[i-1]*y[i-1])/self.d[i]
        x=np.empty_like(b)
        x[-1]=y[-1]
        for i in range(len(b)-2,-1,-1):
            x[i]=y[i]-self.c[i]*x[i+1]
        return x


def params(case):
    perf,met = (0.00105,631.) if case=='sano' else (0.00525,6310.)
    return RHOB*CB*perf, met


def fdm(N=27, case='sano', robin_order=2):
    """Interior x_1...x_(N-1), x_0 Dirichlet, x_N reconstructed.
    Robin exterior de orden 1 o 2, eliminando x_N para matriz tridiagonal.
    """
    assert N >= 4 and robin_order in (1,2)
    dx = L/N
    r=K/dx**2; a=A/DT
    C,D=params(case)
    m=N-1; x=np.linspace(0,L,N+1)
    # Robin: T_N = (h*Tinf + p*T_(N-1) + q*T_(N-2))/Hden
    if robin_order == 1:
        p=K/dx; q=0.; den=H+p
    else:
        kap=K/(2*dx); p=4*kap; q=-kap; den=H+3*kap
    robin_const=H*TINF/den
    def matrix(acc, stage):
        lower=np.full(m-1,-r)
        upper=np.full(m-1,-r)
        diag=np.full(m,acc+2*r+C)
        if stage=='reheat':
            diag[-1]-=r*p/den
            lower[-1]-=r*q/den # note minus r*q / den, q<0 => positive correction
        return Tridiagonal(lower,diag,upper)
    def rhs(state, acc, stage):
        b=np.full(m,C*TA+D)
        if state is not None:
            b+=acc*state
        b[0]+=r*TA
        if stage=='reheat':
            b[-1]+=r*robin_const
        return b
    T = matrix(0.,'reheat').solve(rhs(None,0.,'reheat'))
    cool_solver=matrix(a,'cool')
    for _ in range(round(TCOOL/DT)):
        T=cool_solver.solve(rhs(T,a,'cool'))
    heating_solver=matrix(a,'reheat')
    curve=[]
    times=np.arange(round(THEAT/DT)+1)*DT
    # t=0: immediately after cooling, exterior surface is at 0 deg C.
    curve.append(0.)
    for _ in range(round(THEAT/DT)):
        T=heating_solver.solve(rhs(T,a,'reheat'))
        surface=robin_const+(p*T[-1]+q*T[-2])/den
        curve.append(surface)
    return times,np.asarray(curve), x


def fvm(N=27, case='sano'):
    """N volumes centrados. A x=0 se impone Ta, a x=L Robin.
    Resistencia media celda + conveccion: geff=1/(dx/(2*k)+1/h).
    """
    assert N>=3
    dx=L/N
    r=K/dx**2; a=A/DT
    C,D=params(case)
    x=(np.arange(N)+.5)*dx
    gext=1./(dx/(2*K)+1/H)
    ext_coef=gext/dx
    def matrix(acc,stage):
        lower=np.full(N-1,-r)
        upper=np.full(N-1,-r)
        diag=np.full(N,acc+2*r+C)
        diag[0]+=r  # 3*r en izquierda (conduccion 2*r a Dirichlet)
        if stage=='reheat':
            diag[-1]=acc+r+C+ext_coef
        else:
            diag[-1]=acc+3*r+C   # Dirichlet de superficie T=0
        return Tridiagonal(lower,diag,upper)
    def rhs(state,acc,stage):
        b=np.full(N,C*TA+D)
        if state is not None:
            b+=acc*state
        b[0]+=2*r*TA
        if stage=='reheat':
            b[-1]+=ext_coef*TINF
        return b
    t=matrix(0.,'reheat').solve(rhs(None,0.,'reheat'))
    cool_solver=matrix(a,'cool')
    for _ in range(round(TCOOL/DT)):
        t=cool_solver.solve(rhs(t,a,'cool'))
    heat_solver=matrix(a,'reheat')
    times=np.arange(round(THEAT/DT)+1)*DT
    surface=[0.]
    for _ in range(round(THEAT/DT)):
        t=heat_solver.solve(rhs(t,a,'reheat'))
        surface.append((2*K/dx*t[-1]+H*TINF)/(2*K/dx+H))
    return times,np.asarray(surface),x


def main():
    rows=[]
    for case in ('sano','tumoral'):
        print('CASO:',case,'C=',params(case)[0])
        for N in (27,54,108):
            t,v1,_=fdm(N,case,1)
            _,v2,_=fdm(N,case,2)
            _,vf,_=fvm(N,case)
            for tm in (1,5,10,25,50):
                i=round(tm/DT)
                rows.append(dict(caso=case,N=N,t_s=tm,mdf_robin_1_C=v1[i],mdf_robin_2_C=v2[i],mvf_C=vf[i],dif_mdf2_mvf_K=v2[i]-vf[i],dif_mdf1_mvf_K=v1[i]-vf[i]))
            print(' N=',N,'Tsurf50: FD1',round(v1[-1],5),'FD2',round(v2[-1],5),'FV',round(vf[-1],5),'dif FD2-FV',round(v2[-1]-vf[-1],5))
        t,v1,_=fdm(27,case,1)
        _,v2,_=fdm(27,case,2)
        _,vf,_=fvm(27,case)
        plt.figure(figsize=(8,4.6))
        plt.plot(t,v1,label='MDF + Robin de orden 1',lw=1.4)
        plt.plot(t,v2,label='MDF + Robin de orden 2',lw=1.4)
        plt.plot(t,vf,label='MVF centrado en volúmenes',ls='--',lw=1.4)
        plt.xlabel('Tiempo de recalentamiento (s)')
        plt.ylabel('Temperatura superficial (°C)')
        plt.title('Comparación de métodos: '+case+' (N = 27, Δt = 0,05 s)')
        plt.legend()
        plt.grid(alpha=.2)
        plt.tight_layout()
        plt.savefig(OUT/f'comparacion_{case}.png',dpi=190)
        plt.close()
    with (OUT/'comparacion_mdf_mvf.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    # Verificaciones: condicion inicial, monotonicidad y concordancia por refinamiento.
    for case in ('sano','tumoral'):
        t,u,_=fdm(108,case,2)
        assert abs(u[0]) < 1e-12, 'La superficie tras enfriamiento debe estar a 0 °C'
        assert 0 < u[1] < 37
        assert np.all(np.diff(u)>=-1e-10), 'La superficie debe recalentarse durante el experimento'
        _,v,_=fvm(108,case)
        assert abs(u[-1]-v[-1]) < .3, 'Desacuerdo excesivo entre métodos'
    print('Archivos generados en',OUT)

if __name__=='__main__':
    main()
