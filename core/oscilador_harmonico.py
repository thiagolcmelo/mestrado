#!/usr/bin/env python
# -*- coding: utf-8 -*-

# python standard
import time

# third-party
import numpy as np
import pandas as pd
import scipy.constants as cte
from scipy.integrate import simps
from scipy.sparse import diags
from scipy.linalg import inv
from scipy.fftpack import fft, ifft, fftfreq
from scipy.signal import gaussian
from scipy.special import factorial, hermite, legendre
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# locals
from core.utilidades import chutes_iniciais, autovalor

# ## Constantes físicas
# grandezas de interesse em unidades atomicas
au_l = cte.value('atomic unit of length')
au_t = cte.value('atomic unit of time')
au_e = cte.value('atomic unit of energy')
# outras relacoes de interesse
ev = cte.value('electron volt')
c = cte.value('speed of light in vacuum')
hbar_si = cte.value('Planck constant over 2 pi')
me = cte.value('electron mass')
au2ang = au_l / 1e-10
au2ev = au_e / ev

############################################################
# ## Potencial
def omega(wave_length:float)->tuple:
    """
    Para um dado comprimento de onda [wave_length] em
    metros, retorna a frequência angular em rad/sec

    Parâmetros
    ----------
    wave_length : float
        o comprimento de onda em metros

    Retorna
    -------
    ang_freq : tuple
        frequência angular (ang_freq_si, ang_freq_au)
    """
    f = c / wave_length  # Hz
    w = 2.0 * np.pi * f
    return w, w * au_t

def potencial_au(wave_length:float, L:float, N:int)->tuple:
    """
    Para um [wave_length] (contido em um espaço de tamanho
    [L] e representado por [N] pontos), esta função retorna
    o potencial do oscilador harmônico quântico associado.
    A origem é posicionada no meio do potencial [-L/2,+L/2].

    Parâmetros
    ----------
    wave_length : float
        o comprimento de onda em metros
    L : float
        o tamanho do sistema em Angstrom
    N : int
        o número de pontos no espaço

    Retorna
    -------
    potencial : tuple
        (z_si, z_au, v_ev, v_si, v_au) onde:
            - [z_si] é a malha espacial (SI)
            - [z_au] é a malha espacial (AU)
            - [v_si] é o potencial (SI)
            - [v_ev] é o potencial (eV)
            - [v_au] é o potencial (AU)
    """
    w, _ = omega(wave_length)
    z_si = np.linspace(-(L/2) * 1e-10, (L/2) * 1e-10, N)
    z_au = np.linspace(-L/au2ang/2.0, L/au2ang/2.0, N)
    v_si = 0.5 * me * z_si**2 * w**2 # potential in Joules
    v_ev = v_si / ev # Joules to eV
    v_au = v_ev / au2ev # eV to au
    return z_si, z_au, v_ev, v_si, v_au

# # Solução analítica
def solucao_analitica(L:float=100.0, N:int=2048,
                      wave_length:float=8.1e-6, nmax:int=6)->dict:
    """
    Esta função calcula analiticamente os primeiros [nmax]
    autovalores e autofunções para um oscilador harmônico
    quântico com frequência angular correspondente a um
    comprimento de onda [wave_length].

    Parâmetros
    ----------
    L : float
        tamanho do sistema em Angstrom
    N : int
        numero de pontos
    wave_length : float
        comprimento de onda em metros
    nmax : int
        numero de autoestados e autofunções a serem
        calculados
    
    Retorna
    -------
    result : dictionary
        Um dicionário com as seguintes chaves:
        - `z_si` malha espacial (SI)
        - `z_au` malha espacial (AU)
        - `v_au` potencial (AU)
        - `v_ev` potencial (eV)
        - `v_si` potencial (SI)
        - `eigenvalues_si` autovalores (Joules)
        - `eigenvalues_ev` autovalores (eV)
        - `eigenvalues_au` autovalores (AU)
        - `eigenstates_au` autofunções (AU)
        - `eigenstates_2_au` autofunções na forma 
          |psi|^2 (AU)
        - `eigenstates_si` autofunções (SI)
        - `eigenstates_2_si` autofunções na forma
          |psi|^2 (SI)
    """
    # malha espacial
    z_si, z_au, v_ev, v_si, v_au = \
        potencial_au(wave_length, L, N)
    w, w_au = omega(wave_length)

    # nmax autovalores
    eigenvalues_si = [hbar_si * w * (n+1/2) for n in range(nmax)]
    eigenvalues_si = np.array(eigenvalues_si)
    eigenvalues_ev = eigenvalues_si / ev

    # nmax autoestados
    eigenstates_si = []
    eigenstates_au = []
    mwoh_au = w_au # m * w / hbar em AU
    mwoh_si = me * w / hbar_si # m * w / hbar em unidades do si
    for n in range(nmax):
        an_au = np.sqrt(1.0/(2.0**n * factorial(n))) * \
                        (mwoh_au/np.pi)**(1.0/4.0)
        psin_au = an_au*np.exp(-mwoh_au*z_au**2/2.0) * \
                        hermite(n)(np.sqrt(mwoh_au)*z_au)
        eigenstates_au.append(psin_au)

        an_si = np.sqrt(1.0/(2.0**n * factorial(n))) * \
                        (mwoh_si/np.pi)**(1.0/4.0)
        psin_si = an_si*np.exp(-mwoh_si*z_si**2/2.0) * \
                        hermite(n)(np.sqrt(mwoh_si)*z_si)
        eigenstates_si.append(psin_si)

    return {
        'z_si': z_si,
        'z_au': z_au,
        'v_au': v_au,
        'v_ev': v_ev,
        'v_si': v_si,
        'eigenvalues_si': eigenvalues_si,
        'eigenvalues_ev': eigenvalues_ev,
        'eigenvalues_au': eigenvalues_ev / au2ev,
        'eigenstates_au': eigenstates_au,
        'eigenstates_2_au': np.abs(eigenstates_au)**2,
        'eigenstates_si': eigenstates_si,
        'eigenstates_2_si': np.abs(eigenstates_si)**2,
    }

# # Solução numérica
def solucao_numerica(L:float=100.0, N:int=1024, dt:float=1e-18,
                     wave_length:float=8.1e-6, nmax:int=6,
                     precision:float=1e-2, iterations:int=None,
                     max_time:float=None, eigenstates_au:list=None,
                     method:str='pe', salvar=False)->dict:
    """
    Esta função calcula numericamente os primeiros [nmax]
    autovalores e autofunções para um oscilador harmônico
    quântico com frequência angular correspondente a um
    comprimento de onda [wave_length].

    Parameters
    ----------
    L : float
        tamanho do sistema em Angstrom
    N : int
        numero de pontos
    wave_length : float
        comprimento de onda em metros
    nmax : int
        numero de autoestados e autofunções a serem
        calculados
    dt : float
        o passo de tempo em segundos
    precision : float
        a convergência mínima no autovalor
    iterations : int
        o número máximo de iterações
    max_time : float
        o tempo máximo de processamento
    eigenstates_au : array_like
        um array com os chutes iniciais
    method : string
        o método pode ser:
            - 'pe' para Pseudo-Espectral
            - 'ii' para Interação Inversa
    salvar : bool
        fazer um registro dos autovetores ao longo da evolucao
    
    Returns
    -------
    result : dictionary
        Um dicionário com as seguintes chaves:
        - `z_si` malha espacial (SI)
        - `z_au` malha espacial (AU)
        - `v_au` potencial (AU)
        - `v_ev` potencial (eV)
        - `v_si` potencial (SI)
        - `eigenvalues_si` autovalores (Joules)
        - `eigenvalues_ev` autovalores (eV)
        - `eigenvalues_au` autovalores (AU)
        - `eigenstates_au` autofunções (AU)
        - `eigenstates_2_au` autofunções na forma
          |psi|^2 (AU)
        - `eigenstates_si` autofunções (SI)
        - `eigenstates_2_si` autofunções na forma
          |psi|^2 (SI)
        - `iterations` um array com o número de iterações
          por autovalor
        - `timers` um array com o tempo de processamento por
          autovalor
        - `precisions` um array com a precisão por autovalor
        - `chebyshev` distânca de chebyshev por autofunção
        - `seuclidean` distânca euclidiana por autofunção
        - `sqeuclidean` distânca quadrada euclidiana por
          autofunção
    """
    # soluções analíticas
    analytical = solucao_analitica(L=L, N=N,
                                   wave_length=wave_length,
                                   nmax=nmax)
    eigenvalues_ev_ana = analytical['eigenvalues_ev']
    eigenstates_au_ana = analytical['eigenstates_au']
    # grid values
    z_si, z_au, v_ev, v_si, v_au = \
        potencial_au(wave_length, L, N)
    dt_au = -1j * dt / au_t
    precision /= 100 # it is a percentage
    forecast = eigenvalues_ev_ana.copy() / au2ev
    dz2 = (z_au[1]-z_au[0])**2
    # split step
    meff = np.ones(N)
    dz_au = np.abs(z_au[1] - z_au[0])
    k_au = fftfreq(N, d=dz_au)
    exp_v2 = np.exp(- 0.5j * v_au * dt_au)
    exp_t = np.exp(- 0.5j * (2 * np.pi * k_au) ** 2 * dt_au)
    evolution_operator = lambda p: exp_v2*ifft(exp_t*fft(exp_v2*p))
    # chutes iniciais
    if not eigenstates_au:
        eigenstates_au = chutes_iniciais(n=nmax, tamanho=N)
        eigenvalues_ev = np.zeros(nmax)
    counters = np.zeros(nmax)
    timers = np.zeros(nmax)
    precisions = np.zeros(nmax)
    vectors_chebyshev = np.zeros(nmax)
    vectors_sqeuclidean = np.zeros(nmax)
    vectors_seuclidean = np.zeros(nmax)
    # matrix diagonals
    sub_diag = -(0.5 / dz2) * np.ones(N-1, dtype=np.complex_)
    main_diag = np.zeros(N  , dtype=np.complex_)
    def get_invA(v_shift=0.0):
        "aplica um deslocamento no potencial, o mesmo que H'=H-shift "
        main_diag = (v_au-v_shift+1.0/dz2)
        diagonals = [main_diag, sub_diag, sub_diag]
        return inv(diags(diagonals, [0, -1, 1]).toarray())

    registro = []
    if method == 'pe':
        for s in range(nmax):
            salvando = 1
            while True:
                start_time = time.time()
                eigenstates_au[s] = \
                    evolution_operator(eigenstates_au[s])
                counters[s] += 1
                # gram-shimdt
                for m in range(s):
                    proj = simps(eigenstates_au[s] * \
                           np.conjugate(eigenstates_au[m]), z_au)
                    eigenstates_au[s] -= proj * eigenstates_au[m]
                # normalize
                A = np.sqrt(simps(np.abs(eigenstates_au[s])**2, z_au))
                eigenstates_au[s] /= A
                timers[s] += time.time() - start_time
                
                if salvar and (counters[s] >= salvando \
                    or (iterations and counters[s] >= iterations)):
                    psi2 = (np.abs(eigenstates_au[s])**2).real
                    av = autovalor(z_au, v_au,
                                   eigenstates_au[s],
                                   meff)
                    registro += [{
                        "nivel": s,
                        "iteracoes": min(salvando, iterations),
                        "autovalor": av,
                        "z": z,
                        "autovetor": p.real,
                        "autovetor2": p2
                    } for z, p, p2 in zip(z_au, eigenstates_au[s], psi2)]
                    salvando *= 2
                
                if (iterations and counters[s] >= iterations) \
                    or (max_time and timers[s] >= max_time) \
                    or counters[s] % 1000 == 0:
                    ev = autovalor(z_au, v_au, eigenstates_au[s], meff)
                    eigenvalues_ev[s] = ev * au2ev # eV
                    precisions[s] = np.abs(1-eigenvalues_ev[s] / \
                                           eigenvalues_ev_ana[s])
                    if salvar:
                        filename = 'saidas/oscilador_harmonico_{}.pkl'
                        filename = filename.format(iterations)
                        pd.DataFrame(registro).to_pickle(filename)
                    if (iterations and counters[s] >= iterations) \
                        or (max_time and timers[s] >= max_time) \
                        or (not iterations and not max_time \
                            and precisions[s] < precision):
                        XA = [eigenstates_au[s]]
                        XB = [eigenstates_au_ana[s]]
                        vectors_chebyshev[s] = cdist(XA, XB,
                            'chebyshev')[0][0]
                        vectors_seuclidean[s] = cdist(XA, XB, 
                            'seuclidean')[0][0]
                        vectors_sqeuclidean[s] = cdist(XA, XB,
                            'sqeuclidean')[0][0]
                        break
    elif method == 'ii':
        for s in range(nmax):
            last_ev = 1.0
            last_es = np.zeros(N, dtype=np.complex_)
            shift     = forecast[s]
            invA      = get_invA(shift)
            V_shifted = v_au-shift
            while True:
                start_time = time.time()
                eigenstates_au[s] = invA.dot(eigenstates_au[s])
                counters[s] += 1
                # normalize
                A = np.sqrt(simps(eigenstates_au[s] * \
                                  eigenstates_au[s].conj(), z_au))
                eigenstates_au[s] /= A
                timers[s] += time.time() - start_time
                if (iterations and counters[s] >= iterations) \
                    or (max_time and timers[s] >= max_time) \
                    or counters[s] % 100 == 0:
                    eigenvalues_ev[s] = ev * au2ev # eV

#                     # second derivative
#                     derivative2 = (eigenstates_au[s][:-2] - 2 * \
#                                    eigenstates_au[s][1:-1] + \
#                                    eigenstates_au[s][2:]) / dz_au**2
#                     psi = eigenstates_au[s][1:-1]
#                     # <Psi|H|Psi>
#                     p_h_p = simps(psi.conj() * (-0.5 * derivative2 + \
#                                   V_shifted[1:-1] * psi), z_au[1:-1])
#                     # divide por <Psi|Psi> 
#                     p_h_p /= A**2
#                     eigenvalues_ev[s] = (p_h_p.real + shift) * au2ev # eV

                    # descobre se é repetido
                    eigenvalues = np.array([ev for ev in \
                                            eigenvalues_ev \
                                            if ev != 0.0])
                    eigenvalues = eigenvalues[eigenvalues.argsort()]
                    golden_ones = [0]
                    for i in range(eigenvalues.size):
                        # drop repeated and unbounded states
                        pres = np.abs(eigenvalues[i]/eigenvalues[i-1]-1)
                        if i == 0 or pres < 0.1 \
                           or eigenvalues[i] > np.max(v_ev):
                            continue
                        golden_ones.append(i)  
                    if len(golden_ones) < len(eigenvalues):
                        forecast_diff = forecast[-1] - forecast[-2]
                        forecast_max = max(forecast)
                        forecast.pop(s)
                        forecast.append(forecast_max + forecast_diff)
                        s -= 1
                        break
                    precisions[s] = np.abs(1-eigenvalues_ev[s] / \
                                           eigenvalues_ev_ana[s])
                    if (iterations and counters[s] >= iterations) \
                        or (max_time and timers[s] >= max_time) \
                        or (not iterations and not max_time \
                            and precisions[s] < precision):
                        XA = [eigenstates_au[s]]
                        XB = [eigenstates_au_ana[s]]
                        vectors_chebyshev[s] = cdist(XA, XB,
                            'chebyshev')[0][0]
                        vectors_seuclidean[s] = cdist(XA, XB,
                            'seuclidean')[0][0]
                        vectors_sqeuclidean[s] = cdist(XA, XB,
                            'sqeuclidean')[0][0]
                        break
                        
    # salva evolucao
#     if salvar:
#         registro = pd.DataFrame(registro)
#         registro.to_pickle('saidas/oscilado_harmonico_{}.pkl'.format(iterations))
    # gera autoestados no SI
    eigenstates_si = np.array([np.ones(N, dtype=np.complex_) \
                               for i in range(nmax)],dtype=np.complex_)
    for i, state in enumerate(eigenstates_au):
        A_si = np.sqrt(simps(np.abs(state)**2, z_si))
        eigenstates_si[i] = state / A_si
    return {
        'z_si': z_si,
        'z_au': z_au,
        'v_au': v_au,
        'v_ev': v_ev,
        'v_si': v_si,
        'eigenvalues_si': eigenvalues_ev * ev,
        'eigenvalues_ev': eigenvalues_ev,
        'eigenvalues_au': eigenvalues_ev / au2ev,
        'eigenstates_au': eigenstates_au,
        'eigenstates_2_au': np.abs(eigenstates_au)**2,
        'eigenstates_si': eigenstates_si,
        'eigenstates_2_si': np.abs(eigenstates_si)**2,
        'iterations': counters,
        'timers': timers,
        'precisions': precisions,
        'chebyshev': vectors_chebyshev,
        'seuclidean': vectors_seuclidean,
        'sqeuclidean': vectors_sqeuclidean
    }