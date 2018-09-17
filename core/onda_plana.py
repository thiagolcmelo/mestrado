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
from scipy.stats import norm, skewnorm
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

# ## Constantes físicas
# grandezas de interesse em unidades atomicas
au_l = cte.value('atomic unit of length')
au_t = cte.value('atomic unit of time')
au_e = cte.value('atomic unit of energy')
# outras relacoes de interesse
ev = cte.value('electron volt')
au2ang = au_l / 1e-10
au2ev = au_e / ev

def evolucao_analitica(zi=-20.0, zf=20, E=150.0, deltaz=5.0,
                       L=100.0, N=1024,
                       tempo=2.1739442773545673e-14,
                       simples=True):
    """
    Evolui uma onda com energia [E] e desvio padrão
    [deltaz] da posição inicial [zi] até a posição final
    [zf]. A evolução ocorre em um espaço unidimensional de
    lamanho [L] partido em [N] pontos. Esta evolução é
    pseudo-analítica, pois assume que a integração numérica
    pode ser realizada com grande precisão.

    Parâmetros
    ----------
    zi : float
        Posição inicial em Angstrom
    zf : float
        Posição final em Angstrom
    E : float
        Energia da onda em eV
    deltaz : float
        Desvio padrão em Angstrom
    L : float
        Compimento do sistema em Angstrom
    N : int
        Número de pontos em que [L] será partido
    tempo : float
        O tempo para evolução em segundos
    simples : bool
        Se True, evolui pelo tempo t, se False, dá
        preferência para atingir zf

    Retorna
    -------
    resumo : dict
        Um dicionário com as seguintes chaves:
        - [z_si] malha de pontos em z (SI)
        - [z_au] malha de pontos em z (AU)
        - [z_ang] malha de pontos em z em Angstrom
        - [wave_initial_au] pacote de onda inicial (AU)
        - [wave_final_au] pacote de onda final (AU)
        - [a_initial] norma inicial <psi|psi>
        - [a_final] norma final <psi|psi>
        - [conservation] 100 * [a_final] / [a_initial]
        - [stdev] desvio padrao final em Angstrom
        - [skewness] obliquidade final
        - [tempo] o tempo físico para a onda ir de [zi]
          até [zf]
        - [zf_real] a posição final real
        - também retorna os parâmetros, para facilitar
          salvar em um .csv
    """
    result = locals().copy()
    # mudando para AU
    L_au = L / au2ang
    E_au = E / au2ev
    deltaz_au = deltaz / au2ang
    zi_au = zi / au2ang
    zf_au = zf / au2ang
    k0_au = np.sqrt(2 * E_au)
    # malha direta e recíproca
    z_au = np.linspace(-L_au/2.0, L_au/2.0, N)
    dz_au = np.abs(z_au[1] - z_au[0])
    k_au = fftfreq(N, d=dz_au)
    # tempos
    tempo_aux = tempo / 1e4
    # valores iniciais
    zm_au = zi_au
    zm_au_aux = zi_au
    # pacote de onda inicial
    PN = 1 / (2 * np.pi * deltaz_au ** 2) ** (1 / 4)
    psi = PN * np.exp(1j * k0_au * z_au - \
                      (z_au - zi_au) **2 / \
                      (4 * deltaz_au ** 2))
    psi_initial = np.copy(psi)  # salva uma copia
    # valores iniciais
    A = A0 = np.sqrt(simps(np.abs(psi) ** 2, z_au))
    zm_au = zi_au
    stdev_au = deltaz_au
    skewness = 0.0
    
    while np.abs(zm_au - zf_au) >= 0.00001:
        # novo tempo
        t_au = (tempo) / au_t
        # pacote de onda inicial
        psi = np.copy(psi_initial)
        ####################################################
        # núcleo da solução pseudo-analítica
        psi_k = fft(psi)
        omega_k = k_au**2 / 2
        psi = ifft(psi_k * np.exp(-1j * omega_k * t_au))
        ####################################################
        # main indicadores principais
        A2 = simps(np.abs(psi)**2, z_au).real
        A = np.sqrt(A2)  # norma
        psic = np.conjugate(psi)
        zm_au = (simps(psic * z_au * psi, z_au)).real / A2
        # ajuste do passo de tempo
        if np.abs(zm_au - zf_au) >= 0.00001 and not simples:
            if zm_au_aux < zf_au < zm_au \
                or zm_au < zf_au < zm_au_aux:
                aux = (tempo_aux-tempo) / 2
            elif zf_au < zm_au and zf_au < zm_au_aux:
                aux = - abs(tempo_aux-tempo)
            elif zf_au > zm_au and zf_au > zm_au_aux:
                aux = abs(tempo_aux-tempo)
            tempo_aux = tempo
            tempo += aux
            zm_au_aux = zm_au
            continue
        # indicadores secundários
        zm2 = simps(psic * z_au ** 2 * psi, z_au).real / A2
        zm3 = simps(psic * z_au ** 3 * psi, z_au).real / A2
        stdev_au = np.sqrt(np.abs(zm2-zm_au**2))
        skewness = (zm3 - 3 * zm_au * stdev_au ** 2 - \
                    zm_au ** 3) / stdev_au**3
        if simples:
            break
    result.update({
        'z_si': z_au * au2ang * 1e-10,
        'z_au': z_au,
        'z_ang': z_au * au2ang,
        'wave_initial': psi_initial,
        'wave_final': psi,
        'a_initial': A0,
        'a_final': A,
        'conservation': 100 * A / A0,
        'stdev': stdev_au * au2ang,
        'skewness': skewness,
        'tempo': tempo,
        'zf_real': zm_au * au2ang
    })
    return result


def evolucao_numerica(zi=-20.0, zf=20, E=150.0, deltaz=5.0,
                      L=100.0, N=1024, dt=1e-20, method='pe',
                      tempo=2.1739442773545673e-14,
                      simples=False):
    """
    Evolui uma onda com energia [E] e desvio padrão [deltaz]
    da posição inicial [zi] até a posição final [zf]. A 
    evolução ocorre em um espaço unidimensional de lamanho 
    [L] partido em [N] pontos. O método a ser utilizado 
    deve ser escolhido entre Pseudo-Espectral,
    Crank-Nicolson e Runge-Kutta

    Parâmetros
    ----------
    zi : float
        Posição inicial em Angstrom
    zf : float
        Posição final em Angstrom
    E : float
        Energia da onda em eV
    deltaz : float
        Desvio padrão em Angstrom
    L : float
        Compimento do sistema em Angstrom
    N : integer
        Número de pontos em que [L] será partido
    dt : float
        O passo de tempo em segundos
    method : string
        são aceitos:
        - 'pe' : Pseudo-Espectral
        - 'cn' : Crank-Nicolson
        - 'rk' : Runge-Kutta
    tempo : float
        O tempo para evolução em segundos
    simples : bool
        Se True, evolui pelo tempo t, se False, dá
        preferência para atingir zf

    Retorna
    -------
    resumo : dict
        Um dicionário com as seguintes chaves:
        - [z_si] malha de pontos em z (SI)
        - [z_au] malha de pontos em z (AU)
        - [z_ang] malha de pontos em z em Angstrom
        - [wave_initial_au] pacote de onda inicial (AU)
        - [wave_final_au] pacote de onda final (AU)
        - [a_initial] norma inicial <psi|psi>
        - [a_final] norma final <psi|psi>
        - [conservation] 100 * [a_final] / [a_initial]
        - [stdev] desvio padrao final em Angstrom
        - [skewness] obliquidade final
        - [zf_real] a posição final real
        - [program_time] o tempo de processamento
        - [iterations] o número de iterações de tempo [dt]
        - [tempo] é o tempo do sistema em segundos
        - também retorna os parâmetros, para facilitar
          salvar em um .csv
    """
    result = locals().copy()
    # mudando para AU
    L_au = L / au2ang
    dt_au = dt / au_t
    E_au = E / au2ev
    deltaz_au = deltaz / au2ang
    zi_au = zi / au2ang
    zf_au = zf / au2ang
    k0_au = np.sqrt(2 * E_au)
    # malha direta e recíproca
    z_au = np.linspace(-L_au/2.0, L_au/2.0, N)
    dz_au = np.abs(z_au[1] - z_au[0])
    k_au = fftfreq(N, d=dz_au)
    # começa a contar o tempo do programa
    time_inicial = time.time()
    # runge-kutta 4th order
    if method == 'rk':
        alpha = 1j / (2 * dz_au ** 2)
        beta = - 1j / (dz_au ** 2)
        diagonal_1 = [beta] * N
        diagonal_2 = [alpha] * (N - 1)
        diagonals = [diagonal_1, diagonal_2, diagonal_2]
        D = diags(diagonals, [0, -1, 1]).toarray()
        def evolution(p):
            k1 = D.dot(p)
            k2 = D.dot(p + dt_au * k1 / 2)
            k3 = D.dot(p + dt_au * k2 / 2)
            k4 = D.dot(p + dt_au * k3)
            return p + dt_au * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        def evolution_operator(p): return evolution(p)
    # crank-nicolson
    if method == 'cn':
        alpha = - dt_au * (1j / (2 * dz_au ** 2))/2.0
        beta = 1.0 - dt_au * (- 1j / (dz_au ** 2))/2.0
        gamma = 1.0 + dt_au * (- 1j / (dz_au ** 2))/2.0
        diagonal_1 = [beta] * N
        diagonal_2 = [alpha] * (N - 1)
        diagonals = [diagonal_1, diagonal_2, diagonal_2]
        invB = inv(diags(diagonals, [0, -1, 1]).toarray())
        diagonal_3 = [gamma] * N
        diagonal_4 = [-alpha] * (N - 1)
        diagonals_2 = [diagonal_3, diagonal_4, diagonal_4]
        C = diags(diagonals_2, [0, -1, 1]).toarray()
        D = invB.dot(C)
        def evolution_operator(p): return D.dot(p)
    # split step
    if method == 'pe':
        exp_v2 = np.ones(N, dtype=np.complex_)
        exp_t = np.exp(- 0.5j * (2 * np.pi * k_au) ** 2 * dt_au)
        def evolution_operator(p): return exp_v2 * \
                                   ifft(exp_t * fft(exp_v2 * p))
    # pacote de onda inicial
    PN = 1 / (2 * np.pi * deltaz_au ** 2) ** (1 / 4)
    psi = PN * np.exp(1j * k0_au * z_au - \
                      (z_au - zi_au) ** 2 / \
                      (4 * deltaz_au ** 2))
    psi_initial = np.copy(psi)  # salva uma cópia
    # norma inicial <psi|psi>
    A = A0 = np.sqrt(simps(np.abs(psi) ** 2, z_au))
    # valores iniciais
    zm_au = zi_au
    stdev_au = deltaz_au
    skewness = 0.0
    iterations = 0
    total_time = time.time() - time_inicial
    tempo_sistema_au = 0.0
    tempo_au = tempo / au_t
    while zm_au < zf_au:
        psi = evolution_operator(psi)
        tempo_sistema_au += dt_au
        iterations += 1
        # se a onda se mover para o lado oposto, já identifica logo
        # no início e inverte o sentido
        if zm_au < zi_au:
            k0_au *= -1.0
            psi = PN * np.exp(1j * k0_au * z_au - \
                              (z_au - zi_au) ** 2 / \
                              (4 * deltaz ** 2))
            zm_au = zi_au
            tempo_sistema_au = 0.0
            iterations = 0
            continue
        # indicadores principais
        A2 = simps(np.abs(psi) ** 2, z_au).real
        A = np.sqrt(A2)
        psic = np.conjugate(psi)
        # <psi|z|psi>
        zm_au = (simps(psic * z_au * psi, z_au)).real / A2
        # para de medir o tempo (pode ser o fim)
        total_time = time.time() - time_inicial
        # se <z> >= zf ou o tempo ultrapassou 300 segundos
        if zm_au >= zf_au or total_time > 300 \
            or (simples and tempo_sistema_au >= tempo_au):
            # indicadores secundários
            zm2 = simps(psic * z_au ** 2 * psi, z_au).real / A2
            zm3 = simps(psic * z_au ** 3 * psi, z_au).real / A2
            stdev_au = np.sqrt(np.abs(zm2 - zm_au ** 2))
            skewness = (zm3 - 3 * zm_au * stdev_au ** 2 - \
                        zm_au ** 3) / stdev_au ** 3
            if total_time > 300 or simples:
                break
    result.update({
        'zi': zi,
        'zf': zf,
        'E': E,
        'deltaz': deltaz,
        'L': L,
        'N': N,
        'dt': dt,
        'method': method,
        'z_si': z_au * au2ang * 1e-10,
        'z_au': z_au,
        'z_ang': z_au * au2ang,
        'wave_initial': psi_initial,
        'wave_final': psi,
        'a_initial': A0,
        'a_final': A,
        'conservation': 100 * A / A0,
        'stdev': stdev_au * au2ang,
        'skewness': skewness,
        'zf_real': zm_au * au2ang,
        'program_time': total_time,
        'iterations': iterations,
        'tempo': tempo_sistema_au * au_t
    })
    return result