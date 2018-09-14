#!/usr/bin/env python
# -*- coding: utf-8 -*-

# python standard

# third-party
import numpy as np
import pandas as pd
from scipy.signal import gaussian
from scipy.integrate import simps
from scipy.sparse import diags
from scipy.linalg import inv
from scipy.special import legendre, expit
from scipy.spatial.distance import cdist


def idf(v: list, i: float) -> float:
    """
    Indice flexivel. Por exemplo, i pode ser 1.5 e o
    resultado sera v[2]*(2.0-1.5)+v[1]*(1.5-1.0),
    ou seja, (v[2]+v[1])/2

    Parâmetros
    ----------
    v : array_like
        um vetor
    i : float
        um indice flexivel

    Retorno
    -------
    Uma interpolacao simples de v para o indice flexivel i
    """
    i_up = int(np.ceil(i))
    i_down = int(np.floor(i))
    if i_down < 0.0:
        return v[0]
    elif i_up >= len(v) - 1:
        return v[-1]
    
    up = i - float(i_down)
    down = float(i_up) - i
    return v[i_up] * up + v[i_down] * down


def derivada_muda_pct(x:list, y:list, n:int=10, pct:float=0.05)->float:
    """
    Encontra o ponto x onde a derivada de y(x) muda mais do que uma 
    certa porcentagem pela primeira vez, da esquerda para a direita

    Parâmetros
    ----------
    x : array_like
        um array com os valores em x
    y : array_like
        um array com os valores em y
    n : int
        numero do pontos para ignorar nas bordas
    pct : float
        a porcentagem da derivada que deve mudar

    Retorno
    -------
    O indice do ponto x onde dy/dx muda mais do que pct
    """
    der_y = np.array(y[2:]-y[:-2])/np.array(x[2:]-x[:-2])
    for i in range(n, len(der_y)):
        last_n = np.average(der_y[i-n:i-1])
        if last_n == 0 and der_y[i] != 0 \
                or last_n != 0 and np.abs(der_y[i]/last_n-1) > pct:
            return i
    return int(len(y)/3)


def chutes_iniciais(n:int=2, tamanho:int=1024, mu:float=None)->list:
    """
    Retorna os n primeiros polinomios de legendre
    modulados por uma gaussiana.

    Parâmetros
    ----------
    n : int
        o numero de vetores
    tamanho : int
        o tamanho dos vetores
    mu : float
        centro da gaussiana, entre 0 e 1
    Retorno
    -------
    Um array com n arrays contendo os polinomios modulados
    """
    sg = np.linspace(-1, 1, tamanho)
    g = gaussian(tamanho, std=int(tamanho/100))
    if mu:
        sigma = np.ptp(sg)/100
        g = (1.0 / np.sqrt(2 * np.pi * sigma ** 2))
        g *= np.exp(-(sg - mu) ** 2 / (2 * sigma ** 2))
    vls = [g*legendre(i)(sg) for i in range(n)]
    return np.array(vls, dtype=np.complex_)


def autovalor(z: list, V: list, psi: list, m: list) -> float:
    """
    Calcula um autovalor como E=<Psi|H|Psi>/<Psi|Psi>
    onde H = T + V, T eh o operador de energia cinetica
    em uma dimensao

    Parâmetros
    ----------
    z : array_like
        o eixo z
    V : array_like
        o potencial
    psi : array_like
        a funcao de onda psi(z)
    m : array_like
        a massa efetiva m*(z)

    Retorno
    -------
    O autovalor E=<Psi|H|Psi>/<Psi|Psi>
    """
    dz = np.append(z[1:]-z[:-1], z[1]-z[0])
    dz2 = dz**2
    h_psi = np.zeros(N, dtype=np.complex_)
    for i in range(N):
        h_psi[i] = ((0.5/dz2[i])*(1.0/idf(m, i+0.5) +
                                  1.0/idf(m, i-0.5))+V[i])*psi[i]
        if i > 0:
            h_psi[i] += -(0.5/dz2[i])*(psi[i-1]/idf(m, i-0.5))
        if i < N-1:
            h_psi[i] += -(0.5/dz2[i])*(psi[i+1]/idf(m, i+0.5))
    psi_h_psi = simps(psi.conj()*h_psi, z)
    return (psi_h_psi / simps(psi.conj()*psi, z)).real


def interacao_inversa(z: list, V: list, m: list, nmax:int=20,
                      autovalores:list=[],
                      remover_repetidos:bool=True)->dict:
    """
    Calcula os autovetores e autovalores para uma partícula
    de massa efetiva m, submetida a um potencial V no espaço z
    Todas as entradas e saídas devem ser em unidades atômicas
    
    Parâmetros
    ----------
    z : array_like
        a malha espacial
    V : array_like
        o potencial em cada ponto da malha espacial
    m : array_like
        a massa efetiva da partícula em cada ponto da
        malha espacial
    nmax : int
        o número de chutes iniciais a serem utilizados
        entre min(V) e max(V)
    autovalores : array_like
        mesmo que aproximados, ajudam o algoritmo a convergir
        com maior velocidade e precisão
    remover_repetidos : bool
        remove os autoestados repetidos ou considerados "ruins"

    Retorno
    -------
    Um dicionário com as seguintes chaves:
    {
        "autovalores": [um array com autovalores],
        "autovetores": [um array de arrays com autovetores],
        "contadores": [array com # iterações por autoestado],
        "precisoes": [array com precisão de cada autovalor],
        "cronometros": [array com tempo para calcular cada autovetor],
        "euclides2": [array com distância euclidiana quadrada]*
    }
    * A distância euclidiana no caso indica a convergência do
      autovetor com ele mesmo, sendo a soma dos quadrados das
      diferenças ponto a ponto de um autovetor em um passo n
      e um passo n-1
    """
    # converte para array de numpy para facilitar
    z = np.array(z)
    V = np.array(V)
    m = np.array(m)
    
    # precisão mínima esperada
    precisao = 1e-9
    
    # número máximo de chutes entre Vmin e Vmax
    # estes serão os autovalores aproximados
    # para utilizar de chute inicial para o método
    # da interação inversa
    if len(autovalores) > 0:
        previsao = autovalores[:]
    else:
        previsao = np.linspace(V.min(), V.max(), nmax)
    
    # aqui esperamos dz constante
    dz = z[1]-z[0]
    dz2 = dz ** 2

    # chutes iniciais
    autovetores = chutes_iniciais(nmax, tamanho=N)
    autovalores = np.zeros(nmax)
    contadores = np.zeros(nmax)
    cronometros = np.zeros(nmax)
    precisoes = np.zeros(nmax)
    euclides2 = np.zeros(nmax)
        
    for s in range(nmax):
        last_ev = 1.0  # autovalor inicial fake
        last_es = np.zeros(N, dtype=np.complex_)  # autovetor inicial
        shift = previsao[s]

        # Desloca o potencial do Hamiltoniano por shift
        sub_diag = np.zeros(N-1, dtype=np.complex_)
        main_diag = np.zeros(N, dtype=np.complex_)

        # constroi as diagnais da matriz, a principal e as duas semi
        # principais
        for i in range(N):
            try:
                m1 = 1.0 / idf(m, i + 0.5)
                m2 = 1.0 / idf(m, i - 0.5)
                m_parte = (m1 + m2)
                main_diag[i] = (0.5 / dz2) * m_parte + (V[i] - shift)
            except:
                main_diag[i] = 0.0
            if i < N-1:
                sub_diag[i] = -(0.5/dz2)*(1.0/idf(m, i+0.5))
        diagonals = [main_diag, sub_diag, sub_diag]
        A = diags(diagonals, [0, -1, 1]).toarray()
        invA = inv(A)
        V_shifted = V-shift

        while True:
            start_time = time.time()
            autovetores[s] = invA.dot(autovetores[s])
            contadores[s] += 1

            # normaliza
            A = np.sqrt(simps(autovetores[s]*autovetores[s].conj(), z))
            autovetores[s] /= A
            cronometros[s] += time.time() - start_time
            autovalores[s] = autovalor(z, V_shifted,
                                        autovetores[s], m) + shift

            # confere precisao
            precisoes[s] = np.abs(1-autovalores[s]/last_ev)
            last_ev = autovalores[s]
            if precisoes[s] < precisao:
                XA = [np.abs(autovetores[s])**2]
                XB = [np.abs(last_es)**2]
                euclides2[s] = cdist(XA, XB, 'sqeuclidean')[0][0]
                break

            last_es = np.copy(eigenstates[s])


    if remover_repetidos:
        # alguns estados podem estar repetidos ou estarem fora da
        # região de interesse, por isso vamos removê-los
        sort_index = autovalores.argsort()
        autovalores = autovalores[sort_index]
        eigenstates = eigenstates[sort_index]

        iz_left = derivada_muda_pct(z, V)
        iz_right = len(V)-derivada_muda_pct(z, V[::-1])
        golden_ones = [0]

        for i in range(autovalores.size):
            # remove estados repetidos e nao confinados
            if i == 0 \
                or np.abs(autovalores[i]/autovalores[i-1]-1) < 0.1 \
                or autovalores[i] > np.max(V):
                continue
            # remove os estados nao confinados lateralmente
            state = eigenstates[i].copy()
            state_l = state[:iz_left]
            state_m = state[iz_left:iz_right]
            state_r = state[iz_right:]
            int_left = simps(state_l * state_l.conj(), \
                                z[:iz_left]).real
            int_mid = simps(state_m*state_m.conj(),
                            z[iz_left:iz_right]).real
            int_right = simps(state_r * state_r.conj(), \
                                z[iz_right:]).real
            if int_left+int_right > int_mid:
                continue

            golden_ones.append(i)
    else:
        golden_ones = [i for i in range(len(autovalores))]

    return {
        "autovalores": autovalores[golden_ones],
        "autovetores": autovetores[golden_ones],
        "contadores": contadores[golden_ones],
        "precisoes": precisoes[golden_ones],
        "cronometros": cronometros[golden_ones],
        "euclides2": euclides2[golden_ones]
    }