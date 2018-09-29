#!/usr/bin/env python
# -*- coding: utf-8 -*-

# third-party
import numpy as np


def algaas_gap(x:float)->float:
    """
    Retorna o gap do material ja calculado em funcao da fracao de Aluminio
    utilizamos os valores exatos utilizados pelos referidos autores

    Params
    ------
    x : float
        a fracao de aluminio, entre 0 e 1

    Returns
    -------

    O gap em eV
    """
    if x == 0.2:
        return 0.0
    elif x == 0.4:
        return 0.185897
    return -0.185897


def algaas_meff(x:float)->float:
    """Retorna a massa efetiva do AlGaAs em funcao da fracao de Aluminio
    assim como os referidos autores, utilizamos a massa efetiva do 
    eletron no GaAs ao longo de todo o material

    Params
    ------
    x : float
        a fracao de aluminio, entre 0 e 1

    Returns
    -------

    A massa efetiva do eletron no AlGaAs
    """
    return 0.067


def x_shape(z):
    """Utilizamos a concentracao de Aluminio para determinar o perfil do
    potencial

    Params
    ------

    z : float
        posicao no eixo z em angstrom

    Returns
    -------

    A concentracao de Aluminio na posicao informada
    """
    # concentracoes e larguras do sistema
    xd = 0.2  # concentracao no espaco entre poco e barreira
    xb = 0.4  # concentracao na barreira
    xw = 0.0  # concentracao no poco
    wl = 50.0  # largura do poco em angstrom
    bl = 50.0  # largura da barreira em angstrom
    dl = 40.0  # espacao entre poco e barreira em angstrom

    if np.abs(z) < wl / 2:
        return xw
    elif np.abs(z) < wl / 2 + dl:
        return xd
    elif np.abs(z) < wl / 2 + dl + bl:
        return xb
    return xd