# -*- coding: utf-8 -*-
# neste arquivo apenas listamos as contantes fisicas
# com um nome mais amigável para utilização em outros
# lugares e manter uma certa padronização

# third-party
import scipy.constants as cte

# transformações para unidades atomicas
au_l = cte.value('atomic unit of length')
au_t = cte.value('atomic unit of time')
au_e = cte.value('atomic unit of energy')
au_v = cte.value('atomic unit of electric potential')
au_ef = cte.value('atomic unit of electric field')

# Constantes físicas
me = cte.value('electron mass')
c = cte.value('speed of light in vacuum')
q = cte.value('elementary charge')
hbar_ev = cte.value('Planck constant over 2 pi in eV s')
hbar = hbar_si = cte.value('Planck constant over 2 pi')
h = cte.value('Planck constant')
ev = cte.value('electron volt')

# outras relacoes de interesse
au2ang = au_l / 1e-10
au2ev = au_e / ev
hbar_au = 1.0
me_au = 1.0
