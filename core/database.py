# -*- coding: utf-8 -*-

class Alloy(object):
    alc_a = 0.0
    alc_b = 0.0
    eg_g = 0.0
    alpha_g = 0.0
    beta_g = 0.0
    eg_x = 0.0
    alpha_x = 0.0
    beta_x = 0.0
    eg_l = 0.0
    alpha_l = 0.0
    beta_l = 0.0
    delta_so = 0.0
    me_g = 0.0
    ml_l = 0.0
    mt_l = 0.0
    mdos_l = 0.0
    ml_x = 0.0
    mt_x = 0.0
    mdos_x = 0.0
    gamma_1 = 0.0
    gamma_2 = 0.0
    gamma_3 = 0.0
    mso = 0.0
    ep = 0.0
    f = 0.0
    vbo = 0.0
    ac = 0.0
    av = 0.0
    b = 0.0
    d = 0.0
    c11 = 0.0
    c12 = 0.0
    c44 = 0.0

    def gap_g(self, T=0.0):
        return self.eg_g - self.alpha_g * T ** 2 / (T + self.beta_g)

    def alc(self, T=0.0):
        return self.alc_a + self.alc_b * (T - 300.0)

    @property
    def a(self):
        return self.ac + self.av

    @property
    def mhh_z(self):
        return 1.0 / (self.gamma_1 - 2.0 * self.gamma_2)

    @property
    def mhh_110(self):
        gamma = 2.0 * self.gamma_1 - self.gamma_2 - 3.0 * self.gamma_3
        return 2.0 / gamma

    @property
    def mhh_111(self):
        gamma = self.gamma_1 - 2.0 * self.gamma_3
        return 1.0 / gamma


class GaAs(Alloy):
    def __init__(self):
        self.alc_a = 5.65325
        self.alc_b = 3.88e-5
        self.eg_g = 1.519
        self.alpha_g = 0.0005405
        self.beta_g = 204.0
        self.eg_x = 1.981
        self.alpha_x = 0.00046
        self.beta_x = 204.0
        self.eg_l = 0.0
        self.alpha_l = 0.0
        self.beta_l = 0.0
        self.delta_so = 0.341
        self.me_g = 0.067
        self.ml_l = 1.9
        self.mt_l = 0.0754
        self.mdos_l = 0.56
        self.ml_x = 1.3
        self.mt_x = 0.23
        self.mdos_x = 0.85
        self.gamma_1 = 6.98
        self.gamma_2 = 2.06
        self.gamma_3 = 2.93
        self.mso = 0.172
        self.ep = 28.8
        self.f = -1.94
        self.vbo = -0.8
        self.ac = -7.17
        self.av = -1.16
        self.b = -2.0
        self.d = -4.8
        self.c11 = 1221.0
        self.c12 = 566.0
        self.c44 = 600.0


class AlAs(Alloy):
    def __init__(self):
        self.alc_a = 5.6611
        self.alc_b = 2.9e-5
        self.eg_g = 3.099
        self.alpha_g = 0.000885
        self.beta_g = 530.0
        self.eg_x = 2.24
        self.alpha_x = 0.0007
        self.beta_x = 530.0
        self.eg_l = 0.0
        self.alpha_l = 0.0
        self.beta_l = 0.0
        self.delta_so = 0.28
        self.me_g = 0.15
        self.ml_l = 1.32
        self.mt_l = 0.15
        self.mdos_l = 0.0
        self.ml_x = 0.97
        self.mt_x = 0.22
        self.mdos_x = 0.0
        self.gamma_1 = 3.76
        self.gamma_2 = 0.82
        self.gamma_3 = 1.42
        self.mso = 0.28
        self.ep = 21.1
        self.f = -0.48
        self.vbo = -1.33
        self.ac = -5.64
        self.av = -2.47
        self.b = -2.3
        self.d = -3.4
        self.c11 = 1250.0
        self.c12 = 534.0
        self.c44 = 542.0


class InAs(Alloy):
    def __init__(self):
        self.alc_a = 6.0583
        self.alc_b = 2.74e-5
        self.eg_g = 0.417
        self.alpha_g = 0.000276
        self.beta_g = 93.0
        self.eg_x = 1.433
        self.alpha_x = 0.000276
        self.beta_x = 93.0
        self.eg_l = 1.133
        self.alpha_l = 0.000276
        self.beta_l = 93.0
        self.delta_so = 0.39
        self.me_g = 0.026
        self.ml_l = 0.64
        self.mt_l = 0.05
        self.mdos_l = 0.29
        self.ml_x = 1.13
        self.mt_x = 0.16
        self.mdos_x = 0.64
        self.gamma_1 = 20.0
        self.gamma_2 = 8.5
        self.gamma_3 = 9.2
        self.mso = 0.14
        self.ep = 21.5
        self.f = -2.9
        self.vbo = -0.59
        self.ac = -5.08
        self.av = -1.0
        self.b = -1.8
        self.d = -3.6
        self.c11 = 832.9
        self.c12 = 452.6
        self.c44 = 395.9

class InGaAs(Alloy):
    def __init__(self, x):
        self.c_eg_g = 0.477
        self.c_eg_x = 1.4
        self.c_eg_l = 0.33
        self.c_delta_so = 0.15
        self.c_me_g = 0.0091
        self.c_mhh_001 = -0.145
        self.c_mlh_001 = 0.0202
        self.c_ep = -1.48
        self.c_f = 1.77
        self.c_vbo = -0.38
        self.c_ac = 2.61

        self.alloy_1 = GaAs()
        self.alloy_2 = InAs()

        self.alc_a = self.alloy_1.alc_a * (1.0 - x) + self.alloy_2.alc_a * x
        self.alc_b = self.alloy_1.alc_b * (1.0 - x) + self.alloy_2.alc_b * x
        self.eg_g = self.alloy_1.eg_g * (1.0 - x) + self.alloy_2.eg_g * x - x * (1.0 - x) * self.c_eg_g
        self.alpha_g = self.alloy_1.alpha_g * (1.0 - x) + self.alloy_2.alpha_g * x
        self.beta_g = self.alloy_1.beta_g * (1.0 - x) + self.alloy_2.beta_g * x
        self.eg_x = self.alloy_1.eg_x * (1.0 - x) + self.alloy_2.eg_x * x - x * (1.0 - x) * self.c_eg_x
        self.alpha_x = self.alloy_1.alpha_x * (1.0 - x) + self.alloy_2.alpha_x * x
        self.beta_x = self.alloy_1.beta_x * (1.0 - x) + self.alloy_2.beta_x * x
        self.eg_l = self.alloy_1.eg_l * (1.0 - x) + self.alloy_2.eg_l * x - x * (1.0 - x) * self.c_eg_l
        self.alpha_l = self.alloy_1.alpha_l * (1.0 - x) + self.alloy_2.alpha_l * x
        self.beta_l = self.alloy_1.beta_l * (1.0 - x) + self.alloy_2.beta_l * x
        self.delta_so = self.alloy_1.delta_so * (1.0 - x) + self.alloy_2.delta_so * x
        self.me_g = self.alloy_1.me_g * (1.0 - x) + self.alloy_2.me_g * x - x * (1.0 - x) * self.c_me_g
        self.ml_l = self.alloy_1.ml_l * (1.0 - x) + self.alloy_2.ml_l * x
        self.mt_l = self.alloy_1.mt_l * (1.0 - x) + self.alloy_2.mt_l * x
        self.mdos_l = self.alloy_1.mdos_l * (1.0 - x) + self.alloy_2.mdos_l * x
        self.ml_x = self.alloy_1.ml_x * (1.0 - x) + self.alloy_2.ml_x * x
        self.mt_x = self.alloy_1.mt_x * (1.0 - x) + self.alloy_2.mt_x * x
        self.mdos_x = self.alloy_1.mdos_x * (1.0 - x) + self.alloy_2.mdos_x * x
        self.gamma_1 = self.alloy_1.gamma_1 * (1.0 - x) + self.alloy_2.gamma_1 * x
        self.gamma_2 = self.alloy_1.gamma_2 * (1.0 - x) + self.alloy_2.gamma_2 * x
        self.gamma_3 = self.alloy_1.gamma_3 * (1.0 - x) + self.alloy_2.gamma_3 * x
        self.mso = self.alloy_1.mso * (1.0 - x) + self.alloy_2.mso * x
        self.ep = self.alloy_1.ep * (1.0 - x) + self.alloy_2.ep * x - x * (1.0 - x) * self.c_ep
        self.f = self.alloy_1.f * (1.0 - x) + self.alloy_2.f * x - x * (1.0 - x) * self.c_f
        self.vbo = self.alloy_1.vbo * (1.0 - x) + self.alloy_2.vbo * x - x * (1.0 - x) * self.c_vbo
        self.ac = self.alloy_1.ac * (1.0 - x) + self.alloy_2.ac * x - x * (1.0 - x) * self.c_ac
        self.av = self.alloy_1.av * (1.0 - x) + self.alloy_2.av * x
        self.b = self.alloy_1.b * (1.0 - x) + self.alloy_2.b * x
        self.d = self.alloy_1.d * (1.0 - x) + self.alloy_2.d * x
        self.c11 = self.alloy_1.c11 * (1.0 - x) + self.alloy_2.c11 * x
        self.c12 = self.alloy_1.c12 * (1.0 - x) + self.alloy_2.c12 * x
        self.c44 = self.alloy_1.c44 * (1.0 - x) + self.alloy_2.c44 * x
