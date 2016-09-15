# -*- encoding: utf-8 -*-
"""
Programa para calcular la opacidad LTE de hidrógeno

William Henney - 06 Mar 2012

Curso de Astrofísica Estelar - 2012-II - Tarea 03
"""

import numpy as np
from numpy import exp

# 
# Constantes físicos
#
#                                           # Unidades
#                                           # --------   
BOLTZMANN_CGS = 1.3806503e-16               # erg/K
EV_CGS = 1.602176462e-12                    # erg/eV
ANGSTROM_CGS = 1.e-8                        # cm
LIGHTSPEED_CGS = 2.99792458e10              # cm/s
PLANCK_CGS = 6.62606876e-27                 # erg/s
MASA_PROTON = 1.67262158e-24                # g

BOLTZMANN = BOLTZMANN_CGS / EV_CGS          # eV/K
PLANCK = PLANCK_CGS / EV_CGS                # eV/s
RYDBERG = 13.6                              # eV
HMINUS_EION = 0.754                         # eV

# H0 función de partición (se supone constante)
H0_U = 2.0
def H0_level_population(n, T):
    """
    Calcular la población ETL del nivel n de hidrógeno neutro
    a una temperatura T kelvin
    """
    # Energía de excitación respeto a n=1
    E = RYDBERG * (1.0 - 1.0/n**2)
    # Peso estadístico
    g = 2.0*n**2
    return (g/H0_U)*exp(-E/(BOLTZMANN*T))

def Saha_Phi(Eion, Ui, Un, T):
    """
    Función Phi(T) = (Ni Ne / Nn) de Saha 
    para energía de ionización Eion,
    y con funciones de partición Ui y Un
    """
    return 4.8293744908e15 * (Ui/Un) * T**1.5 * exp(-Eion/(BOLTZMANN*T))

def Saha_Phi_H0Hplus(T):
    return Saha_Phi(RYDBERG, H0_U, H0_U, T)

def Saha_Phi_HminusH0(T):
    return Saha_Phi(HMINUS_EION, H0_U, 1.0, T)

@np.vectorize
def Hplus_fraction(Hden, T):
    """
    Calcular fracción de hidrógeno ionizado
    """
    A = Saha_Phi_H0Hplus(T) / Hden
    # Resolver polinomio: y**2 + A*y - A = 0
    y = np.roots([1.0, A, -A])[1] # tomar raiz positivo
    return y

def Hminus_fraction(Hden, T):
    """
    Calcular fracción del ión negativo de hidrógeno
    """ 
    y = Hplus_fraction(Hden, T)
    return y * (1. - y) * Hden/Saha_Phi_HminusH0(T)


"""
Solving ion balance for hydrogen:

n+ + n- + n0 = n
(ne n+ / n0) = Phi1(T)
(ne n0 / n-) = Phi2(T)
ne = n+ - n-

Assume that n- abundance is always negligible

Then ionized fraction y = n+/n obeys the quadratic equation

y**2 + A*y - A = 0

where A = Phi1(T)/n
"""

# Sección eficaz de n=1 de H0 a E = 1 Rydberg (Gray, Eq. 8.4)
# (suponer factor gaunt = 1)
XSEC0 = 7.92609446707e-18       # cm^2/H atom
@np.vectorize
def xsec_H0_boundfree(n, nu):
    """
    Sección eficaz de fotoionización de nivel n de H0 a frecuencia nu Hz

    Multiplicar por densidad de H0(n) para dar coeficiente de absorción (cm^{-1})
    """
    E = PLANCK*nu               # energía de fotón
    E0 = RYDBERG/n**2           # energía de ionización de nivel n

    if E >= E0:
        xsec = XSEC0*n*(E0/E)**3
    else:
        xsec = 0.0

    return xsec

def xsec_H0_freefree(T, nu):
    """
    Sección eficaz por electrón de bremsstrahlung a frecuencia nu Hz

    Multiplicar por Ne N(H+) para dar coeficiente de absorción (cm^{-1})
    """
    return 0.018 * T**-1.5 / nu**2 # Rybicki, eq. 5.19b

@np.vectorize
def xsec_Hminus_boundfree(nu):
    """
    Sección eficaz de fotoionización del ión negativo H- a frecuencia nu Hz

    Multiplicar por N(H-) para dar coeficiente de absorción (cm^{-1})
    """
    wav = LIGHTSPEED_CGS / (nu * 1.e4 * ANGSTROM_CGS) # lambda en unidades de 10,000 Å (1 micron)
    # Fórmula y constantes de Gray, Eq. 8.11
    A = [1.99654, -1.18267e-1, 2.64243e2, -4.40524e2, 3.23992e2, -1.39568e2, 2.78701e1]
    xsec = 0.0
    # El ajuste es preciso para 2250 Å <= lambda <= 15,000 Å 
    # Hay que cortarlo a partir de 16,200 Å porque el ajuste va negativo
    for i, a in enumerate(A):
        if wav <= 1.62:
            xsec += a*wav**i
    return xsec * 1.e-18

@np.vectorize
def xsec_Hminus_freefree(T, nu):
    """
    Opacidad libre-libre del ión negativo H- a frecuencia nu Hz

    Multiplicar por Pe N(H0) para dar coeficiente de absorción (cm^{-1})
    + Ojo que no hay que multiplicar por N(H-)
    + Y esto ya incluye la correción por emisión estimulada
    """
    wav = LIGHTSPEED_CGS / (nu * ANGSTROM_CGS) # lambda en unidades de Å
    # if 2600.0 <= wav <= 113900.0:
    logwav = np.log10(wav)
    # Eq. 8.13 de Gray
    f0 = -2.2763 - 1.6850*logwav + 0.76661*logwav**2 - 0.053346*logwav**3
    f1 = 15.2827 - 9.2846*logwav + 1.99381*logwav**2 - 0.142631*logwav**3
    f2 = -197.789 + 190.266*logwav - 67.9775*logwav**2 + 10.6913*logwav**3 - 0.625151*logwav**4
    theta = np.log10(np.e) / (BOLTZMANN*T) # aproximadamente theta = 5040/T
    xsec = 1.e-26 * 10**(f0 + f1*np.log10(theta) + f2*np.log10(theta)**2)
    # else:
    #     xsec = 0.0
    return xsec


def funcPe(Hden, T):
    """
    Presión electrónica como función de densidad total y temperatura
    """
    return Hden*Hplus_fraction(Hden, T)*BOLTZMANN_CGS*T

def funcHden(Pe, T):
    """
    Densidad total como función de Pe y T

    Esta función busca numericamente el raiz para Hden de la función

    funcPe(Hden, T) - Pe = 0

    empezando con un primer intento que suponga 50% ionización
    """
    from scipy.optimize import fsolve
    Hden0 = 0.5*Pe / (BOLTZMANN_CGS*T) # primer intento es 50% ionizado
    return fsolve(lambda Hden: funcPe(Hden, T) - Pe, Hden0)[0]

NMAX = 20                       # Nivel cuántico de H0 más alto para considerar
def opacidad_total(Hden, T, nu):
    """
    Calcular la opacidad total del continuo de un gas de H puro en ETL

    Parámetros de entrada:
    
    Hden : densidad total de hidrógeno (cm^{-3})
    T    : temperatura (K)
    nu   : frecuencia (Hz)

    Resultado: 

    opacities: dict con coeficiente de absorción por masa (cm^2/g)
               elementos son "Total", "H0bf", "H0ff", "Hmbf", "Hmff"
    """

    y = Hplus_fraction(Hden, T)           # fracción de ionización
    Hpden = y*Hden                        # densidad de H+
    eden = y*Hden                         # densidad de electrones
    Pe = funcPe(Hden, T)                  # presión de electrones
    H0den = (1.0 - y)*Hden                # densidad de H0
    Hmden = Hden*Hminus_fraction(Hden, T) # densidad de H-

    stimulated_correction = (1.0 - np.exp(-PLANCK_CGS*nu / (BOLTZMANN_CGS*T)))
    opacities = dict()
    
    # H0 ligado-libre
    opacities["H0bf"] = 0.0
    for n in range(1, NMAX+1):
        opacities["H0bf"] += H0den * H0_level_population(n, T) * xsec_H0_boundfree(n, nu)
    opacities["H0bf"] *= stimulated_correction
    # H0 libre-libre
    opacities["H0ff"] = Hpden * eden * xsec_H0_freefree(T, nu)
    opacities["H0ff"] *= stimulated_correction
    # H- ligado-libre
    opacities["Hmbf"] = Hmden * xsec_Hminus_boundfree(nu)
    opacities["Hmbf"] *= stimulated_correction 
    # H- libre-libre (que ya incluye emisión estimulada)
    opacities["Hmff"] = H0den * Pe * xsec_Hminus_freefree(T, nu)

    # convertir a opacidad por masa
    total = 0.0
    for k in opacities.keys():
        opacities[k] /= Hden*MASA_PROTON
        total += opacities[k]
    opacities["Total"] = total
    return opacities
