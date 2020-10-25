# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.3
#   kernel_info:
#     name: python3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Astrofísica Estelar - Tarea 04 
#
# La meta de esta tarea es reproducir las figuras 8.5(a, b, c, d) del libro de Gray. Estas figuras muestran el coeficiente de absorción continua $\kappa_\lambda$ para para diferentes temperaturas $T$ y presiones electrónicas $P_{e}$ en equilibrio termodinámico local (ETL).
#
# Por simplicidad, consideramos solo hidrógeno, en forma de átomo neutro y los iones positivos y negativos. Fracciones de ionización y excitación de los niveles ligados se calculan con las ecuaciones de Saha y de Boltzmann.
#
# - [Nota importante](#org6020450)
# - [Plan general](#orgc561c97)
# - [Constantes físicos](#org01d7ad1)
# - [Balance de ionización de hidrógeno](#orga12a6f4)
#   - [La ecuación de Saha general](#orge0650df)
#     - [Ejercicio 1: Evaluar el constante en la ecuación Saha](#org146d3c0)
#     - [Ejercicio 2: Escribir una función para evaluar $\Phi_j(T)$](#orgcccbee6)
#   - [La abundancia de H⁺](#orgc887a85)
#     - [Ejercicio 3: Encontrar la fracción de ionización para H⁺](#orgac0b637)
#   - [La abundancia de H⁻](#org8bb3534)
#     - [Ejercicio 4:  Encontrar la fracción de ionización para H⁻](#org4dd8ec0)
#   - [Tabla y gráficas de las fracciones de ionización](#org59a416b)
#     - [Ejercicio 5: Graficar la fracción de H⁺ contra temperatura](#org5507a52)
#     - [Ejercicio 6: Graficar la fracción de H⁻ contra temperatura](#orga2b0907)
# - [Excitación de los niveles ligados de H⁰](#orga370889)
#   - [Ejercicio 7: calcular la población del nivel $n$](#org2b7637a)
# - [Secciones rectas en función de longitud de onda](#org25c3f89)
#   - [Fotoionización de hidrógeno neutro H⁰](#org92461cb)
#     - [Ejercicio 8: Escribe una función para calcular la sección eficaz ligado-libre de H⁰](#orgcd12f0f)
#     - [Ejercicio 9: Grafique los resultados contra λ para $n \le 5$](#org17c0d4e)
#   - [Libre-libre H⁺ + e⁻](#orgfc0ad95)
#     - [Ejercicio 10: Calcule el valor numérico de $\alpha_0$](#org9b28e05)
#     - [Ejercicio 11: Escribe funciones para calcular la sección eficaz libre-libre](#org43f09b2)
#   - [Ion negativo de hidrógeno H⁻](#org997652e)
#     - [Ejercicio 12: ligado-libre de H⁻](#org24ba48b)
#     - [Ejercicio 12: libre-libre de H⁰ + e⁻](#org7622dfc)
# - [Relación entre densidad total de hidrógeno $N_\mathrm{H}$ y presión de electrones $P_e$](#org5453896)
#   - [Ejercicio 13: Graficar $P_e(T)$ para diferentes densidades](#org0e8231a)
#   - [Solución implícita para invertir la relación](#org769e5eb)
#   - [Ejercicio 14: Graficar $N_\mathrm{H}(P_e, T)$ para diferentes presiones electrónicas](#orgb46c79b)
# - [Opacidad total en función de longitud de onda](#orgc5b161c)
#   - [Opacidad por unidad de masa](#org7e396f1)
#     - [Ejercico 15: Probar la función `opacidad_total()`](#orga37529e)
#   - [Figuras 8.5(a)-(d) de Gray](#org77c35cf)
#     - [Ejercicio 16: Reproducción de la figura 8.5](#org45d3515)

# <a id="org6020450"></a>
#
# # Nota importante
#
# Hay unos errores en el libro, que están corregidos en [esta página](http://astro.uwo.ca/~dfgray/Photospheres.html). La única corrección importante para esta tarea es a la figura 8.5(a).

# <a id="orgc561c97"></a>
#
# # Plan general
#
# Las cuatro contribuciones a la opacidad que hay que calcular (anotadas con las ecuaciones relevantes de Gray) son:
#
# 1.  Ligado-libre de H⁺
#     -   ec. (8.8) - *pero ver abajo*
# 2.  Libre-libre de H⁺ + e⁻
#     -   ec. (8.10)
# 3.  Ligado-libre de H⁻
#     -   ecs. (8.11) y (8.12)
# 4.  Libre-libre de H⁰ + e⁻
#     -   ec. (8.13)
#
# Noten que Gray defina la $\kappa$ como la absorción por átomo de hidrógeno neutro. Para convertir esta en la opacidad por masa, hay que multiplicar por la fracción de H<sup>0</sup> y dividir por la masa del hidrógeno: $m_{\mathrm{H}} = 1.67 \times 10^{{-24}}$ g. Ver su ecuación 8.18.
#
# Para las contribuciones (1) y (2), las ecuaciones (8.8) y (8.10) de Gray mezclan la contribuciones de la ionización, la excitación, y las secciones rectas. Yo creo que es más ilustrativo calcular estos partes por separado. Es un poco más de trabajo, pero tiene la ventaja que podemos hacer pruebas y gráficas de las diferentes partes.

# <a id="org01d7ad1"></a>
#
# # Constantes físicos
#
# Trabajamos con las energías en electron-volts, entonces queremos que todos nuestros constantes sean consistentes con esto. Aquí hay una manera de obtener eso

# +
import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u
from astropy.constants import k_B, h, m_p, a0, m_e
from astropy.constants import c as light_speed

BOLTZMANN = k_B.to(u.eV/u.K).value
PLANCK = h.to(u.eV*u.s).value
RYDBERG = (1.0*u.Ry).to(u.eV).value
HMINUS_EION = (0.754*u.eV).value
BOHR_RADIUS = a0.cgs.value

print('BOLTZMANN =', BOLTZMANN, 'eV/K')
print('PLANCK =', PLANCK, 'eV.s')
print('RYDBERG =', RYDBERG, 'eV')
print('HMINUS_EION =', HMINUS_EION, 'eV')
print('BOHR_RADIUS =', BOHR_RADIUS, 'cm')
# -

# A mi me da el resultado:
#
#     BOLTZMANN = 8.617330337217213e-05 eV/K
#     PLANCK = 4.1356676623401646e-15 eV.s
#     RYDBERG = 13.60569300965081 eV
#     HMINUS_EION = 0.754 eV
#     BOHR_RADIUS = 5.2917721067e-09 cm
#
#
# <a id="orga12a6f4"></a>
#
# # Balance de ionización de hidrógeno
#
# Las diferentes contribuciones a la opacidad son proporcionales a las abundancias de diferentes especies. Por ejemplo, ligado-libre es proporcional a H⁰, mientras libre-libre es proporcional a H⁺. Por lo tanto, tenemos que conocer el grado de ionización.
#
# Entonces, vamos a resolver la ecuación Saha, de manera similar a la Tarea~01. Una diferencia es que vamos a suponer que conocemos la densidad total de H, y que la densidad electrónica (o presión electrónica) es algo que hay que encontrar.

# <a id="orge0650df"></a>
#
# ## La ecuación de Saha general
#
# Usaremos la siguiente versión de la ecuación de Saha: $$ \frac{N_{j+1} N_e}{N_j} = \Phi_j(T), $$ que relaciona las densidades de las etapas de ionización adyacentes: $j$ y $j+1$.
#
# La función en el lado derecho es $$ \Phi_j(T) = 2 \left( \frac{2\pi m_e k T}{h^2} \right)^{1.5} \frac{U_{j+1}}{U_j} e^{-E_j/k T}, $$ donde $E_j$ es la $j$-ésima potencia de ionización y $U_j$, $U_{j+1}$ son las funciones de partición.

# <a id="org146d3c0"></a>
#
# ### Ejercicio 1: Evaluar el constante en la ecuación Saha
#
# Usar los constantes `k_B`, `m_e`, `h` (ver arriba) y `np.pi` para evaluar el constante $2 (2\pi m_e k / h^2)^{3/2}$ en sistema cgs:

saha_C = ????
saha_C.cgs

# Note que unos libros (por ejemplo, Hubeny y Mihalas) usan un constante que es el recíproco de esto.

# <a id="orgcccbee6"></a>
#
# ### Ejercicio 2: Escribir una función para evaluar $\Phi_j(T)$
#
# Usamos valores de defecto para la potencia de ionización `Eion` y las funciones de partición, `Un` y `Ui`, que son apropiadas para H⁰ (a bajas temperaturas se puede aproximar $U_j = 2$, $U_{j+1} = 1$). Hay que usar el constante `BOLTZMANN` en eV/K que definimos antes.

# +
SAHA_CONSTANT = saha_C.cgs.value

def Saha_Phi(T, Eion=1.0*RYDBERG, Ui=1.0, Un=2.0):
    """
    Función Phi(T) = (Ni Ne / Nn) de Saha para energía de ionización Eion (en eV),
    y con funciones de partición Ui y Un.  La T es en kelvin
    """
    return SAHA_CONSTANT * ?????


# -

# Luego, probamos la función con unas temperaturas típicas:

Ts = np.array([3, 5, 9, 15])*u.kK
Ts.cgs

# Llamamos la función así:

Saha_Phi(Ts.cgs.value)


# Note que el argumento `T` a `Saha_Phi` es un número normal (`float`).

# <a id="orgc887a85"></a>
#
# ## La abundancia de H⁺
#
# Suponemos que la abundancia de H⁻ es siempre una fracción despreciable, entonces tenemos números iguales de electrones y electrones libres: $N_+ = N_e$.

# <a id="orgac0b637"></a>
#
# ### Ejercicio 3: Encontrar la fracción de ionización para H⁺
#
# Muestre que la fracción de H⁺, $y = N_+ / N_H$, es la solución de la ecuación cuadrática $y^2 + A y - A = 0$, donde $A = \Phi_{H_0} / N_H$.
#
# Una función para evaluar esta fracción en función de la densidad de H y la temperatura podría ser la siguiente:

@np.vectorize
def Hplus_fraction(Hden, T):
    """
    Calcular fracción de hidrógeno ionizado

    `Hden` es densidad de partículas totales de H en cm^{-3}
    `T` es temperatura en K
    """
    A = Saha_Phi(T) / Hden
    # Resolver polinomio: y**2 + A*y - A = 0
    y = np.roots([1.0, A, -A]).max() # tomar raiz positivo
    return y


# Notas:
#
# 1.  Usamos la función `np.roots()` que encuentra todos los raíces de un polinomio. Esta devuelva dos raíces, entonces usamos `.max()` para seleccionar la positiva.
# 2.  Usamos el decorador `@np.vectorize` para que se puede aplica la función a vectores de densidad y temperatura. Es necesario aquí porque `np.roots()` es para un polinomio a la vez.

# <a id="org8bb3534"></a>
#
# ## La abundancia de H⁻
#
# La ecuación de Saha para H⁻ es $$ \frac{N_{H^0} N_e}{N_{H^-}} = \Phi_{H^-}(T), $$

# <a id="org4dd8ec0"></a>
#
# ### Ejercicio 4:  Encontrar la fracción de ionización para H⁻
#
# Muestre que $$ N_{H^-} \big/ N_H = (1 - y) y N_H \big/ \Phi_{H^-} $$ y entonces escribe una función para calcular esta fracción:

def Hminus_fraction(Hden, T):
    """
    Calcular fracción del ión negativo de hidrógeno
    """ 
    y = Hplus_fraction(Hden, T)
    return ????


# En esta se puede usar la función `Saha_Phi`, pero cambiando los argumentos opcionales: `Saha_Phi(T, Eion=????, Ui=????, Un=????)`

# <a id="org59a416b"></a>
#
# ## Tabla y gráficas de las fracciones de ionización
#
# Podemos definir unas densidades típicas para atmósferas, y luego escribir una tabla de las fracciones:

Ns = np.array([10, 3, 1, 0.5])*1e15/u.cm**3
Ns

# Usamos la librería `astropy.table` para construir la tabla para que salga con formato bonito en el notebook:

from astropy.table import Table
Table(
  data=[
    Column(Ns.cgs, name=r'$N_H$'),
    Column(Ts.cgs, name=r'$T$'),
    Column(Hplus_fraction(Ns.cgs.value, Ts.cgs.value), name=r'$N_+/N_H$'),
    Column(Hminus_fraction(Ns.cgs.value, Ts.cgs.value), name=r'$N_-/N_H$'),
  ])

# Note que la fracción de H⁻ es siempre muy pequeño, lo cual justifica despreciarla cuando estimamos la densidad electrónica.

# <a id="org5507a52"></a>
#
# ### Ejercicio 5: Graficar la fracción de H⁺ contra temperatura
#
# Definimos una serie de densidades: $\{10^4, 10^6, \dots, 10^{18}, 10^{20}\}$ y un rango de temperaturas entre 2000 y 20,000 K:

logNgrid = range(4, 20, 2)
Tgrid = np.linspace(2e3, 2e4, 500)


# Grafique la $y$ en función de la $T$ para cada densidad (una línea para cada valor de `logNgrid`). Use la función `Hplus_fraction` de arriba. Para las densidades típicas de fotósferas, ¿en que rango de temperatura ocurre la transición entre hidrógeno neutro e ionizado?

# <a id="orga2b0907"></a>
#
# ### Ejercicio 6: Graficar la fracción de H⁻ contra temperatura
#
# Repite la gráfica pero para el ion negativo. Esta vez, use una escala logarítmica en el eje vertical (una manera de hacerlo es usar `ax.set_yscale("log")`). Se debe encontrar que la fracción $N_{H^-} / N_H$ tiene un pico en una temperatura intermedia. Comparando con la gráfica anterior ¿a qué valor de $y$ corresponde este pico? ¿por qué?

# <a id="orga370889"></a>
#
# # Excitación de los niveles ligados de H⁰
#
# Utilizamos la ecuación de Boltzmann para calcular la población fraccional de un nivel ligado dado, $n$, de hidrógeno neutro: $$ \frac{N_n}{N_{H^0}} = \frac{g_n}{U(T)} e^{-E_n/k T} $$ donde la degenerancia es $g_n = 2 n^2$ y la energía arriba del nivel base ($n=1$) es $E_n = 1 - n^{-2}$ Rydberg.

# <a id="org2b7637a"></a>
#
# ## Ejercicio 7: calcular la población del nivel $n$
#
# Se puede suponer que la función de partición es 2. Complete la función siguiente:

def H0_level_population(n, T, U=2.0):
    """
    Calcular la población ETL del nivel n de hidrógeno neutro
    a una temperatura T kelvin
    """
    # Energía de excitación respeto a n=1
    E = ????
    # Peso estadístico
    g = ????
    return ????


# Esta función será usada más adelante.

# <a id="org25c3f89"></a>
#
# # Secciones rectas en función de longitud de onda

# <a id="org92461cb"></a>
#
# ## Fotoionización de hidrógeno neutro H⁰
#
# Para la fotoionización del nivel $n$, hay una energía umbral, $E_n = n^{-2}\ \mathrm{Ry}$, con una frecuencia mínima asociada, $\nu_n = E_n/h$, o longitud de onda máxima, $\lambda_n = h c / E_n$. La sección eficaz se da por $$ \sigma_\mathrm{bf}(n, \nu) = \sigma_0 n \frac{\nu_n^3}{\nu^3} g_\mathrm{bf}(n, \nu) $$ donde $\sigma_0 = 2.815\times 10^{29} \nu_1^{-3} = 7.906 \times 10^{-18}\ \mathrm{cm}^2$ y $g_\mathrm{bf}(n, \nu)$ es el factor Gaunt que corrige por efectos de la mecánica cuántica.

# <a id="orgcd12f0f"></a>
#
# ### Ejercicio 8: Escribe una función para calcular la sección eficaz ligado-libre de H⁰
#
# Se puede seguir un ejemplo como esto:

@np.vectorize
def xsec_H0_boundfree(n, nu, xsec0=7.906e-18):
    """
    Sección eficaz de fotoionización de nivel n de H0 a frecuencia nu Hz

    Multiplicar por la fracción de H0(n) para dar cm^2 por átomo H^0
    """
    E = PLANCK*nu               # energía de fotón
    E0 = RYDBERG/n**2           # energía de ionización de nivel n

    if E >= E0:
        xsec = ???? # ESCRIBIR ALGO AQUÍ
    else:
        xsec = 0.0

    return xsec


# Para evaluar el factor Gaunt, se puede usar la aproximación de Menzel y Perkis que se da en ec.~(8.5): $$ g_\mathrm{bf}(n, \nu) = 1 - \frac{0.3456}{(\lambda R)^{1/3}} \left( \frac{\lambda R}{n^2} - \frac{1}{2}\right) . $$ Se sugiere escribir una función para esto, por ejemplo:

def gaunt_H0_boundfree(n, nu):
  """
  Factor Gaunt para fotoionización de nivel `n` de H0 a frecuencia `nu` Hz
  """
  lambda_R = RYDBERG/(PLANCK*nu)
  return ????? # ESCRIBIR ALGO AQUÍ


# <a id="org17c0d4e"></a>
#
# ### Ejercicio 9: Grafique los resultados contra λ para $n \le 5$
#
# Primero, se puede definir un arreglo de $\lambda$ (`wavs`) un arreglo equivalente en frecuencia (`freqs`).

wavs = np.linspace(40.0, 20000.0, 500)*u.AA
freqs = (light_speed/wavs).cgs.value

# Por ejemplo, para graficar los factores Gaunt se puede hacer algo así:

fig, ax = plt.subplots(1, 1)
for n in range(1, 5):
  # Restringir a las frecuencias capaces de ionizar cada nivel
  m = h*freqs >= 1.0*u.Ry/n**2
  ax.plot(wavs[m], gaunt_H0_boundfree(n, freqs[m]), 
         label=r'$n = {}$'.format(n))
ax.legend()
ax.set_ylim(0.0, None)
ax.set_xlabel(r'Wavelength, Å')
ax.set_ylabel(r'$g_\mathrm{bf}(n, \nu)$')
ax.set_title('Bound-free gaunt factors');

# El factor de Gaunt debe ser de orden 1, con tendencia de aumentarse un poco en la ultravioleta.
#
# Luego repite la gráfica pero para las secciones rectas. Cheque que puedes reproducir la figura 8.2 de Gray.

# <a id="orgfc0ad95"></a>
#
# ## Libre-libre H⁺ + e⁻
#
# La sección recta por electrón (ver por ejemplo Rybicki sec 5.3) se escribe como $$ \alpha_\mathrm{ff} = \alpha_0 \frac{g_\mathrm{ff}(T, \nu)}{\nu^3 T^{1/2}} \quad \mathrm{cm^2\ \big/\ e^-}, $$ donde $$ \alpha_0 = \frac{4 e^6}{3 m h c} \left(\frac{2\pi}{3 k m}\right)^{1/2} , $$ y el factor Gaunt libre-libre se puede aproximar (Gray, Eq. 8.6) como $$ g_\mathrm{ff}(T, \nu) = 1 - \frac{0.3456}{(\lambda R)^{1/3}} \left( \frac{k T}{h \nu} + \frac{1}{2}\right) . $$

# <a id="org9b28e05"></a>
#
# ### Ejercicio 10: Calcule el valor numérico de $\alpha_0$
#
# Busque en `astropy.constants` para los constantes físicos que se necesitan. Note que para la carga electrónica `e`, hay que especificar un sistema específico para sacar el valor en cgs. Se debe usar el sistema electrostático (`e.esu` en `astropy.constants`).

from astropy.constants import ????
alpha0 = ????
alpha0.cgs

# El resultado debe ser aproximadamente $3.7 \times 10^8$.

# <a id="org43f09b2"></a>
#
# ### Ejercicio 11: Escribe funciones para calcular la sección eficaz libre-libre
#
# Escribe una función `gaunt_H0_freefree(T, nu)` y úsela en otra función `xsec_H0_freefree(T, nu)` para calcular la sección eficaz por electrón.
#
# Se puede usar algo como lo siguiente para graficarlas:

fig, ax = plt.subplots(1, 1)
for T in [5e3, 1e4, 2e4]:
  ax.plot(wavs, gaunt_H0_freefree(T, freqs), 
         label=r'$T = {:.0f}$ K'.format(T))
ax.set_ylim(0.0, None)
ax.legend(loc='lower left')
ax.set_xlabel(r'Wavelength, Å')
ax.set_ylabel(r'$g_\mathrm{ff}(T, \nu)$')
ax.set_title('H$^0$ free-free gaunt factors');

# Los factores Gaunt deben de caer entre aproximadamente 0.5 y 0.9.

fig, ax = plt.subplots(1, 1)
for T in [5e3, 1e4, 2e4]:
  ax.plot(wavs, xsec_H0_freefree(T, freqs), 
         label=r'$T = {:.0f}$ K'.format(T))
ax.set_ylim(0.0, None)
ax.legend(loc='upper left')
ax.set_xlabel(r'Wavelength, Å')
ax.set_ylabel(r'$\alpha_\mathrm{ff}(T, \nu) / N_e N_{H^+}$, cm$^5$')
ax.set_title('Free-free H$^0$ cross sections');


# Debe de subir con λ, alcanzando un poco más que $10^{-36}$ para $T = 5000$ K.

# <a id="org997652e"></a>
#
# ## Ion negativo de hidrógeno H⁻
#
# Las páginas 154 a 156 de Gray proporcionan ajustes a la sección recta ligado-libre y libre-libre asociada a H⁻.

# <a id="org24ba48b"></a>
#
# ### Ejercicio 12: ligado-libre de H⁻
#
# Escribe una función `xsec_Hminus_boundfree(nu)` y úsela para reproducir la figura 8.3 de Gray (línea continua). Queremos una función de $\nu$ por consistencia con las demás, pero se tendrá que convertir a $\lambda$ dentro de la función para poder usar la ec. 8.11. Note que la figura tiene escala logarítmica para el eje $y$.

# <a id="org7622dfc"></a>
#
# ### Ejercicio 12: libre-libre de H⁰ + e⁻
#
# Escribe una función `xsec_Hminus_freefree(T, nu)` y úsela para reproducir la figura 8.4 de Gray. Note que esta sección recta es por átomo de H⁰ y por unidad de $P_e$. Note que la figura tiene escala logarítmica para ambos ejes, $x$ y $y$. También el rango de $\lambda$ es más amplia que en las gráficas anteriores.

# <a id="org5453896"></a>
#
# # Relación entre densidad total de hidrógeno $N_\mathrm{H}$ y presión de electrones $P_e$
#
# Las figuras de Gray que esperamos reproducir son para valores fijos de $T$ y $P_e$. Sin embargo, nuestras funciones para H⁰ están en función de la densidad. Entonces, necesitamos funciones para traducir entre los dos. Note que habría sido más fácil (aunque menos divertido &#x2026;) escribir todo en función de $P_e$ desde el principio.
#
# Pasar de densidad de hidrógeno a la presión de electrones es fácil: $P_e = N_e k T$, donde $N_e = y N_\mathrm{H}$.

def funcPe(Hden, T):
    """
    Presión electrónica como función de densidad total y temperatura
    """
    return Hden*Hplus_fraction(Hden, T)*k_B.cgs.value*T


# <a id="org0e8231a"></a>
#
# ## Ejercicio 13: Graficar $P_e(T)$ para diferentes densidades
#
# Use cuatro densidades: $\{10^{12}, 10^{14}, 10^{16}, 10^{18}\}$ y la misma `Tgrid` que antes. Use una escala logarítmica en el eje $y$. A temperaturas altas, $P_e$ crece linealmente con $T$, que se ve plano en la figura. A temperaturas bajas, las curvas son mucho más empinadas. ¿Por qué?

# <a id="org769e5eb"></a>
#
# ## Solución implícita para invertir la relación
#
# Pasar en la otra dirección, $N_\mathrm{H}(P_e, T)$, es más complicada. Una solución es buscar (para una `Pe` y `T` dada) el valor de `Hden` que es raiz de la ecuación: `funcPe(Hden, T) - Pe = 0`. Aquí hay una posible implementación de esta idea:

@np.vectorize
def funcHden(Pe, T):
    """
    Densidad total como función de Pe y T

    Esta función busca numericamente el raiz para Hden de la función

    funcPe(Hden, T) - Pe = 0

    empezando con un primer intento que suponga 50% ionización
    """
    from scipy.optimize import fsolve
    Hden0 = 0.5*Pe / (k_B.cgs.value*T) # primer intento es 50% ionizado
    return fsolve(lambda Hden: funcPe(Hden, T) - Pe, Hden0)[0]


# Se usa la función `fsolve` de la librería `scipy.optimize`, que usa iteración a partir de una estimación inicial.

# <a id="orgb46c79b"></a>
#
# ## Ejercicio 14: Graficar $N_\mathrm{H}(P_e, T)$ para diferentes presiones electrónicas
#
# Use cuatro valores: $P_e = \{1, 10, 100, 1000\}$ electrones/cm$^-3$ y un rango de temperatura que empieza en 4500 K. No se puede encontrar una solución para temperaturas muy bajas si la $P_e$ es alta. ¿por qué?

# <a id="orgc5b161c"></a>
#
# # Opacidad total en función de longitud de onda
#
# Ya tenemos todas las piezas necesarias para reproducir la Fig 5.8.

# <a id="org7e396f1"></a>
#
# ## Opacidad por unidad de masa
#
# Primero, escribimos una función que calcula la opacidad total por unidad de masa porque esta es una cantidad estándar.

def opacidad_total(Pe, T, wavs):
    """
    Calcular la opacidad total del continuo de un gas de H puro en ETL

    Parámetros de entrada:

    Pe   : presión de electrones (dyne cm^{-2}) 
    T    : temperatura (K)
    wavs : arreglo de longitudes de onda (Å)

    Resultado: 

    opacities: dict con coeficiente de absorción por masa (cm^2/g)
               elementos son "Total", "H0bf", "H0ff", "Hmbf", "Hmff"
    """

    Hden = funcHden(Pe, T)                # densidad total de H
    y = Hplus_fraction_U(Hden, T)           # fracción de ionización
    Hpden = y*Hden                        # densidad de H+
    eden = y*Hden                         # densidad de electrones
    H0den = (1.0 - y)*Hden                # densidad de H0
    Hmden = Hden*Hminus_fraction(Hden, T) # densidad de H-

    # frequencies are pure numbers in Hz
    nu = (light_speed/(wavs*u.AA)).cgs.value
    stimulated_correction = (1.0 - np.exp(-h.cgs.value*nu / (k_B.cgs.value*T)))
    opacities = {}

    # H0 ligado-libre
    opacities["H0bf"] = 0.0
    nmax = int(nmax_pressure_ionization(Hden))
    Un = H0_partition_function(T, nmax)
    for n in range(1, nmax+1):
        opacities["H0bf"] += H0den * H0_level_population(n, T, Un) * xsec_H0_boundfree(n, nu)
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
      m = opacities[k] < 0.0
      opacities[k][m] = 0.0
      opacities[k] /= H0den*m_p.cgs.value
      total += opacities[k]
    opacities["Total"] = total
    # guardar metadata
    opacities["metadata"] = {'N_H': Hden, 'y_H': y}

    return opacities


# <a id="orga37529e"></a>
#
# ### Ejercico 15: Probar la función `opacidad_total()`
#
# Utilice la función con diferentes valores de `Pe` y `T`.

opacidad_total(Pe=10.0, T=1e4, wavs=[3000, 10000])

# Asegúrese que entiende los resultados.

# <a id="org77c35cf"></a>
#
# ## Figuras 8.5(a)-(d) de Gray
#
# Aquí hay una función para producir una figura en las unidades que se usan en el libro (sección eficaz por H⁰ por $P_e$).

styles = {
  'Total': {'color': 'k', 'ls': '-'},
  'H0bf': {'color': 'r', 'ls': '-'},
  'H0ff': {'color': 'r', 'ls': '--'},
  'Hmbf': {'color': 'g', 'ls': '-'},
  'Hmff': {'color': 'g', 'ls': '--'},
}
def plot_opacities(Pe, T, wavrange=[3000., 20000.], yscale='linear'):
  wavs = np.linspace(wavrange[0], wavrange[1], 500)
  fig, ax = plt.subplots(1, 1)
  opac = opacidad_total(Pe, T, wavs)
  data = opac.pop('metadata')
  for kwd in opac.keys():
    ax.plot(wavs, opac[kwd]*m_p.cgs.value/Pe/1e-26, label=kwd, **styles[kwd])
  frame = ax.legend(loc='upper right', **legend_box_params).get_frame()
  frame.set_facecolor('white')
  strings = []
  strings.append('$T = {}$ K'.format(T))
  strings.append(r'$\log_{{10}} P_e = {:.2f}$'.format(np.log10(Pe)))
  strings.append(r'$\log_{{10}} N_H = {:.2f}$'.format(np.log10(float(data['N_H']))))
  strings.append('$y = {:.5f}$'.format(float(data['y_H'])))
  ax.set_title(r'$\quad$'.join(strings), fontsize='small')
  ax.set_xlabel('Wavelength, Å')
  ax.set_ylabel('Opacity per H per unit electron pressure / $10^{-26}$')
  ax.set_yscale(yscale)
  return None


# <a id="org45d3515"></a>
#
# ### Ejercicio 16: Reproducción de la figura 8.5
#
# Use la función `plot_opacities(Pe, T)` para reproducir los cuatro paneles de la figura. Para la (a), hay que ver los errata
