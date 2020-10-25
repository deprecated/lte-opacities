---
jupyter:
  jupytext:
    formats: ipynb,py:light,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernel_info:
    name: python3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

Astrofísica Estelar - Tarea 04
==============================

La meta de esta tarea es reproducir las figuras 8.5(a, b, c, d) del
libro de Gray. Estas figuras muestran el coeficiente de absorción
continua $\kappa_\lambda$ para para diferentes temperaturas $T$ y
presiones electrónicas $P_{e}$ en equilibrio termodinámico local (ETL).

Por simplicidad, consideramos solo hidrógeno, en forma de átomo neutro y
los iones positivos y negativos. Fracciones de ionización y excitación
de los niveles ligados se calculan con las ecuaciones de Saha y de
Boltzmann.

Nota importante
---------------

Hay unos errores en el libro, que están corregidos en [esta
página](http://astro.uwo.ca/~dfgray/Photospheres.html)

Plan general
------------

Las cuatro contribuciones a la opacidad que hay que calcular (anotadas
con las ecuaciones relevantes de Gray) son:

1.  Ligado-libre de H^0^
    -   ec. (8.8) - *pero ver abajo*
2.  Libre-libre de H^+^ + e^-^
    -   ec. (8.10)
3.  Ligado-libre de H^-^
    -   ecs. (8.11) y (8.12)
4.  Libre-libre de H^0^ + e^-^
    -   ec. (8.13)

Noten que Gray defina la $\kappa$ como la absorción por átomo de
hidrógeno neutro. Para convertir esta en la opacidad por masa, hay que
multiplicar por la fracción de H^0^ y dividir por la masa del hidrógeno:
$m_{\mathrm{H}}
= 1.67 \times 10^{{-24}}$ g. Ver su ecuación 8.18.

1.  Ligado-libre de H<sup>0</sup>.
    -   ec. (8.8) - *pero ver abajo*
2.  Libre-libre de H<sup>+</sup> + e<sup>-</sup>.
    -   ec. (8.10)
3.  Ligado-libre de H<sup>-</sup>.
    -   ecs. (8.11) y (8.12)
4.  Libre-libre de H<sup>0</sup> + e<sup>-</sup>.
    -   ec. (8.13)

Para las contribuciones (1) y (2), las ecuaciones (8.8) y (8.10) de Gray
mezclan la contribuciones de la ionización, la excitación, y las
secciones rectas. Yo creo que es más ilustrativo calcular estos partes
por separado. Es un poco más de trabajo, pero tiene la ventaja que
podemos hacer pruebas y gráficas de las diferentes partes.

Constantes físicos
------------------

Trabajamos con las energías en electron-volts, entonces queremos que
todos nuestros constantes sean consistentes con esto. Aquí hay una
manera de obtener eso

```python
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
```

A mi me da el resultado:

``` {.example}
BOLTZMANN = 8.617330337217213e-05 eV/K
PLANCK = 4.1356676623401646e-15 eV.s
RYDBERG = 13.60569300965081 eV
HMINUS_EION = 0.754 eV
BOHR_RADIUS = 5.2917721067e-09 cm
```
