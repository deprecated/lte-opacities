# -*- coding: utf-8 -*-
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

# Tarea 4 - LTE opacities for a pure H atmosphere


We try to reproduce Figure 8.5 a, b, c, d from Gray, which show the wavelength-dependent continuous absorption coefficient $\kappa_\lambda$ for different temperatures $T$ and electron pressures $P_e$.  For simplicity, we consider only hydrogen, in the form of the neutral atom and the positive and negative ions.  Ion fractions and excitation of bound levels is calculated under the assumption of local thermodynamic equilibrium. 

```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set(context='notebook', 
        style='whitegrid',
        palette='dark',
        font_scale=1.5,
        color_codes=True,
        rc={'figure.figsize': (8,6)},
       )
```

## Set up the constants we need

```python
import astropy.units as u
from astropy.constants import k_B, h, m_p, a0, m_e
from astropy.constants import c as light_speed
from astropy.table import Table, Column
```

We work with all energies in electron volts. We define the constants as regular floats because it doesn't seem possible to use `astropy.units` quantities with `@np.vectorize`d functions.

```python
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

## Ionization balance of hydrogen


### The general Saha equation

We use the following version of the Saha equation:
$$
\frac{N_{j+1} N_e}{N_j} = \Phi_j(T), 
$$
which relates the densities of adjacent ionization stages $j$ and $j+1$. 

First define the $T$-dependent function $\Phi_j = 4.8293744908\times 10^{15} \left(U_{j+1}/U_j\right) T^{1.5} e^{-E_j/k T}$, where $E_j$ is the $j$-th ionization potential and $U_j$, $U_{j+1}$ are the partition functions.


The temperature-dependent function on the RHS is
$$
\Phi_j(T) = 2 \left( \frac{2\pi m_e k T}{h^2} \right)^{1.5} \frac{U_{j+1}}{U_j} e^{-E_j/k T},
$$
where $E_j$ is the $j$-th ionization potential and $U_j$, $U_{j+1}$ are the partition functions.

We evaluate the constant

```python
saha_C = 2*(2*np.pi*m_e*k_B / (h**2))**1.5
saha_C.cgs
```

which is sometimes (e.g., Mihalas) given the other way up

```python
1./saha_C.cgs
```

and define a function to evaluate $\Phi_j(T;\, E_j, U_j, U_{j+1})$.  Default values of $U_j = 2$, $U_{j+1} = 1$ are correct for the H$^+$/H$^0$ balance at low temperatures.

```python
SAHA_CONSTANT = saha_C.cgs.value

def Saha_Phi(T, Eion=1.0*RYDBERG, Ui=1.0, Un=2.0):
    """
    Función Phi(T) = (Ni Ne / Nn) de Saha 
    para energía de ionización Eion,
    y con funciones de partición Ui y Un
    """
    return SAHA_CONSTANT * (Ui/Un) * T**1.5 * np.exp(-Eion/(BOLTZMANN*T))



```

Test the function for some typical temperatures.  

```python
Ts = np.array([3, 5, 9, 15])*u.kK
Ts.cgs
```

Note that the `T` argument should be a normal number (e.g, `float`) in units of Kelvin.  In this example, we set up the temperature array in kilo-Kelvin, so we need to convert to cgs (or SI) and take the `value` before sending it to the function.  

```python
Saha_Phi(Ts.cgs.value)
```

### The abundance of the positive hydrogen ion

We assume that the abundance of the negative ion H$^-$ is always a negligible of total H, so that we have equal numbers of protons and free electrons: $N_+ = N_e$. Then the H positive ionization fraction, $y = N_+ / N_H$, is the solution of the polynomial $y^2 + A y - A = 0$, where $A = \Phi_{H_0} / N_H$. 

We define a function `Hplus_fraction` that calculates $y$ as a function of total hydrogen density and temperature.  We use the `@np.vectorize` decorator so that we can apply the function to arrays of density and temperature. This is necessary here since `np.roots` solves only a single polynomial. 

```python
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
```

```python

```

For simplicity, we are here assuming that the H$^0$ partition function is equal to the statistical weight of the ground level: $g_1 = 2$.  This is a good approximation at lowish temperatures, where the population of excited levels is negligible.  We treat the more general case in the function `Hplus_fraction_U()` below. 


### The abundance of the negative hydrogen ion

The Saha equation for H⁻ is:
$$
\frac{N_{H^0} N_e}{N_{H^-}} = \Phi_{H^-}(T),
$$
from which it follows that 
$$
N_{H^-} \big/ N_H = \left( N_{H^0} \big/ N_H \right) N_e \big/ \Phi_{H^-} = (1 - y) y N_H \big/ \Phi_{H^-}
$$


```python
def Hminus_fraction(Hden, T):
    """
    Calcular fracción del ión negativo de hidrógeno
    """ 
    y = Hplus_fraction(Hden, T)
    return y * (1. - y) * Hden/Saha_Phi(T, Eion=HMINUS_EION, Ui=2.0, Un=1.0)
```

### Table and graphs of the ion fractions

Define some typical atmospheric densities.  Then, make a table of the ion fractions four these four densities and the four temperatures that we defined above.

```python
Ns = np.array([10, 3, 1, 0.5])*1e15/u.cm**3
Ns
```

```python
fmt = "0.4e"
Table(
  data=[
    Column(Ns.cgs, name=r'$N_H$', format=fmt),
    Column(Ts.cgs, name=r'$T$', format=fmt),
    Column(Hplus_fraction(Ns.cgs.value, Ts.cgs.value), name=r'$N_+/N_H$', format=fmt),
    Column(Hminus_fraction(Ns.cgs.value, Ts.cgs.value), name=r'$N_-/N_H$', format=fmt),
  ])
```

Note that the H⁻ fraction is always very small, which justifies ignoring its effect on the electron density. 

Next, we plot the ion fractions against temperature for a wide range of densities. 

```python
logNgrid = range(4, 20, 2)
Tgrid = np.linspace(2e3, 2e4, 500)
fig, ax = plt.subplots(1, 1)
legend_box_params = {
  'frameon': True,
  'fancybox': True,
  'fontsize': 'x-small',
}
colors = sns.color_palette('magma_r', n_colors=len(logNgrid))
epsilon = 0.01
for logN, c in zip(logNgrid, colors):
  ax.plot(Tgrid, Hplus_fraction(10**logN, Tgrid), color=c, 
          label=r'$N = 10^{{{}}}\ \mathrm{{cm}}^{{-3}}$'.format(logN))
frame = ax.legend(loc='lower right', **legend_box_params).get_frame()
frame.set_facecolor('white')
ax.set_ylim(-epsilon, 1 + epsilon)
ax.set_title('Positive hydrogen ion abundance')
ax.set_xlabel('Temperature, K')
ax.set_ylabel('H$^+$ fraction');
```

At the lower densities, hydrogen transitions from almost fully neutral to almost fully ionized over a narrow range of temperatures around 4000 K.  But such low densities are only seen in the corona, where LTE does not apply.   As the density is increased, higher temperatures are required and the curves shift to the right.  For densities characteristic of stellar photospheres, the transition occurs around 7000 to 10,000 K.

```python
fig, ax = plt.subplots(1, 1)
for logN, c in zip(logNgrid, colors):
  ax.semilogy(Tgrid, Hminus_fraction(10**logN, Tgrid), color=c,
          label=r'$N = 10^{{{}}}\ \mathrm{{cm}}^{{-3}}$'.format(logN))
frame = ax.legend(loc='lower left', ncol=2, **legend_box_params).get_frame()
frame.set_facecolor('white')
ax.set_title('Negative hydrogen ion abundance')
ax.set_xlabel('Temperature, K')
ax.set_ylabel('H$^-$ fraction');
```

The abundance of negative hydrogen ion is shown on a logarithmic scale.  It generally increases with density, and it has a peak at the temperature where H is about 50% ionized, as can be seen by comparing this graph with the previous one.


## Excitation of bound levels of H⁰


We use the Boltzmann equation to calculate the fractional population of a given bound level, $n$, of neutral hydrogen.  
$$
\frac{N_n}{N_{H^0}} = \frac{g_n}{U(T)} e^{-E_n/k T}
$$
where the degeneracy is $g_n = 2 n^2$ and the energy in Rydbergs above the ground ($n=1$) level is $E_n = 1 - n^{-2}$.

Here is the function to do that:

```python
def H0_level_population(n, T, U=2.0):
    """
    Calcular la población ETL del nivel n de hidrógeno neutro
    a una temperatura T kelvin
    """
    # Energía de excitación respeto a n=1
    E = RYDBERG * (1.0 - 1.0/n**2)
    # Peso estadístico
    g = 2.0*n**2
    return (g/U)*np.exp(-E/(BOLTZMANN*T))
```

```python

```

**[Extra credit: not required for tarea]**

At low temperatures, the population of excited levels is negligible and we can take $U(T) \approx g_1 = 2$.  But, in general we need to evaluate the partition function as
$$
U(T) = \sum_1^{n_\mathrm{max}}\ g_n\ e^{-E_n/k T}
$$
We can calculate this by re-using the `H0_level_population` function:

```python
@np.vectorize
def H0_partition_function(T, nmax):
  U = np.zeros_like(T)
  for n in range(1, int(nmax)+1):
    U += H0_level_population(n, T, U=1.0)
  return U
```

We cannot take $n_\mathrm{max} \to \infty$ in this func, since the sum diverges.  It is therefore important to find a physically motivated argument for determining the highest bound level, $n_\mathrm{max}$.

Taking account of the *pressure ionization* due to perturbations from neighboring particles, we make the approximation that in order that a level $n$ should be bound, the radius of the level, $r_n$, must be less than the average distance between particles: $\sim (N_H)^{-1/3}$.  Using $r_n = n^2 a_0$, where $a_0$ is the Bohr radius, this gives a maximum bound level $n_\mathrm{max} = a_0^{-1/2} N_H^{-1/6}$.  See Hubeny & Mihalas, Chapter 4, p. 91 for more details.


```python
H0_partition_function(15000.0, 1000)
```

```python
def nmax_pressure_ionization(Hden):
  """
  Calcular el nivel máximo ligado de H, sujeto a perturbaciones 
  por vecinos con densidad `Hden`
  """
  return 1./np.sqrt(BOHR_RADIUS*Hden**(1./3.))
```

Now we use the above function to make a table of $n_\mathrm{max}$ for different densities.  It is typicially $\sim 100$ for photospheric densities.  At the higher densities found in stellar interiors ($N_H > 10^{21}\ \mathrm{cm}^{-3}$) even the $n = 1$ level becomes unbound and H is fully ionized at all temperatures.

```python
logNgrid_wide = range(4, 28, 2)
Ns = (10**np.array(logNgrid_wide, dtype='float'))*u.cm**-3
Table(data=[
  Column(Ns, 
        name=r'Hydrogen density, $N_H$', format='{:.0e}'),
  Column(nmax_pressure_ionization(Ns.value).astype(int), 
        name=r'Maximum bound level, $n_\mathrm{max}$')])
```

Finally, we can return to the partition function, plotting it against $T$ using the $n_\mathrm{max}$ appropriate to different densities. For each density, the curves are only plotted for $T$ where the neutral hydrogen fraction, $1 - y$, is larger than $10^{-6}$.  We also show with symbols the points where the ionization fraction is $y = 0.95$ (squares) and $y = 0.999$ (circles).

```python
np.zeros_like(0.0)
```

```python
fig, ax = plt.subplots(1, 1)
colors = sns.color_palette('magma_r', n_colors=len(Ns))
for Hden, c in zip(Ns.value[::-1], colors[::-1]):
  nmax = int(nmax_pressure_ionization(Hden))
  Ugrid = H0_partition_function(Tgrid, nmax=nmax)
  mask = 1.0 - Hplus_fraction(Hden, Tgrid) > 1.e-6
  ax.plot(Tgrid[mask], Ugrid[mask], color=c,
         label=r'$N = 10^{{{}}}\ \mathrm{{cm}}^{{-3}}$'.format(int(np.log10(Hden))))
  for y, sym in [0.95, 's'], [0.999, 'o']:
    i0 = np.argmin(np.abs(Hplus_fraction(Hden, Tgrid) - y)) 
    ax.plot(Tgrid[i0], Ugrid[i0], sym, color=c)
ax.set_ylim(None, 10.)
frame = ax.legend(loc='lower left', ncol=2, **legend_box_params).get_frame()
frame.set_facecolor('white')
sigmatext = r'$U(T) = \sum_1^{n_\mathrm{max}}\ g_n\, e^{-E_n/k T}$'
ax.set_title('H$^0$ partition function: ' + sigmatext)
ax.set_xlabel('Temperature, K')
ax.set_ylabel(r'$U(T)$');        
```

It can be seen that $U(T)$ only rises noticeably above $2$ for densities above $10^8\ \mathrm{cm}^{-3}$, and that it only becomes large when the hydrogen is nearly completely ionized ($y \gtrapprox 0.999$).  For the highest density of $10^{26}\ \mathrm{cm}^{-3}$, we have $n_\mathrm{max} = 0$, which means that there are no bound states at all, so $U(T) = 0$.


In the function `Hplus_fraction` above, we calculated the hydrogen ionization fraction under the approximation that $U(T) = 2$.  We will now redo this function, but using the better approximation to $U(T)$ that we have just found.

For consistency, we should also incorporate the *continuum lowering* effect in the ionization balance.  It can be included in a simple way by reducing the H⁰ ionization potential. However, once the ground level becomes unbound, then the approximations that we are using are no longer valid, so we should not expect this to be accurate for very large densities. 

```python
@np.vectorize
def Hplus_fraction_U(Hden, T):
    """
    Calcular fracción de hidrógeno ionizado con un U(T) más realista
   	
   	`Hden` es densidad de partículas totales de H en cm^{-3}
    `T` es temperatura en K
    """
    nmax = nmax_pressure_ionization(Hden)
    if nmax < 1.0:
        # pressure ionization
        y = 1.0
    else:
        U = H0_partition_function(T, nmax=int(nmax))
        Ei = RYDBERG*(1.0 - 1.0/nmax**2)
        A = Saha_Phi(T, Eion=Ei, Un=U) / Hden
        # Resolver polinomio: y**2 + A*y - A = 0
        y = np.roots([1.0, A, -A])[1] # tomar raiz positivo
    return y
```

Now we compare the two approximations.  The constant-$U$ version is shown as a dashed line and the new version as a solid line.  We change to a logarithmic scale in temperature so we can see the effects of very large densities more clearly.

```python
Tgrid_wide = np.logspace(3.7, 7.7, 500)
logNgrid_wide = list(range(15, 25, 2)) + [24, 25]
#colors_wide = sns.color_palette('magma_r', n_colors=len(logNgrid_wide))
colors_wide = 'ygcbmrk'
fig, ax = plt.subplots(1, 1)
for logN, c in zip(logNgrid_wide, colors_wide):
  Hden = 10**logN
  nmax = int(nmax_pressure_ionization(Hden))
  ax.plot(Tgrid_wide, Hplus_fraction_U(Hden, Tgrid_wide), color=c, 
          label=rf'$N = 10^{{{logN}}}\ \mathrm{{cm}}^{{-3}}, n_\mathrm{{max}} = {nmax}$')
  ax.plot(Tgrid_wide, Hplus_fraction(Hden, Tgrid_wide), '--', color=c, 
          label=None)
frame = ax.legend(loc='lower right', **legend_box_params).get_frame()
frame.set_facecolor('white')
ax.set_ylim(-epsilon, 1 + epsilon)
ax.set_xscale('log')
ax.set_title('Positive hydrogen ion abundance with more realistic $U(T)$')
ax.set_xlabel('Temperature, K')
ax.set_ylabel('H$^+$ fraction');
```

At densities ${}\le 10^{15}\ \mathrm{cm}^{-3}$, there is almost no effect at all. At moderate densities of $10^{17}$ to $10^{21}\ \mathrm{cm}^{-3}$ the prinicipal effect is to increase the neutral fraction at temperatures where H is nearly fully ionized.  This is due to $U$ increasing, which favors the neutral atom.  But, for the very highest densities $\gt 10^{21}\ \mathrm{cm}^{-3}$, the continuum lowering starts to dominate and the partial ionization extends to lower temperatures due to reduction in the ionization potential.  This also tends to reduce $U$ again, since there *are* no excited levels to populate. Finally, for $N > 10^{24}\ \mathrm{cm}^{-3}$, even the ground level is unbound, so ionization is complete at all temperatures.


## Wavelength-dependent cross sections


### Neutral hydrogen H⁰

#### Bound-free photoionization cross sections

For photoionization from level $n$, there is a threshold energy, $E_n = n^{-2}\ \mathrm{Ry}$, with a corresponding minimum frequency, $\nu_n = E_n/h$, or maximum wavelength, $\lambda_n = h c / E_n$.  The cross section is given by
$$
\sigma_\mathrm{bf}(n, \nu) = \sigma_0 n \frac{\nu_n^3}{\nu^3} g_\mathrm{bf}(n, \nu)
$$
where $\sigma_0 = 2.815\times 10^{29} \nu_1^{-3} =  7.906 \times 10^{-18}\ \mathrm{cm}^2$ and $g_\mathrm{bf}(n, \nu)$ is the Gaunt factor that corrects for quantum mechanical effects.  

```python
@np.vectorize
def xsec_H0_boundfree(n, nu, xsec0=7.906e-18):
    """
    Sección eficaz de fotoionización de nivel n de H0 a frecuencia nu Hz

    Multiplicar por densidad de H0(n) para dar coeficiente de absorción (cm^{-1})
    """
    E = PLANCK*nu               # energía de fotón
    E0 = RYDBERG/n**2           # energía de ionización de nivel n

    if E >= E0:
        xsec = gaunt_H0_boundfree(n, nu)*xsec0*n*(E0/E)**3
    else:
        xsec = 0.0

    return xsec
```

```python

```

For the gaunt factor we use the Menzel & Perkis approximation given in Gray's Eq (8.5):
$$
g_\mathrm{bf}(n, \nu) = 1 - \frac{0.3456}{(\lambda R)^{1/3}} 
\left( \frac{\lambda R}{n^2} - \frac{1}{2}\right) .
$$

```python
def gaunt_H0_boundfree(n, nu):
  """
  Factor Gaunt para fotoionización de nivel `n` de H0 a frecuencia `nu` Hz
  """
  lambda_R = RYDBERG/(PLANCK*nu)
  return 1.0 - 0.3456*(lambda_R/n**2 - 0.5)/lambda_R**(1./3.)
```

Define an array of wavelengths for plotting and calculate the corresponding frequencies. 

```python
wavs = np.linspace(40.0, 20000.0, 500)*u.AA
freqs = (light_speed/wavs).cgs
freqs[[0, -1]]
```

```python
fig, ax = plt.subplots(1, 1)
for n in range(1, 5):
  m = h*freqs >= 1.0*u.Ry/n**2
  ax.plot(wavs[m], gaunt_H0_boundfree(n, freqs.value[m]), 
         label=r'$n = {}$'.format(n))
ax.set_ylim(0.0, None)
ax.legend()
ax.set_xlabel(r'Wavelength, Å')
ax.set_ylabel(r'$g_\mathrm{bf}(n, \nu)$')
ax.set_title('Bound-free gaunt factors');
```

The gaunt factors are of order unity, tending to increase slightly in the ultraviolet.  For each $n$, it only makes sense to plot them for $\lambda < \lambda_n$. 

```python
fig, ax = plt.subplots(1, 1)
for n in range(1, 6):
  ax.plot(wavs, xsec_H0_boundfree(n, freqs.value), 
         label=r'$n = {}$'.format(n))
ax.set_ylim(0.0, None)
ax.legend(loc='upper left')
ax.set_xlabel(r'Wavelength, Å')
ax.set_ylabel(r'$\sigma_\mathrm{bf}(n, \nu)$')
ax.set_title('Bound-free H$^0$ cross sections');
```

The cross sections can be compared with Gray's Fig 8.2.

<!-- #region -->
### Free-free H⁰ cross-sections


The cross section per electron (see Rybicki, section 5.3) can be written as 
$$
\alpha_\mathrm{ff} = \alpha_0 \frac{g_\mathrm{ff}(T, \nu)}{\nu^3 T^{1/2}} \quad \mathrm{cm^2\ \big/\ e^-}, 
$$
where 
$$
\alpha_0 = \frac{4 e^6}{3 m h c} \left(\frac{2\pi}{3 k m}\right)^{1/2} , 
$$
and the free-free Gaunt factor can be approximated (Gray, Eq. 8.6) as 
$$
g_\mathrm{ff}(T, \nu) = 1 - \frac{0.3456}{(\lambda R)^{1/3}} 
\left( \frac{k T}{h \nu} + \frac{1}{2}\right) .
$$
We calculate the numerical value of the constant, $\alpha_0$:
<!-- #endregion -->

```python
from astropy.constants import e, m_e
alpha0 = np.sqrt(2*np.pi/(3*k_B*m_e))*(4*e.esu**6)/(3*m_e*h*light_speed)
alpha0.cgs
```

```python
def xsec_H0_freefree(T, nu):
    """
    Sección eficaz por electrón de bremsstrahlung a frecuencia nu Hz

    Multiplicar por Ne N(H+) para dar coeficiente de absorción (cm^{-1})
    """
    # cf. Rybicki, eq. 5.18b, but we omit the (1 - exp(-h nu/k T)) term
    # since we will apply it later
    return alpha0.cgs.value * gaunt_H0_freefree(T, nu) * T**-0.5 / nu**3
```

```python
def gaunt_H0_freefree(T, nu):
  """
  Factor Gaunt para absorción libre-libre H0 a frecuencia `nu` Hz
  """
  lambda_R = RYDBERG/(PLANCK*nu)
  return 1.0 - 0.3456*(BOLTZMANN*T/(PLANCK*nu) + 0.5)/lambda_R**(1./3.)
```

```python
fig, ax = plt.subplots(1, 1)
for T in [5e3, 1e4, 2e4]:
  ax.plot(wavs, gaunt_H0_freefree(T, freqs.value), 
         label=r'$T = {:.0f}$ K'.format(T))
ax.set_ylim(0.0, None)
ax.legend(loc='lower left')
ax.set_xlabel(r'Wavelength, Å')
ax.set_ylabel(r'$g_\mathrm{ff}(T, \nu)$')
ax.set_title('H$^0$ free-free gaunt factors');
```

```python
fig, ax = plt.subplots(1, 1)
for T in [5e3, 1e4, 2e4]:
  ax.plot(wavs, xsec_H0_freefree(T, freqs.value), 
         label=r'$T = {:.0f}$ K'.format(T))
ax.set_ylim(0.0, None)
ax.legend(loc='upper left')
ax.set_xlabel(r'Wavelength, Å')
ax.set_ylabel(r'$\sigma_\mathrm{ff}(T, \nu) / N_e N_{H^+}$, cm$^5$')
ax.set_title('Free-free H$^0$ cross sections');
```

### Negative hydrogen ion H⁻

#### Bound-free H⁻ cross section

We use the polynomial fit from Gray, which is stated to be accurate in the range $2250~Å < \lambda < 15,000~Å.$  This gives the cross section in $\mathrm{cm}^2$, so it needs to be multiplied by $N_{H^-}$. 

```python
@np.vectorize
def xsec_Hminus_boundfree(nu):
    """
    Sección eficaz de fotoionización del ión negativo H- a frecuencia nu Hz

    Multiplicar por N(H-) para dar coeficiente de absorción (cm^{-1})
    """
    # convertir nu a lambda en unidades de micras (10,000 Å)
    wav = (light_speed / (nu * u.Hz)).to(u.micron).value  
    # Fórmula y constantes de Gray, Eq. 8.11
    A = [1.99654, -1.18267e-1, 2.64243e2, 
         -4.40524e2, 3.23992e2, -1.39568e2, 2.78701e1]
    xsec = 0.0
    # El ajuste es preciso para 2250 Å <= lambda <= 15,000 Å 
    # Hay que cortarlo a partir de 16,200 Å porque el ajuste va negativo
    for i, a in enumerate(A):
        if wav <= 1.62:
            xsec += a*wav**i
    return xsec * 1.e-18
```

```python
fig, ax = plt.subplots(1, 1)
ax.plot(wavs, xsec_Hminus_boundfree(freqs.value)/1e-18, 
         label=r'bf')
ax.set_ylim(0.0, 5e-17)
#ax.legend(loc='lower center')
ax.set_yscale('log')
ax.set_ylim(1.0, 100.0)
ax.set_xlabel(r'Wavelength, Å')
ax.set_ylabel(r'$\sigma_\mathrm{bf,H^-}(\nu)$, $10^{-18}\ \mathrm{cm}^2 / \mathrm{H}^-$')
ax.set_title('Bound-free H$^-$ cross section');
```

The graph above bears a reasonable resemblance to Gray's Fig. 8.3


#### Free-free H⁻ opacity

This is also calculated from polynomial fits given by Gray, which accurately reproduce the results of Bell & Berrington (1987) for the range $1823~Å < \lambda < 151,890~Å$ and $1400~\mathrm{K} < T < 10,080~\mathrm{K}$.

```python
def Hz_to_AA(nu):
  """
  Utility function to translate frequency to wavelength
  """
  return (light_speed / (nu / u.s)).to(u.AA).value

@np.vectorize
def xsec_Hminus_freefree(T, nu):
    """
    Opacidad libre-libre del ión negativo H- a frecuencia nu Hz

    Multiplicar por Pe N(H0) para dar coeficiente de absorción (cm^{-1})
    + Ojo que no hay que multiplicar por N(H-)
    + Y esto ya incluye la correción por emisión estimulada
    """
    # convertir nu a lambda en unidades de Å
    wav = Hz_to_AA(nu)  
    logwav = np.log10(wav)
    # Eq. 8.13 de Gray
    f0 = -2.2763 - 1.6850*logwav + 0.76661*logwav**2 - 0.053346*logwav**3
    f1 = 15.2827 - 9.2846*logwav + 1.99381*logwav**2 - 0.142631*logwav**3
    f2 = (-197.789 + 190.266*logwav 
          - 67.9775*logwav**2 + 10.6913*logwav**3 - 0.625151*logwav**4)
    theta = np.log10(np.e) / (BOLTZMANN*T) # aproximadamente theta = 5040/T
    xsec = 1.e-26 * 10**(f0 + f1*np.log10(theta) + f2*np.log10(theta)**2)
    return xsec
```

The free free opacity is more important at longer wavelengths, so we define an extended range of wavelengths for plotting, up to just over $10~µm$. 

```python
wavs_extend = np.logspace(3.1, 5.1, 500)*u.AA
freqs_extend = (light_speed/wavs_extend).cgs
```

```python
fig, ax = plt.subplots(1, 1)
for T in [2520.0, 5040.0, 10080.0]:
  ax.plot(wavs_extend, xsec_Hminus_freefree(T, freqs_extend.value)/1e-26, 
         label=r'$T = {:.0f}$ K'.format(T))
ax.plot()
ax.set_ylim(0.01, 500)
ax.set_xlim(1000.0, 150000.0)
ax.legend(loc='upper left')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'Wavelength, Å')
ax.set_ylabel(r'$\alpha_\mathrm{ff,H^-}(T, \nu),\quad 10^{-26}/ P_e N_\mathrm{H^0}$ ')
ax.set_title('Free-free H$^-$ opacity');
```

This graph closely resemble Gray's Fig. 8.4.  Note that the fits already include the correction for stimulated emission and are per neutral H atom and per unit electron pressure.


## Finding the total hydrogen density in terms of electron pressure

The graphs we are trying to reproduce are for fixed values of $T$ and $P_e$, but most of our equations are in terms of densities, so need functions to convert between the two.  Going from hydrogen density to electron pressure is straightforward:

```python
def funcPe(Hden, T):
    """
    Presión electrónica como función de densidad total y temperatura
    """
    return Hden*Hplus_fraction_U(Hden, T)*k_B.cgs.value*T
```

```python
def funcPe_simple(Hden, T):
    """
    Presión electrónica como función de densidad total y temperatura
    """
    return Hden*Hplus_fraction(Hden, T)*k_B.cgs.value*T
```

At high temperatures, ionization is complete and $P_e$ increases linearly with $T$, which looks quite flat on the following graph because of the logarithmic scale on the $y$ axis. At lower temperatures the ionization fraction falls, and so $P_e$ drops steeply.

```python
fig, ax = plt.subplots(1, 1)
for Hden in [1e12, 1e14, 1e16, 1e18]:
  ax.plot(Tgrid, funcPe(Hden, Tgrid), 
          label=r'$N_H = 10^{{{:.0f}}}\ \mathrm{{cm^{{-3}}}}$'.format(np.log10(Hden)))
  ax.plot(Tgrid, funcPe_simple(Hden, Tgrid), ls=":", c="k",
          label="_nolabel_")
frame = ax.legend(loc='lower right', **legend_box_params).get_frame()
frame.set_facecolor('white')
ax.set_yscale('log')
ax.set_ylim(1e-6, None)
ax.set_title('Electron pressure')
ax.set_xlabel('Temperature, K')
ax.set_ylabel('$P_e$, dyne cm$^{-3}$');
```

Going in the other direction requires solving an implicit equation:

```python
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
```

```python

```

We now test this function by making a graph of total hydrogen density for electron pressures $P_e = 1 \to 1000\ \mathrm{dyne\ cm^2}$ and temperatures $T = 4500 \to 20,000\ \mathrm{K}$. We can't go to much lower temperatures because the electron fraction becomes so low that it is impossible to find a reasonable solution for the higher values of $P_e$.

```python
fig, ax = plt.subplots(1, 1)
for Pe in [1.0, 10., 100., 1000.]:
  m = Tgrid >= 4500.0
  ax.plot(Tgrid[m], funcHden(Pe, Tgrid[m]), 
          label=r'$P_e = {:.0f}\ \mathrm{{dyne\ cm^{{-2}}}}$'.format(Pe))
frame = ax.legend(loc='upper right', **legend_box_params).get_frame()
frame.set_facecolor('white')
ax.set_yscale('log')
ax.set_ylim(None, 3e21)
ax.set_title('Total hydrogen density required for given electron pressure')
ax.set_xlabel('Temperature, K')
ax.set_ylabel('$N_{H}$, cm$^{-3}$');
```

## Total wavelength-dependent opacities


```python
def opacidad_total(Pe, T, wavs):
    """
    Calcular la opacidad total del continuo de un gas de H puro en ETL

    Parámetros de entrada:
    
    Pe   : presión de electrones (dyne cm^{-2})
    T    : temperatura (K)
    wavs : longitud de onda (Å)

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
```

```python

```

```python
opacidad_total(Pe=3000.0, T=1e4, wavs=[3000, 10000])
```

### Reproducing Gray's Fig 8.5

```python
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
```

#### Fig 8.5 (a) — 5143 K

```python
plot_opacities(10**1.08, 5143.0)
```

Lowest temperature. Dominated by H$^-$ opacity.  _Why does Gray not get such a high free-free opacity as we do?_


#### Fig 8.5 (b) — 6429 K

```python
plot_opacities(10**1.77, 6429.0)
```

Start to see the H$^0$ absorption edges superimposed on the H$^-$.  Of the four graphs, this is the one that looks most like Gray's version.


#### Fig 8.5 (c) — 7715 K

```python
plot_opacities(10**2.50, 7715.0)
```

H$^0$ and H$^-$ are of roughly equal importance at this temperature.


#### Fig 8.5 (d) — 11,572 K

```python
plot_opacities(10**2.76, 11572.0)
```

H$^0$ opacity now completely dominates. _My excited levels are not as high as in Gray's graph, – why?_ 

Note that in the 2016 version, my general magnitude was 20 times lower.  But that is because I was normalizing by the total H density, whereas Gray is using (strangely) the neutral atomic hytrogen density.  This only matters for panel (d), since it is the only case where H is appreciably ionized.


#### A much higher temperature and density

```python
plot_opacities(10**7.76, 50000.0, wavrange=[300, 100000])
```

Now we see the pressure-ionization of the upper levels (only $n \le 8$ are populated). 

```python
plot_opacities(10**4.76, 50000.0, wavrange=[300, 1e6])
```

```python
plot_opacities(1e-6, 20000.0, wavrange=[300, 1e9])
```

## Check against Cloudy results

```python
from astropy.table import Table
```

```python
Table.read?
```

```python
fn = "cloudy/lte_opacity_6429.opac"
tc = Table.read(fn, format='ascii.tab', names=['nu', 'tot', 'abs', 'scat', 'albedo', 'elem'])
```

```python
tc['wav'] = 912.0/tc['nu']
wavmin, wavmax = 2500, 20000
m = (tc['wav'] >= wavmin) & (tc['wav'] <= wavmax)
tc[m]
```

```python
fig, ax = plt.subplots()
ax.plot(tc[m]['wav'], tc[m]['tot'])
ax.set(xlim=[wavmin, wavmax], ylim=[0.0, None])
```

### First concentrate on ionization fractions and level populations

I have run some grid models based on the Cloudy test script `limit_lte_hminus.in`, in which I vary the density.


```python
fn = "cloudy/limit_lte_hminus_density_grid.hcond"
thc = Table.read(fn, format='ascii.commented_header', delimiter='\t')
thl = Table.read(fn.replace('hminus', 'hminus_large'), format='ascii.commented_header', delimiter='\t')
thf = Table.read(fn.replace('limit', 'force'), format='ascii.commented_header', delimiter='\t')
```

The `thc` table is from model where we let Cloudy find the equilibrium on its own

```python
thc[::4]
```

While the `thf` table is from model where we try to force LTE.

```python
thf[::4]
```

The `thl` model has 40 H$^0$ resolved levels instead of just 10 

```python
thl[::4]
```

#### Plot the H$^0$ and H$^+$ fractions:

```python
fig, ax = plt.subplots()
#ax.plot(thc['HDEN'], thc['HI/H'], label='Cloudy H$^0$')
ax.plot(thc['HDEN'], thc['HII/H'], label='Cloudy H$^{+}$')
ax.plot(thc['HDEN'], thf['HII/H'], label='Cloudy LTE H$^{+}$')
ax.plot(thc['HDEN'], thl['HII/H'], label='Cloudy Large H$^{+}$')
ax.plot(thc['HDEN'], Hplus_fraction_U(thc['HDEN'], 5000.), 'o', c='y', label='My LTE H$^{+}$')
ax.set(xscale='log', yscale='log', xlabel='H density, cm$^{-3}$', ylabel='Fraction')
leg = ax.legend(frameon=True, framealpha=0.8)
leg.get_frame().set_facecolor('white')
fig.savefig('cloudy/density_grid_hplus.pdf')
fig.savefig('cloudy/density_grid_hplus.png')
None
```

The green line and yellow dots are indistinguishable. This is a good start - at least we agree about the LTE H ionization.  However, the blue line (Cloudy free-wheeling model) *does* show deviations for $N > 10^{13.5}$.


#### Now the negative hydrogen abundance: H$^-$

```python
fig, ax = plt.subplots()
ax.plot(thc['HDEN'], thc['H-/H'], label='Cloudy H$^{-}$')
ax.plot(thc['HDEN'], thf['H-/H'], label='Cloudy LTE H$^{-}$')
ax.plot(thc['HDEN'], thl['H-/H'], label='Cloudy Large H$^{-}$')
ax.plot(thc['HDEN'], Hminus_fraction(thc['HDEN'], 5000.), 'o', c='y', label='My LTE H$^{-}$')
ax.set(xscale='log', yscale='log', xlabel='H density, cm$^{-3}$', ylabel='Fraction')
leg = ax.legend(frameon=True, framealpha=0.8)
leg.get_frame().set_facecolor('white')
None
```

So, again, we see deviations for high densities.  But this time, even the LTE Cloudy model disagrees with my Saha equation. 


Could it be some kind of continuum lowering that is going on?  Since H$^-$ has only one bound level, then the close packing effect will not kick in until the density is so high that pressure ionization occurs. 


#### Level populations


We can't just read the data in with `Table.read` because the lines are of unequal lengths


Define a function to deal with inconsistent row lengths

```python
def pad_row(str_vals, ncols):
    nmissing = ncols - len(str_vals)
    if nmissing > 0:
        # Pad with empty strings if we have too few values
        return str_vals + ['']*nmissing
    elif nmissing < 0:
        # Truncate if we have too many values
        return str_vals[:ncols]
    else:
        # Case of "just right"
        return str_vals
```

```python
pad_row(['a', 'b'], 4)
```

```python
pad_row(['a', 'b', 'c', 'd', 'e'], 4)
```

```python
from astropy.io import ascii

##
## NOTE: This class can only be defined once per session
##
class NoHeaderVary(ascii.NoHeader):
    """Reader for Cloudy output files with variable length rows
    """
    _format_name = 'vary'
    _description = 'Basic table with variable length rows'
    def inconsistent_handler(self, str_vals, ncols):
        return pad_row(str_vals, ncols)

```

```python
def get_level_lists(nresolved = 10, ncollapsed = 100):
    levels_n_l = [[f'n({qn},{ql})' for ql in range(qn)] for qn in range(1, nresolved+1)] 
    levels_n_l += [[f'n({qn})'] for qn in range(nresolved+1, ncollapsed+nresolved)]

    flat_levels = []
    for sublevels in levels_n_l:
        flat_levels.extend(sublevels)

    return levels_n_l, flat_levels


levels_n_l, flat_levels = get_level_lists()
levels_n_l_large, flat_levels_large = get_level_lists(40, 60)

flat_levels_large[-100:-50]
```

```python
names = ['depth', 'n(H0)', 'n(H+)'] + flat_levels
tpc = Table.read(fn.replace('.hcond', '.hpop'), format='ascii.vary', names=names, guess=False, delimiter='\t')
tpf = Table.read(fn.replace('limit', 'force').replace('.hcond', '.hpop'), format='ascii.vary', names=names, delimiter='\t')
```


```python
tpc
```

```python
len(tpc[1])
```

```python
tpf
```

```python
names = ['depth', 'n(H0)', 'n(H+)'] + flat_levels_large
tpl = Table.read(fn.replace('hminus', 'hminus_large').replace('.hcond', '.hpop'), format='ascii.vary', names=names, guess=False, delimiter='\t')
```

```python
tpl
```

```python
len(tpl[1])
```

Sum over all the $l$ states to get the population fractions for each $n$.

```python
def make_table_pop_n(tab_pop_n_l, levels_n_l):
    # Start with one column of the total H0
    tnfracs = tab_pop_n_l[['n(H0)']]
    # Loop over all the n-levels
    for i, sublevels in enumerate(levels_n_l):
        qn = i + 1  # quantum number n
        # Add a column with total fraction for this n by summing the individual l populations
        tnfracs[f'n({qn})'] = np.sum([tab_pop_n_l[s] for s in sublevels], axis=0)/tab_pop_n_l['n(H0)']
    return tnfracs

tpf_n = make_table_pop_n(tpf, levels_n_l)
tpc_n = make_table_pop_n(tpc, levels_n_l)
tpl_n = make_table_pop_n(tpl, levels_n_l_large)
```

```python
fig, ax = plt.subplots()
T = 5000.0
qn_maxplot = 40
colors = sns.color_palette('magma_r', n_colors=qn_maxplot)
nmax = nmax_pressure_ionization(tpf_n['n(H0)'])
print(nmax.data.astype(int))
U = H0_partition_function(T, nmax=nmax)
for i, c in list(enumerate(colors)):
    qn = i + 1
    myfracs = H0_level_population(qn, T, U=U)
    ax.plot(tpf_n['n(H0)'], tpf_n[f'n({qn})'] / myfracs, color=c, label=f'$n = {qn}$')
ax.set(xscale='log', yscale='linear', xlabel='H density, cm$^{-3}$', 
       ylabel='Cloudy fraction / My LTE fraction', 
       ylim=[0.0, 1.3])
ax.legend(ncol=4, fontsize='xx-small', loc='lower left', title='H$^0$ levels')
None
```

```python
fig, ax = plt.subplots()
T = 5000.0
qn_maxplot = 40
colors = sns.color_palette('magma_r', n_colors=qn_maxplot)
nmax = nmax_pressure_ionization(tpc_n['n(H0)'])
#print(nmax.data.astype(int))
#print(U - 2)
U = H0_partition_function(T, nmax=nmax)
for i, c in list(enumerate(colors)):
    qn = i + 1
    myfracs = H0_level_population(qn, T, U=U)
    ax.plot(tpc_n['n(H0)'], tpc_n[f'n({qn})'] / myfracs, color=c, label=f'$n = {qn}$')
ax.set(xscale='log', yscale='linear', xlabel='H density, cm$^{-3}$', 
       ylabel='Cloudy fraction / My LTE fraction', 
       ylim=[-0.19, 1.19])
leg = ax.legend(ncol=4, fontsize='xx-small', loc='lower left', title='H$^0$ levels', frameon=True, framealpha=0.8)
leg.get_frame().set_facecolor('white')
None
```

```python
fig, ax = plt.subplots()
T = 5000.0
qn_maxplot = 40
colors = sns.color_palette('magma_r', n_colors=qn_maxplot)
nmax = nmax_pressure_ionization(tpl_n['n(H0)'])
#print(nmax.data.astype(int))
#print(U - 2)
U = H0_partition_function(T, nmax=nmax)
for i, c in list(enumerate(colors)):
    qn = i + 1
    myfracs = H0_level_population(qn, T, U=U)
    ax.plot(tpl_n['n(H0)'], tpl_n[f'n({qn})'] / myfracs, color=c, label=f'$n = {qn}$')
ax.set(xscale='log', yscale='linear', xlabel='H density, cm$^{-3}$', 
       ylabel='Cloudy fraction / My LTE fraction', 
       ylim=[-0.19, 1.19])
leg = ax.legend(ncol=4, fontsize='xx-small', loc='lower left', title='H$^0$ levels', frameon=True, framealpha=0.8)
leg.get_frame().set_facecolor('white')
None
```

Something simpler would just be to sum all the excited levels.

```python
frac_tab = Table({'n(H0)': tpc_n['n(H0)'].data, 
                  'n*/n': np.sum(tpc_n.columns.values()[2:], axis=0), 
                  'n*/n LTE': np.sum(tpf_n.columns.values()[2:], axis=0), 
                  'n*/n large': np.sum(tpl_n.columns.values()[2:], axis=0)})
```

```python
#np.sum([c.data for c in tpl_n.columns.values()[2:]], axis=0)
np.sum(tpl_n.columns.values()[2:], axis=0)
#[c for c in tpl_n.columns.values() if '1' in c.name][:4]
```

```python
frac_tab
```

And we do the same from my own functions:

```python
@np.vectorize
def H0_total_excited_population(Hden, T):
  nmax = nmax_pressure_ionization(Hden)
  U = H0_partition_function(T, nmax=nmax)
  pop = np.zeros_like(T)
  for n in range(2, int(nmax)+1):
    pop += H0_level_population(n, T, U=U)
  return pop
```

```python
H0_total_excited_population(1e12, 5000.)
```

```python
fig, ax = plt.subplots()
T = 5000.0
ax.plot(frac_tab['n(H0)'], frac_tab['n*/n'], label=r'Cloudy Total $n=2 \to 110$')
ax.plot(frac_tab['n(H0)'], frac_tab['n*/n LTE'], label=r'Cloudy LTE Total $n=2 \to 110$')
ax.plot(frac_tab['n(H0)'], frac_tab['n*/n large'], label=r'Cloudy Large Total $n=2 \to 99$')
ax.plot(frac_tab['n(H0)'], H0_total_excited_population(frac_tab['n(H0)'], T), 'o', c='y', label=r'My LTE Total $n=2 \to \infty$')
ax.set(xscale='log', yscale='linear', ylim=[0.0, 2.e-8], xlabel='H density, cm$^{-3}$', ylabel=r'Fraction')
leg = ax.legend(title='Sum of H$^0$ excited levels', frameon=True, framealpha=0.8)
leg.get_frame().set_facecolor('white')
None
```

So the disagreement of Cloudy with my values for low density is to be expected, since Cloudy only uses 110 levels, which is less than $n_\mathrm{max}$ for $N < 10^{13}\ \mathrm{cm}^{-3}$.


For higher densities, the agreement is very good with the Cloudy forced LTE model, but the non-forced model falls consistently below. 

```python

```
