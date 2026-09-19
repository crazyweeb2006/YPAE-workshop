<div align="center">

# 🌌 Cosmic Gas Disc

### A real-time protoplanetary disk sandbox: one disk, five ways of seeing it

Interactive · GPU-accelerated with Taichi · 100,000 dust tracers · embedded planet · live astrophysical telemetry

</div>

<!--
  Add a hero GIF or screenshot here before publishing, e.g.
  ![Cosmic Gas Disc demo](docs/demo.gif)
-->

---

## What is this?

**Cosmic Gas Disc** lets you fly around a young star's protoplanetary disk and poke at it. A hundred thousand dust tracers of different sizes orbit a 1 M☉ star inside an analytic, flared gas disk. They feel gas drag, settle toward the midplane, drift radially, and get stirred by an embedded planet whose mass, orbit, eccentricity and migration you control in real time.

A telemetry deck computes the textbook quantities behind what you see (Hill radius, Crida gap parameter, Stokes number, Toomre Q, and more). Five colour modes, inspired by ALMA, JWST, AstroSat UVIT and CO kinematics, map the same particles to different "wavelengths".

It is an **interactive exploration and teaching tool**, not a research code. See [Scope & limitations](#-scope--limitations) for exactly what is and isn't modelled.

> The default setup is a compact, Solar-System-scale analogue: a disk from 0.6 to 18 AU with a Jupiter-mass planet at 5.2 AU. The visual language is inspired by images of systems like PDS 70, HL Tau and TW Hya.

---

## ✨ Highlights

- **100,000 dust tracers** integrated on the GPU (Taichi), with grain sizes drawn from an MRN-style power law
- **Size-dependent gas drag** with a density-dependent stopping time, radial drift, and turbulent vertical settling
- **Embedded planet**: mass 0-12 M<sub>Jup</sub>, orbital radius 1.2-13 AU, eccentricity up to 0.5, optional inward migration
- **Orbital overlays**: planet orbit, Hill sphere, L1-L5, 2:1 inner and 1:2 outer resonances, dust sublimation ring
- **Live telemetry deck** with ~30 computed quantities (star, planet, disk, dust, gap opening)
- **Five spectral views** of the same disk (`1`-`5`)
- **Click the disk** to inspect the local temperature, scale height, density, viscosity and stability
- **Camera bookmarks** with eased fly-to transitions, plus a cinematic auto-orbit

---

## 🚀 Quick start

```bash
git clone https://github.com/YOUR_USERNAME/cosmic-gas-disc.git
cd cosmic-gas-disc
pip install taichi numpy
python cosmic_gas_disc.py
```

| Flag | Backend |
| --- | --- |
| *(none)* | Vulkan (default, recommended) |
| `--opengl` | OpenGL fallback (support depends on your GPU and Taichi build) |
| `--cpu` | CPU (slowest, works everywhere) |

A Vulkan-capable GPU is recommended for 100,000 particles at interactive frame rates.

### A 60-second tour

1. Launch it and let the disk orbit. Drag with the left mouse button to orbit the camera.
2. Press `1` to `5` to flip between the spectral views.
3. Press `M` a few times to raise the planet's mass and watch the **Crida gap metric** and **gap depth** in the right-hand panel change.
4. Press `G` to switch gas drag off and compare how the dust behaves.
5. Click anywhere on the disk to inspect local conditions at that radius.

---

## 🎮 Controls

| Input | Action |
| --- | --- |
| `SPACE` | Pause / resume |
| `R` | Reset dust particles |
| `1` `2` `3` `4` `5` | ALMA · JWST NIRCam · UV / Hα · Thermal IR · CO Doppler |
| `G` | Toggle gas drag |
| `+` / `-` | Time warp |
| `[` / `]` | Planet orbital radius in / out |
| `M` / `N` | Planet mass up / down |
| `H` | Toggle the control and telemetry panels |
| Left-drag / `A` `D` | Orbit camera |
| Right-drag / `W` `S` | Zoom |
| Click disk | Inspect the disk at that radius |
| Click sky | Look up the nearest catalogue star along the view ray (77-star Yale BSC5 subset) |
| `ESC` | Quit |

The side panel also has sliders for time warp, substeps, planet mass, radius and eccentricity, migration rate, Stokes-number scale, aspect ratio, turbulence α, base surface density, particle size, and toggles for guide rings and Lagrange points.

---

## 🔭 Spectral views

All five views colour the **same tracer particles** differently. They are illustrative false-colour maps, not radiative-transfer or instrument simulations.

| Key | View | What it emphasises | How it is drawn |
| :-: | --- | --- | --- |
| `1` | **ALMA 1.3 mm dust continuum** *(DSHARP-inspired)* | Pebble concentrations, cleared gap, a dust-trap ring | Colour scales with the tracer's Stokes number, is darkened near the planet, and is brightened in a ring near 6.4 AU |
| `2` | **JWST NIRCam scattered light** | Flared surface layers, inner-rim shadow, spiral arms | Brightness grows with height above the midplane in units of the local scale height, plus an analytic two-armed spiral phased to the planet |
| `3` | **UV / Hα** *(AstroSat UVIT-inspired)* | Accretion-heated regions near the star and around the planet | Glow profiles centred on the star and the planet |
| `4` | **Thermal mid-IR** *(JWST MIRI 10 µm-inspired)* | Warm inner disk fading to a cool outer disk | Colour follows the radial temperature profile T ∝ r<sup>-0.43</sup> |
| `5` | **CO Doppler map** | Rotation and the velocity field | Tracer velocity projected onto a fixed line of sight in the disk plane (blue approaching, red receding) |

---

## 🧮 Physics model

The simulation works in **AU · M☉ · years**, where G = 4π².

### Gas disk (analytic, prescribed)

$$T(r)=280\,\mathrm{K}\,\left(\frac{r}{\mathrm{AU}}\right)^{-0.43},\qquad c_s=\sqrt{\frac{\gamma k_B T}{\mu m_H}}\quad(\gamma=1.4,\ \mu=2.3)$$

$$H=\frac{c_s}{\Omega_K},\qquad h(r)=\frac{H}{r}\approx h_0\left(\frac{r}{\mathrm{AU}}\right)^{0.285}\quad\left(0.285=\tfrac{1-q}{2}\ \text{for}\ q=0.43\right)$$

$$\Sigma(r)=\Sigma_0\left(\frac{r}{\mathrm{AU}}\right)^{-1}e^{-r/R_c},\qquad R_c=18\ \mathrm{AU},\quad \Sigma_0=1700\ \mathrm{g\,cm^{-2}}$$

The vertical structure is Gaussian, and the gas rotates slightly sub-Keplerian:

$$v_\mathrm{gas}=v_K(1-\eta),\qquad \eta\simeq1.375\,h^2$$

### Dust aerodynamics

- **Grain sizes** follow an MRN-style distribution, $dn/ds\propto s^{-3.5}$, starting at 1 µm.
- **Stokes number** is a scaled prescription: $\mathrm{St}=\mathrm{St}_0\,(s/1\,\mathrm{mm})\,(r/\mathrm{AU})^{0.75}$, where $\mathrm{St}_0$ is the GUI "Stokes No. Scale".
- **Stopping time** is density-dependent: $t_s=\mathrm{St}/(\Omega_K\hat\rho(z))$, with $\hat\rho=\max(e^{-z^2/2H^2},\,0.02)$.
- **Drag** uses the exact linear (Epstein-type) update $\mathbf v\leftarrow\mathbf v_\mathrm{gas}+(\mathbf v-\mathbf v_\mathrm{gas})\,e^{-\Delta t/t_s}$.
- **Initial conditions** use the classic drag-coupled drift solution (Nakagawa et al. 1986): $v_r=-2\eta v_K\,\mathrm{St}/(1+\mathrm{St}^2)$.
- **Vertical settling** is set at initialisation by $H_d=H/\sqrt{1+\mathrm{St}/\alpha}$. During the run, drag and a stochastic vertical kick scaled with $\sqrt{\alpha}\,c_s$ set the balance between settling and mixing.

### Planet and gap opening (diagnostics)

$$R_H=a\left(\frac{M_p}{3M_\star}\right)^{1/3},\qquad R_L=a\,\frac{0.49\,q^{2/3}}{0.6\,q^{2/3}+\ln(1+q^{1/3})}\ \ (\text{Eggleton 1983})$$

$$P_\mathrm{Crida}=\frac{3H}{4R_H}+\frac{50}{q\,\mathrm{Re}},\qquad \frac{\Sigma_\mathrm{gap}}{\Sigma_0}=\frac{1}{1+0.04K},\quad K=q^2h^{-5}\alpha^{-1}\ \ (\text{Kanagawa et al. 2015})$$

Here $q=M_p/M_\star$ and $\mathrm{Re}=a^2\Omega_K/\nu$ with $\nu=\alpha c_s H$. The HUD also reports the circumplanetary-disk radius ($\approx R_H/3$), Toomre $Q=c_s\Omega_K/\pi G\Sigma$, the dust sublimation radius (from stellar luminosity, $T_\mathrm{sub}=1500$ K), and an illustrative planetary Hα luminosity (an assumed accretion rate and conversion fraction, not a fit to observations).

Overlay geometry: L1/L2 sit at $a\mp R_H$, L3 opposite the planet, and L4/L5 at ±60°. The resonance rings are at $0.630\,a$ (2:1 inner) and $1.587\,a$ (1:2 outer).

### Numerical integration

Each substep is a velocity-Verlet-style **kick → drag/turbulence → drift → kick** update. Gravity comes from the central star and the planet, both softened. Drag is applied between the half-kicks. Particles that leave the annulus 0.45-20.7 AU are re-injected near the outer edge (about 15.5-17.9 AU) so the disk stays populated. The timestep is 0.0025 yr × time warp ÷ substeps.

---

## 📊 Telemetry deck

| Group | Quantities |
| --- | --- |
| **Star** | Mass, luminosity (T<sub>eff</sub> = 4000 K, R = 2 R☉), dust sublimation radius |
| **Planet** | Mass, orbital radius, orbital speed and period, Hill radius, Roche lobe, CPD radius, Hα luminosity |
| **Disk (at the planet or at a clicked radius)** | Temperature, sound speed, scale height, aspect ratio, surface density, midplane density, viscosity, Reynolds number |
| **Dust and gap opening** | Stopping time, maximum drift speed and timescale, Crida gap metric, gap depth, Toomre Q |

---

## ⚠️ Scope & limitations

This is a **visualisation-first** project. The approximations below are deliberate, and it is worth knowing where they are:

- **The gas is analytic, not simulated.** There is no hydrodynamics, so the planet does not carve a gap in the gas. The Crida and Kanagawa numbers are *diagnostics*; they are not fed back into the gas density.
- **Dust are test particles.** There is no dust back-reaction and no dust-dust interaction. The planet follows a prescribed orbit (eccentricity is geometric, and migration is a constant inward rate rather than a torque calculation).
- **Stokes number is a scaled prescription**, controlled by the "Stokes No. Scale" slider. The "Base Sigma_0" slider affects the telemetry (density, Toomre Q), not the particle dynamics.
- **Telemetry uses the analytic temperature profile**, so its aspect ratio does not follow the "(H/r)₀" slider that drives the particle dynamics. They agree at the default settings.
- **The spectral views are stylised colour maps** (see the table above), not synthetic observations. Features such as the ring near 6.4 AU and the spiral pattern are hand-placed analytic shapes, and the UV/Hα glow is schematic.
- **No radiative transfer, chemistry, self-gravity or magnetic fields.**
- **The disk is compact** (0.6-18 AU) compared with observed ALMA disks, which typically extend over 100 AU or more.

---

## 🛣️ Roadmap

- Feed the Kanagawa gap profile back into the gas so pressure bumps form and trap dust, giving rings that emerge rather than being painted
- Explicit mm-cm grain populations, and validation of drift speeds against the analytic solution
- Enable the implemented procedural Milky Way / catalogue-star backdrop and bloom passes in the live view
- Dust back-reaction and multi-planet systems
- Instrument-specific synthetic observations and simple radiative transfer
- Data export and plotting tools

---

## 🗂️ Code map

The project is a single file, `cosmic_gas_disc.py` (Python + Taichi).

| Piece | Role |
| --- | --- |
| `initialize_disk` | Samples radii, MRN grain sizes, vertical settling and drag-coupled initial velocities on the GPU |
| `advance_particles_symplectic` | Gravity, Epstein-type drag, stochastic vertical kick, boundary recycling |
| `compute_shader_colors` | The five spectral-view colour maps |
| `update_guide_overlays` | Orbit, Hill sphere, resonance, sublimation rings and L1-L5 |
| `get_astrophysics_diagnostics` | CPU-side telemetry equations |
| `OrbitalCamera` | Orbit / zoom camera with eased bookmark transitions |
| `main` | Event loop, GUI panels, click-to-inspect |

---

## 📚 References

- Andrews et al. 2018, *ApJL* 869, L41. ALMA DSHARP
- Ballesteros 2012, *EPL* 97, 34008. B-V colour index to temperature
- Chiang & Goldreich 1997, *ApJ* 490, 368. Irradiated flared disks
- Crida, Morbidelli & Masset 2006, *Icarus* 181, 587. Gap-opening criterion
- Dubrulle, Morfill & Sterzik 1995, *Icarus* 114, 237. Dust settling and turbulence
- Eggleton 1983, *ApJ* 268, 368. Roche-lobe approximation
- Hayashi 1981, *Prog. Theor. Phys. Suppl.* 70, 35. Minimum-mass solar nebula
- Hoffleit & Jaschek 1991. *The Bright Star Catalogue* (Yale, 5th ed.)
- Kanagawa et al. 2015, *ApJL* 806, L15. Gap-depth scaling
- Mathis, Rumpl & Nordsieck 1977, *ApJ* 217, 425. MRN grain-size distribution
- Nakagawa, Sekiya & Hayashi 1986, *Icarus* 67, 375. Drag-coupled drift
- Shakura & Sunyaev 1973, *A&A* 24, 337. α-viscosity
- Toomre 1964, *ApJ* 139, 1217. Gravitational stability
- Weidenschilling 1977, *MNRAS* 180, 57. Aerodynamics of solids in the solar nebula

---


<div align="center">

*One disk. Five wavelengths. Everything you can drag, pause and poke.* 🌌

</div>
