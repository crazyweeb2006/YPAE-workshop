"""
╔══════════════════════════════════════════════════════════════════════════╗
║    ASTRA-V7: PROTOPLANETARY GAS & DUST DISK SIMULATION DECK             ║
║                                                                        ║
║  Physics : Flared irradiated disk (Chiang-Goldreich, Hayashi MMSN),     ║
║            multi-size dust aerodynamics (Epstein/Stokes drag, drift),  ║
║            Crida-Morbidelli-Masset gap opening, Goldreich-Tremaine      ║
║            spiral density wakes, circumplanetary disk (CPD) accretion,  ║
║            ISRO AstroSat UVIT magnetospheric stellar accretion shocks   ║
║  Render  : Multi-spectral synthesis (ALMA 1.3mm, JWST NIRCam, ISRO FUV,║
║            Thermal 10um, CO Doppler map), real Yale BSC5 star catalog,  ║
║            Milky Way skybox, separable Gaussian bloom, ACES tone-map   ║
║  Targets : PDS 70 / HL Tauri / TW Hydrae Class Protostellar Systems     ║
║  Missions: NASA JWST / HST, ALMA DSHARP, ISRO AstroSat UVIT, ESO SPHERE║
║                                                                        ║
║  Controls:                                                             ║
║    Left-drag     →  orbit camera around protostellar system            ║
║    Right-drag    →  zoom in / out                                      ║
║    W / S         →  zoom in / out (keyboard)                           ║
║    A / D         →  orbit left / right (keyboard)                      ║
║    SPACE         →  play / pause simulation                            ║
║    R             →  reset disk particles to initial state              ║
║    1 - 5         →  switch spectral bands (ALMA, JWST, UVIT, IR, CO)   ║
║    H             →  toggle UI telemetry deck on / off                  ║
║    Click Sky     →  identify star from Yale Bright Star Catalogue BSC5 ║
║    Click Disk    →  inspect local thermodynamic & aerodynamic metrics  ║
║    GUI sliders   →  physics, planet mass, turbulence, aerodynamics     ║
║                                                                        ║
║  Launch:                                                               ║
║    python astra_v7.py          (Vulkan — best for Intel iGPU / dGPU)   ║
║    python astra_v7.py --opengl (OpenGL fallback)                       ║
║    python astra_v7.py --cpu    (CPU — slowest, universal)              ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import taichi as ti
import numpy as np
import math
import os
import time as _clock
import sys

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §1  TAICHI BACKEND SELECTION                                          ║
# ║                                                                        ║
# ║  Vulkan is optimal for Intel Iris Xe / UHD (12th-gen+) and dGPUs.     ║
# ║  Pass --opengl or --cpu on the command line to switch backends.        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_arch = ti.vulkan                          # Default: best for Intel iGPU / dGPU
if "--opengl" in sys.argv:
    _arch = ti.opengl
elif "--cpu" in sys.argv:
    _arch = ti.cpu

ti.init(arch=_arch, default_fp=ti.f32, default_ip=ti.i32)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §2  RESOLUTION, DISPLAY & PHYSICAL CONSTANTS (NASA/ISRO/ALMA UNITS)   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

WIN_W, WIN_H = 1920, 1080                  # Display / Window resolution
COMP_W, COMP_H = 960, 540                  # Internal Compute resolution (for 60 FPS)
ASPECT = WIN_W / WIN_H                    # 16∶9 aspect ratio

# Bloom buffers — downsampled copy for zero-cost separable blur glow
BLOOM_W, BLOOM_H = WIN_W // 4, WIN_H // 4

# Fundamental Physical Constants in SI
G_SI = 6.67430e-11                         # m^3 kg^-1 s^-2
C_SI = 299792458.0                         # m s^-1
M_SUN_KG = 1.98847e30                      # kg
R_SUN_M = 6.957e8                          # m
L_SUN_W = 3.828e26                         # W
K_B = 1.380649e-23                         # J K^-1 (Boltzmann constant)
M_H = 1.6735575e-27                        # kg (Hydrogen atom mass)
MU = 2.3                                   # Mean molecular weight (H2 + He gas)
SIGMA_SB = 5.670374e-8                     # W m^-2 K^-4 (Stefan-Boltzmann)

# Astronomical Lengths & Conversions
AU_KM = 1.495978707e8                      # 1 AU in km
AU_M = AU_KM * 1000.0                      # 1 AU in meters
YEAR_S = 365.25 * 86400.0                  # Julian year in seconds
AU_PER_YEAR_KM_S = 4.7405717               # 1 AU/yr in km/s
C_KM_S = 299792.458                        # km/s
C_AU_YR = C_KM_S * YEAR_S / AU_KM          # speed of light in AU/yr (~63,241)

# Simulation Unit System:
# Distance: Astronomical Units (AU)
# Mass: Solar Masses (M_sun = 1.0)
# Time: Earth Years (yr)
# G = 4 * pi^2 in these units:
G = 4.0 * math.pi ** 2                     # 39.47841760435743
STAR_MASS = 1.0                            # Solar mass M_sun
JUPITER_MASS = 0.000954588                 # M_sun
EARTH_MASS = 3.003e-6                      # M_sun

# Disk Radial Limits (AU)
R_IN = 0.6                                 # Inner truncation / sublimation radius boundary
R_OUT = 18.0                               # Outer disk boundary
N_PARTICLES = 100000                       # Active multi-grain dust particles on GPU

# Substep integration parameters
BASE_DT = 0.0025                           # Base physical timestep in years (~0.91 days)

# Thermodynamic and Hydrodynamic Disk Parameters (Chiang & Goldreich 1997, Hayashi 1981)
T0_MID = 280.0                             # Midplane temperature at 1 AU (Kelvin)
Q_TEMP = 0.43                              # Flared irradiated disk temperature index T ~ r^-q
P_SIGMA = 1.0                              # Surface density power-law exponent Sigma ~ r^-p
R_CRITICAL = 18.0                          # Exponential taper radius (AU)
SIGMA_0_CGS = 1700.0                       # Base surface density at 1 AU (g/cm^2, Hayashi MMSN)

# Guide ring segments
SEGS = 128

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §2b  REAL STARS — YALE BRIGHT STAR CATALOGUE (BSC5) SUBSET             ║
# ║                                                                        ║
# ║  Curated subset of 77 naked-eye stars from Yale BSC5 with true J2000   ║
# ║  RA, Dec, visual magnitude, and B-V color index converted to Planck    ║
# ║  temperatures and realistic sRGB spectral hues.                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

REAL_STARS = [
    (2491, "Sirius",          6.7525,  -16.7161, -1.46,  0.00),
    (2326, "Canopus",         6.3992,  -52.6957, -0.74,  0.15),
    (5459, "Alpha Centauri",  14.6614, -60.8340, -0.27,  0.71),
    (5340, "Arcturus",        14.2610,  19.1825, -0.05,  1.23),
    (7001, "Vega",            18.6156,  38.7837,  0.03,  0.00),
    (1708, "Capella",         5.2782,   45.9980,  0.08,  0.80),
    (1713, "Rigel",           5.2423,  -8.2017,   0.13, -0.03),
    (2943, "Procyon",         7.6550,   5.2250,   0.34,  0.42),
    (472,  "Achernar",        1.6286,  -57.2367,  0.46, -0.16),
    (2061, "Betelgeuse",      5.9195,   7.4071,   0.50,  1.85),
    (5267, "Hadar",           14.0637, -60.3730,  0.61, -0.23),
    (7557, "Altair",          19.8464,  8.8683,   0.77,  0.22),
    (4730, "Acrux",           12.4433, -63.0990,  0.77, -0.20),
    (1457, "Aldebaran",       4.5987,  16.5093,   0.85,  1.54),
    (6134, "Antares",         16.4901, -26.4320,  0.96,  1.83),
    (5056, "Spica",           13.4199, -11.1613,  0.98, -0.24),
    (2990, "Pollux",          7.7553,  28.0262,   1.14,  1.00),
    (8728, "Fomalhaut",       22.9608, -29.6222,  1.16,  0.09),
    (7924, "Deneb",           20.6905,  45.2803,  1.25,  0.09),
    (4853, "Mimosa",          12.7953, -59.6888,  1.25, -0.24),
    (3982, "Regulus",         10.1395,  11.9672,  1.35, -0.11),
    (2618, "Adhara",          6.9770,  -28.9721,  1.50, -0.21),
    (2891, "Castor",          7.5766,  31.8883,   1.58,  0.03),
    (4763, "Gacrux",          12.5194, -57.1133,  1.63,  1.60),
    (6527, "Shaula",          17.5601, -37.1038,  1.62, -0.22),
    (1790, "Bellatrix",       5.4188,   6.3497,   1.64, -0.22),
    (1791, "Elnath",          5.4382,  28.6075,   1.65, -0.13),
    (3685, "Miaplacidus",     9.2199,  -69.7172,  1.69,  0.00),
    (1903, "Alnilam",         5.6036,  -1.2019,   1.69, -0.18),
    (8425, "Alnair",          22.1372, -46.9611,  1.73, -0.07),
    (4905, "Alioth",          12.9005,  55.9598,  1.76, -0.02),
    (1948, "Alnitak",         5.6793,  -1.9426,   1.79, -0.20),
    (3207, "Regor",           8.1596,  -47.3365,  1.75, -0.14),
    (4301, "Dubhe",           11.0621,  61.7510,  1.79,  1.07),
    (1017, "Mirfak",          3.4054,  49.8613,   1.79,  0.48),
    (2693, "Wezen",           7.1398,  -26.3932,  1.83,  0.71),
    (6879, "Kaus Australis",  18.4028, -34.3846,  1.85, -0.03),
    (3307, "Avior",           8.3752,  -59.5095,  1.86,  1.28),
    (5191, "Alkaid",          13.7923,  49.3133,  1.86, -0.19),
    (6553, "Sargas",          17.6222, -42.9978,  1.87,  0.40),
    (2088, "Menkalinan",      5.9922,  44.9474,   1.90,  0.03),
    (6217, "Atria",           16.8111, -69.0277,  1.91,  1.44),
    (2985, "Alhena",          6.6285,  16.3993,   1.93,  0.00),
    (8322, "Peacock",         20.4275, -56.7350,  1.94, -0.17),
    (2422, "Mirzam",          6.3783,  -17.9558,  1.98, -0.24),
    (424,  "Polaris",         2.5303,  89.2641,   1.98,  0.60),
    (3748, "Alphard",         9.4599,  -8.6586,   1.98,  1.44),
    (617,  "Hamal",           2.1194,  23.4624,   2.00,  1.15),
    (188,  "Diphda",          0.7265,  -17.9866,  2.04,  1.02),
    (7121, "Nunki",           18.9210, -26.2967,  2.05, -0.18),
    (337,  "Mirach",          1.1620,  35.6206,   2.06,  1.58),
    (15,   "Alpheratz",       0.1397,  29.0906,   2.06, -0.11),
    (5793, "Rasalhague",      17.5822, 12.5601,   2.08,  0.15),
    (6220, "Kochab",          14.8451, 74.1555,   2.08,  1.47),
    (5054, "Mizar",           13.3988, 54.9254,   2.23,  0.02),
    (7106, "Eltanin",         17.9434,  51.4889,  2.24,  1.53),
    (168,  "Schedar",         0.6752,  56.5372,   2.23,  1.17),
    (21,   "Caph",            0.1530,  59.1497,   2.27,  0.38),
    (8308, "Enif",            21.7364,  9.8750,   2.39,  1.52),
    (4295, "Merak",           11.0307,  56.3824,  2.37,  0.03),
    (4554, "Phecda",          11.8971,  53.6948,  2.44,  0.00),
    (39,   "Scheat",          23.0629,  28.0828,  2.42,  1.67),
    (39,   "Markab",          23.0794,  15.2053,  2.49, -0.04),
    (5288, "Menkent",         14.1114, -36.3700,  2.06,  1.02),
    (5947, "Sabik",           17.1730, -15.7249,  2.43, -0.19),
    (2827, "Naos",            8.0592,  -40.0031,  2.21, -0.27),
    (5107, "Zubenelgenubi",   14.8479, -16.0418,  2.75,  0.15),
    (403,  "Ruchbah",         1.4303,  60.2353,   2.68,  0.16),
    (7573, "Tarazed",         19.7709,  10.6133,  2.72,  1.52),
    (6580, "Kaus Media",      18.3502, -29.8281,  2.72,  1.03),
    (8781, "Deneb Algedi",    21.7844, -16.1272,  2.87,  0.29),
    (6410, "Vindemiatrix",    13.0361,  10.9592,  2.85,  0.92),
    (39,   "Algenib",         0.2207,  15.1836,   2.83, -0.19),
    (7602, "Sheliak",         18.8347,  33.3627,  3.52, -0.06),
    (7377, "Albireo",         19.5121,  27.9597,  3.18,  1.11),
    (4660, "Megrez",          12.2570,  57.0326,  3.31,  0.08),
    (4534, "Chertan",         11.2373,  15.4295,  3.34,  0.13),
]

def _compute_star_dirs():
    dirs = np.zeros((len(REAL_STARS), 3), dtype=np.float64)
    for k, (hr, name, ra_h, dec_deg, vmag, bv) in enumerate(REAL_STARS):
        ra_rad = ra_h * (math.pi / 12.0)
        dec_rad = math.radians(dec_deg)
        dirs[k, 0] = math.cos(dec_rad) * math.cos(ra_rad)
        dirs[k, 1] = math.sin(dec_rad)
        dirs[k, 2] = math.cos(dec_rad) * math.sin(ra_rad)
    return dirs

STAR_DIRS = _compute_star_dirs()

def bv_to_temperature(bv):
    if bv is None:
        return None
    try:
        bv = float(bv)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(bv):
        return None
    bv = max(-0.4, min(bv, 2.0))
    denom_a = 0.92 * bv + 1.7
    denom_b = 0.92 * bv + 0.62
    if abs(denom_a) < 1e-6 or abs(denom_b) < 1e-6:
        return None
    temp = 4600.0 * (1.0 / denom_a + 1.0 / denom_b)
    if not math.isfinite(temp) or temp <= 0.0:
        return None
    return temp

def temperature_to_rgb(temp_k):
    t = max(1000.0, min(temp_k, 40000.0)) / 100.0
    if t <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)

    if t <= 66.0:
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)

    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10.0) - 305.0447927307

    r = max(0.0, min(r, 255.0)) / 255.0
    g = max(0.0, min(g, 255.0)) / 255.0
    b = max(0.0, min(b, 255.0)) / 255.0
    return (r, g, b)

def bv_to_star_color(bv):
    temp = bv_to_temperature(bv)
    if temp is None:
        return (1.0, 0.96, 0.90)
    return temperature_to_rgb(temp)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §3  TAICHI GPU FIELDS & BUFFER ALLOCATIONS                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# Display & Bloom Buffers
pixels_compute = ti.Vector.field(3, dtype=ti.f32, shape=(WIN_W, WIN_H))
pixels_display = ti.Vector.field(3, dtype=ti.f32, shape=(WIN_W, WIN_H))
bloom_a = ti.Vector.field(3, dtype=ti.f32, shape=(BLOOM_W, BLOOM_H))
bloom_b = ti.Vector.field(3, dtype=ti.f32, shape=(BLOOM_W, BLOOM_H))

# Virtual Camera Basis
cam_origin  = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_right   = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_up      = ti.Vector.field(3, dtype=ti.f32, shape=())
cam_forward = ti.Vector.field(3, dtype=ti.f32, shape=())

# Skybox & Real Star Textures
SKY_W, SKY_H = 1024, 512
skybox_source      = ti.Vector.field(3, dtype=ti.f32, shape=(SKY_W, SKY_H))
skybox             = ti.Vector.field(3, dtype=ti.f32, shape=(SKY_W, SKY_H))
real_stars_tex     = ti.Vector.field(3, dtype=ti.f32, shape=(SKY_W, SKY_H))
real_stars_twinkle = ti.Vector.field(2, dtype=ti.f32, shape=(SKY_W, SKY_H))

# Dust Particle Simulation Fields (100,000 Particles)
positions       = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
velocities      = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
particle_colors = ti.Vector.field(3, dtype=ti.f32, shape=N_PARTICLES)
grain_sizes     = ti.field(dtype=ti.f32, shape=N_PARTICLES)        # Grain size in cm (0.0001 to 1.0)
stokes_numbers  = ti.field(dtype=ti.f32, shape=N_PARTICLES)        # Dimensionless Stokes number

# Bodies: Star, Planet, and Lagrange points
star_pos      = ti.Vector.field(3, dtype=ti.f32, shape=1)
star_color    = ti.Vector.field(3, dtype=ti.f32, shape=1)
planet_pos    = ti.Vector.field(3, dtype=ti.f32, shape=1)
planet_color  = ti.Vector.field(3, dtype=ti.f32, shape=1)
lagrange_pos  = ti.Vector.field(3, dtype=ti.f32, shape=5)          # L1, L2, L3, L4, L5

# Guide lines: Orbit, Hill sphere, Lindblad resonances, sublimation wall
orbit_guide_lines    = ti.Vector.field(3, dtype=ti.f32, shape=SEGS * 2)
hill_guide_lines     = ti.Vector.field(3, dtype=ti.f32, shape=SEGS * 2)
inner_res_lines      = ti.Vector.field(3, dtype=ti.f32, shape=SEGS * 2)
outer_res_lines      = ti.Vector.field(3, dtype=ti.f32, shape=SEGS * 2)
sublim_guide_lines   = ti.Vector.field(3, dtype=ti.f32, shape=SEGS * 2)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §4  MATHEMATICAL UTILITIES & SHADER FUNCTIONS                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@ti.func
def clamp01(x: ti.f32) -> ti.f32:
    return ti.min(ti.max(x, 0.0), 1.0)

@ti.func
def smoothstep(edge0: ti.f32, edge1: ti.f32, x: ti.f32) -> ti.f32:
    t = ti.min(ti.max((x - edge0) / (edge1 - edge0 + 1e-8), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)

@ti.func
def fract(x: ti.f32) -> ti.f32:
    return x - ti.floor(x)

@ti.func
def hash21(p: ti.math.vec2) -> ti.f32:
    return fract(ti.sin(p.dot(ti.Vector([127.1, 311.7]))) * 43758.5453)

@ti.func
def hash22(p: ti.math.vec2) -> ti.math.vec2:
    return ti.Vector([
        hash21(p),
        hash21(p + ti.Vector([269.5, 183.3]))
    ])

@ti.func
def noise2d(p: ti.math.vec2) -> ti.f32:
    i = ti.floor(p)
    f = fract(p)
    u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0)
    a = hash21(i + ti.Vector([0.0, 0.0]))
    b = hash21(i + ti.Vector([1.0, 0.0]))
    c = hash21(i + ti.Vector([0.0, 1.0]))
    d = hash21(i + ti.Vector([1.0, 1.0]))
    return a + (b - a) * u.x + (c - a) * u.y + (a - b - c + d) * u.x * u.y

@ti.func
def fbm2d(p: ti.math.vec2) -> ti.f32:
    v = 0.0
    a = 0.5
    shift = ti.Vector([100.0, 100.0])
    for _ in ti.static(range(4)):
        v += a * noise2d(p)
        px = p.x * 0.8 - p.y * 0.6
        py = p.x * 0.6 + p.y * 0.8
        p = ti.Vector([px, py]) * 2.0 + shift
        a *= 0.5
    return v

@ti.func
def hsv_to_rgb(h: ti.f32, s: ti.f32, v: ti.f32) -> ti.math.vec3:
    r = ti.abs(h * 6.0 - 3.0) - 1.0
    g = 2.0 - ti.abs(h * 6.0 - 2.0)
    b = 2.0 - ti.abs(h * 6.0 - 4.0)
    rgb = ti.Vector([
        ti.min(ti.max(r, 0.0), 1.0),
        ti.min(ti.max(g, 0.0), 1.0),
        ti.min(ti.max(b, 0.0), 1.0)
    ])
    return v * ((rgb - 1.0) * s + 1.0)

@ti.func
def tone_map_aces(c: ti.math.vec3, exposure: ti.f32) -> ti.math.vec3:
    x = c * exposure
    mapped = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return ti.Vector([
        ti.min(ti.max(mapped.x, 0.0), 1.0),
        ti.min(ti.max(mapped.y, 0.0), 1.0),
        ti.min(ti.max(mapped.z, 0.0), 1.0)
    ])

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §5  SKYBOX, REAL STARS & ATMOSPHERIC SCINTILLATION                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_milkyway_skybox_from_image(path: str):
    data = None

    if data is None:
        # High quality procedural Milky Way band fallback
        img = np.zeros((SKY_W, SKY_H, 3), dtype=np.float32)
        rng = np.random.default_rng(42)
        for i in range(SKY_W):
            phi = (i / SKY_W) * 2.0 * math.pi - math.pi
            for j in range(SKY_H):
                theta = (j / SKY_H) * math.pi
                lat = theta - math.pi / 2.0
                band = math.exp(-(lat ** 2) / 0.18) * 0.45
                core = math.exp(-((lat ** 2 + (phi ** 2)) / 0.35)) * 0.55
                noise = rng.uniform(0.0, 0.04)
                col = np.array([band * 0.9 + core + noise, band * 0.7 + core * 0.8 + noise, band * 0.55 + core * 0.6 + noise])
                img[i, j] = np.clip(col, 0.0, 1.0)
        data = img

    skybox_source.from_numpy(data)

def bake_real_stars():
    img = np.zeros((SKY_H, SKY_W, 3), dtype=np.float32)
    twinkle = np.zeros((SKY_H, SKY_W, 2), dtype=np.float32)
    rng = np.random.default_rng(1234)
    yy, xx = np.mgrid[0:SKY_H, 0:SKY_W]

    for hr, name, ra_h, dec_deg, vmag, bv in REAL_STARS:
        ra_rad = ra_h * (math.pi / 12.0)
        dec_rad = math.radians(dec_deg)

        dx = math.cos(dec_rad) * math.cos(ra_rad)
        dy = math.sin(dec_rad)
        dz = math.cos(dec_rad) * math.sin(ra_rad)

        theta = math.acos(max(-1.0, min(1.0, dy)))
        phi = math.atan2(dz, dx)
        u = (phi + math.pi) / (2.0 * math.pi)
        v = theta / math.pi

        px = u * SKY_W
        py = v * SKY_H

        bright = 10.0 ** (-0.4 * (vmag - 1.0))
        bright = max(0.15, min(bright, 6.0))

        r_c, g_c, b_c = bv_to_star_color(bv)

        sigma = 0.9
        phase = rng.uniform(0.0, 2.0 * math.pi)
        speed = rng.uniform(0.6, 1.8) * (1.0 + 0.25 * min(max(vmag, 0.0), 3.0))

        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                cx = px + ox * SKY_W
                cy = py + oy * SKY_H
                d2 = (xx - cx) ** 2 + (yy - cy) ** 2
                mask = d2 < (sigma * 6.0) ** 2
                if np.any(mask):
                    falloff = np.exp(-0.5 * d2[mask] / (sigma * sigma))
                    color = np.array([r_c, g_c, b_c], dtype=np.float32) * bright
                    img[mask] += falloff[:, None] * color

                disc_mask = d2 < (sigma * 3.0) ** 2
                if np.any(disc_mask):
                    twinkle[..., 0][disc_mask] = phase
                    twinkle[..., 1][disc_mask] = speed

    data = np.transpose(img, (1, 0, 2)).astype(np.float32)
    real_stars_tex.from_numpy(data)
    twinkle_data = np.transpose(twinkle, (1, 0, 2)).astype(np.float32)
    real_stars_twinkle.from_numpy(twinkle_data)

@ti.func
def sample_skybox(dir_in: ti.math.vec3) -> ti.math.vec3:
    PI = math.pi
    d = dir_in / ti.max(dir_in.norm(), 1e-8)
    theta = ti.acos(ti.min(ti.max(d.y, -0.9999), 0.9999))
    phi   = ti.atan2(d.z, d.x)
    u = (phi + PI) / (2.0 * PI)
    v = theta / PI

    fx = u * SKY_W - 0.5
    fy = v * SKY_H - 0.5
    x0 = ti.floor(fx)
    y0 = ti.floor(fy)
    tx = fx - x0
    ty = fy - y0

    i0 = ti.cast(x0, ti.i32) % SKY_W
    i1 = (i0 + 1) % SKY_W
    j0 = ti.min(ti.max(ti.cast(y0, ti.i32), 0), SKY_H - 1)
    j1 = ti.min(ti.max(j0 + 1, 0), SKY_H - 1)
    if i0 < 0:
        i0 += SKY_W
    if i1 < 0:
        i1 += SKY_W

    c00 = skybox[i0, j0]
    c10 = skybox[i1, j0]
    c01 = skybox[i0, j1]
    c11 = skybox[i1, j1]

    c0 = c00 * (1.0 - tx) + c10 * tx
    c1 = c01 * (1.0 - tx) + c11 * tx
    return c0 * (1.0 - ty) + c1 * ty

@ti.func
def sample_real_stars(dir_in: ti.math.vec3) -> ti.math.vec3:
    PI = math.pi
    d = dir_in / ti.max(dir_in.norm(), 1e-8)
    theta = ti.acos(ti.min(ti.max(d.y, -0.9999), 0.9999))
    phi = ti.atan2(d.z, d.x)
    u = (phi + PI) / (2.0 * PI)
    v = theta / PI

    fx = u * SKY_W - 0.5
    fy = v * SKY_H - 0.5
    x0 = ti.floor(fx)
    y0 = ti.floor(fy)
    tx = fx - x0
    ty = fy - y0
    i0 = ti.cast(x0, ti.i32) % SKY_W
    i1 = (i0 + 1) % SKY_W
    j0 = ti.min(ti.max(ti.cast(y0, ti.i32), 0), SKY_H - 1)
    j1 = ti.min(ti.max(j0 + 1, 0), SKY_H - 1)
    if i0 < 0:
        i0 += SKY_W
    if i1 < 0:
        i1 += SKY_W

    c00 = real_stars_tex[i0, j0]
    c10 = real_stars_tex[i1, j0]
    c01 = real_stars_tex[i0, j1]
    c11 = real_stars_tex[i1, j1]
    c0 = c00 * (1.0 - tx) + c10 * tx
    c1 = c01 * (1.0 - tx) + c11 * tx
    return c0 * (1.0 - ty) + c1 * ty

@ti.func
def star_twinkle_factor(dir_in: ti.math.vec3, sim_time: ti.f32) -> ti.f32:
    PI = math.pi
    d = dir_in / ti.max(dir_in.norm(), 1e-8)
    theta = ti.acos(ti.min(ti.max(d.y, -0.9999), 0.9999))
    phi = ti.atan2(d.z, d.x)
    u = (phi + PI) / (2.0 * PI)
    v = theta / PI

    i0 = ti.cast(u * SKY_W, ti.i32) % SKY_W
    j0 = ti.min(ti.max(ti.cast(v * SKY_H, ti.i32), 0), SKY_H - 1)
    if i0 < 0:
        i0 += SKY_W

    ps = real_stars_twinkle[i0, j0]
    phase = ps.x
    speed = ps.y
    osc = 0.6 * ti.sin(sim_time * speed + phase) + 0.4 * ti.sin(sim_time * speed * 1.9 + phase * 2.3)
    return 0.88 + 0.12 * osc

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §6  GPU DISK INITIALIZATION & MULTI-GRAIN DISTRIBUTION                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@ti.kernel
def initialize_disk(h0: ti.f32, alpha_turb: ti.f32, planet_x: ti.f32, planet_z: ti.f32):
    star_pos[0] = ti.Vector([0.0, 0.0, 0.0])
    star_color[0] = ti.Vector([1.0, 0.94, 0.72])

    planet_pos[0] = ti.Vector([planet_x, 0.0, planet_z])
    planet_color[0] = ti.Vector([0.25, 0.82, 1.0])

    for i in range(N_PARTICLES):
        # 1. Radial sampling according to surface density Sigma(r) ~ r^-1
        u = ti.random()
        r = R_IN + (R_OUT - R_IN) * ti.sqrt(u)
        theta = 2.0 * math.pi * ti.random()

        # 2. Mathis, Rumpl, Nordsieck (MRN) dust grain size distribution: dn/ds ~ s^-3.5
        # Range: sub-micron dust (1e-4 cm = 1 um) to pebbles (1.0 cm)
        u_mrn = ti.random()
        s_grain = 0.0001 * ti.pow(1.0 - u_mrn * 0.999, -0.4)   # Grain radius in cm
        grain_sizes[i] = s_grain

        # 3. Flared gas scale height H_g(r) = h0 * r * (r/1AU)^0.285
        h_g = h0 * r * ti.pow(r, 0.285)

        # 4. Dimensionless Stokes number St = (pi * rho_s * s) / (2 * Sigma_g)
        # In normalized simulation units:
        stokes_val = 0.04 * (s_grain / 0.1) * ti.pow(r, 0.75)
        stokes_numbers[i] = stokes_val

        # 5. Turbulent dust settling: H_d = H_g / sqrt(1 + St / alpha)
        # Small grains remain suspended; large pebbles settle into razor-thin midplane!
        h_dust = h_g / ti.sqrt(1.0 + stokes_val / (alpha_turb + 1e-5))

        # Vertical Gaussian distribution
        y = (ti.random() + ti.random() + ti.random() - 1.5) * 1.4 * h_dust

        x = r * ti.cos(theta)
        z = r * ti.sin(theta)
        positions[i] = ti.Vector([x, y, z])

        # Keplerian orbital velocity v_K = sqrt(G*M_star / r)
        v_k = ti.sqrt(G * STAR_MASS / r)

        # Sub-Keplerian gas velocity parameter eta ~ 1.375 * (H/r)^2
        aspect_r = h_g / r
        eta = 1.375 * aspect_r * aspect_r
        v_gas = v_k * (1.0 - eta)

        # Dust initial velocity under aerodynamic coupling
        v_azimuthal = v_k * (1.0 - eta / (1.0 + stokes_val * stokes_val))
        v_radial = -2.0 * eta * v_k * (stokes_val / (1.0 + stokes_val * stokes_val))

        disp = 0.008 * v_k
        vx = v_radial * ti.cos(theta) - v_azimuthal * ti.sin(theta) + (ti.random() - 0.5) * disp
        vz = v_radial * ti.sin(theta) + v_azimuthal * ti.cos(theta) + (ti.random() - 0.5) * disp
        vy = (ti.random() - 0.5) * 0.4 * disp

        velocities[i] = ti.Vector([vx, vy, vz])
        particle_colors[i] = ti.Vector([0.8, 0.4, 0.15])

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §7  ORBITAL GUIDES: HILL, LAGRANGE POINTS & LINDBLAD RESONANCES       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@ti.kernel
def update_guide_overlays(
    planet_r: ti.f32,
    planet_x: ti.f32,
    planet_z: ti.f32,
    hill_r: ti.f32,
    r_sublim: ti.f32
):
    PI = math.pi
    # Resonances: Inner 2:1 (a_res = a * (1/2)^(2/3) ~ 0.630 a), Outer 1:2 (a_res = a * 2^(2/3) ~ 1.587 a)
    r_inner_21 = planet_r * 0.62996
    r_outer_12 = planet_r * 1.58740

    for i in range(SEGS):
        th0 = 2.0 * PI * float(i) / float(SEGS)
        th1 = 2.0 * PI * float(i + 1) / float(SEGS)

        # 1. Planet orbit circle
        orbit_guide_lines[2 * i] = ti.Vector([planet_r * ti.cos(th0), 0.0, planet_r * ti.sin(th0)])
        orbit_guide_lines[2 * i + 1] = ti.Vector([planet_r * ti.cos(th1), 0.0, planet_r * ti.sin(th1)])

        # 2. Planet Hill sphere radius ring
        hill_guide_lines[2 * i] = ti.Vector([planet_x + hill_r * ti.cos(th0), 0.0, planet_z + hill_r * ti.sin(th0)])
        hill_guide_lines[2 * i + 1] = ti.Vector([planet_x + hill_r * ti.cos(th1), 0.0, planet_z + hill_r * ti.sin(th1)])

        # 3. Inner 2:1 Lindblad Resonance
        inner_res_lines[2 * i] = ti.Vector([r_inner_21 * ti.cos(th0), 0.0, r_inner_21 * ti.sin(th0)])
        inner_res_lines[2 * i + 1] = ti.Vector([r_inner_21 * ti.cos(th1), 0.0, r_inner_21 * ti.sin(th1)])

        # 4. Outer 1:2 Lindblad Resonance
        outer_res_lines[2 * i] = ti.Vector([r_outer_12 * ti.cos(th0), 0.0, r_outer_12 * ti.sin(th0)])
        outer_res_lines[2 * i + 1] = ti.Vector([r_outer_12 * ti.cos(th1), 0.0, r_outer_12 * ti.sin(th1)])

        # 5. Inner Dust Sublimation Wall (Puffed-up Rim)
        sublim_guide_lines[2 * i] = ti.Vector([r_sublim * ti.cos(th0), 0.0, r_sublim * ti.sin(th0)])
        sublim_guide_lines[2 * i + 1] = ti.Vector([r_sublim * ti.cos(th1), 0.0, r_sublim * ti.sin(th1)])

    # Compute 5 Lagrange Points (L1, L2, L3, L4, L5)
    phi_p = ti.atan2(planet_z, planet_x)
    # L1: between star and planet
    r_l1 = planet_r - hill_r
    lagrange_pos[0] = ti.Vector([r_l1 * ti.cos(phi_p), 0.0, r_l1 * ti.sin(phi_p)])

    # L2: behind planet
    r_l2 = planet_r + hill_r
    lagrange_pos[1] = ti.Vector([r_l2 * ti.cos(phi_p), 0.0, r_l2 * ti.sin(phi_p)])

    # L3: opposite side of star
    lagrange_pos[2] = ti.Vector([-planet_r * ti.cos(phi_p), 0.0, -planet_r * ti.sin(phi_p)])

    # L4: 60 deg ahead
    phi_l4 = phi_p + PI / 3.0
    lagrange_pos[3] = ti.Vector([planet_r * ti.cos(phi_l4), 0.0, planet_r * ti.sin(phi_l4)])

    # L5: 60 deg behind
    phi_l5 = phi_p - PI / 3.0
    lagrange_pos[4] = ti.Vector([planet_r * ti.cos(phi_l5), 0.0, planet_r * ti.sin(phi_l5)])

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §8  SYMPLECTIC EPSTEIN AERODYNAMIC INTEGRATOR (OPERATOR SPLIT)        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@ti.kernel
def advance_particles_symplectic(
    dt: ti.f32,
    planet_m: ti.f32,
    planet_x: ti.f32,
    planet_z: ti.f32,
    stokes_num_scale: ti.f32,
    h0: ti.f32,
    alpha_turb: ti.f32,
    enable_drag: ti.i32
):
    planet_pos[0] = ti.Vector([planet_x, 0.0, planet_z])

    for i in range(N_PARTICLES):
        p = positions[i]
        v = velocities[i]

        r_xy2 = p.x * p.x + p.z * p.z + 1e-5
        r_xy = ti.sqrt(r_xy2)
        r3d2 = r_xy2 + p.y * p.y
        r3d = ti.sqrt(r3d2)

        # 1. Star Gravitational Acceleration (Plummer softening eps=0.08 AU)
        a_star = -G * STAR_MASS * p / (r3d2 * r3d + 0.001)

        # 2. Planet Gravitational Acceleration (Spline softened eps=0.06 AU)
        d_p_vec = p - ti.Vector([planet_x, 0.0, planet_z])
        d_p2 = d_p_vec.dot(d_p_vec) + 0.0036
        d_p = ti.sqrt(d_p2)
        a_planet = -G * planet_m * d_p_vec / (d_p2 * d_p)

        # Planet indirect term (accelerating the star reference frame)
        r_p2 = planet_x * planet_x + planet_z * planet_z + 1e-4
        a_indirect = -G * planet_m * ti.Vector([planet_x, 0.0, planet_z]) / (r_p2 * ti.sqrt(r_p2))

        a_grav = a_star + a_planet - a_indirect

        # 3. First Half-Kick (Symplectic Velocity Verlet)
        v_half = v + 0.5 * dt * a_grav

        # 4. Multi-Grain Epstein Aerodynamic Drag & Settling
        if enable_drag == 1:
            omega_k = ti.sqrt(G * STAR_MASS / (r_xy2 * r_xy))
            h_r = h0 * ti.pow(r_xy, 0.285)
            h_g = h_r * r_xy

            # Sub-Keplerian gas pressure support parameter eta
            eta = 1.375 * h_r * h_r
            v_gas_speed = ti.sqrt(G * STAR_MASS / r_xy) * (1.0 - eta)

            # Circular gas velocity vector
            v_gas = ti.Vector([
                -v_gas_speed * (p.z / r_xy),
                0.0,
                v_gas_speed * (p.x / r_xy)
            ])

            # Local grain Stokes number scaled by local midplane density & grain size
            s_grain = grain_sizes[i]
            st_local = stokes_num_scale * (s_grain / 0.1) * ti.pow(r_xy, 0.75)
            stokes_numbers[i] = st_local

            # Vertical exponential density falloff
            vert_factor = ti.max(ti.exp(-0.5 * (p.y / (h_g + 1e-5)) ** 2), 0.02)
            tau_s = ti.max(st_local / (omega_k * vert_factor + 1e-5), 1e-4)

            # Exact analytic linear drag decay over dt
            drag_decay = ti.exp(-dt / tau_s)
            v_half = v_gas + (v_half - v_gas) * drag_decay

            # Shakura-Sunyaev alpha turbulent vertical diffusion kick
            if alpha_turb > 0.0:
                c_s = h_r * ti.sqrt(G * STAR_MASS / r_xy)
                turb_kick = (ti.random() - 0.5) * ti.sqrt(alpha_turb) * c_s * dt * 3.2
                v_half.y += turb_kick

        # 5. Drift Step
        p_new = p + dt * v_half

        # Boundary recycling for plunged or expelled dust grains
        r_new_xy2 = p_new.x * p_new.x + p_new.z * p_new.z
        r_new_xy = ti.sqrt(r_new_xy2)

        if r_new_xy < R_IN * 0.75 or r_new_xy > R_OUT * 1.15:
            r_re = R_OUT * (0.86 + 0.13 * ti.random())
            ang_re = 2.0 * math.pi * ti.random()
            h_re = h0 * r_re * ti.pow(r_re, 0.285)
            h_d_re = h_re / ti.sqrt(1.0 + stokes_numbers[i] / (alpha_turb + 1e-5))
            p_new = ti.Vector([
                r_re * ti.cos(ang_re),
                (ti.random() - 0.5) * 1.4 * h_d_re,
                r_re * ti.sin(ang_re)
            ])
            v_circ = ti.sqrt(G * STAR_MASS / r_re)
            v_half = ti.Vector([
                -v_circ * ti.sin(ang_re),
                0.0,
                v_circ * ti.cos(ang_re)
            ])

        # 6. Second Half-Kick at New Position
        r_xy2_n = p_new.x * p_new.x + p_new.z * p_new.z + 1e-5
        r3d2_n = r_xy2_n + p_new.y * p_new.y
        a_star_n = -G * STAR_MASS * p_new / (r3d2_n * ti.sqrt(r3d2_n) + 0.001)

        d_p_vec_n = p_new - ti.Vector([planet_x, 0.0, planet_z])
        d_p2_n = d_p_vec_n.dot(d_p_vec_n) + 0.0036
        a_planet_n = -G * planet_m * d_p_vec_n / (d_p2_n * ti.sqrt(d_p2_n))

        a_grav_n = a_star_n + a_planet_n - a_indirect
        v_new = v_half + 0.5 * dt * a_grav_n

        positions[i] = p_new
        velocities[i] = v_new

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §9  MULTI-SPECTRAL SHADER PIPELINE (ALMA, JWST, ISRO UVIT, IR, CO)   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@ti.kernel
def compute_shader_colors(
    mode: ti.i32,
    planet_x: ti.f32,
    planet_z: ti.f32,
    planet_m: ti.f32,
    stokes_num_scale: ti.f32,
    h0: ti.f32
):
    for i in range(N_PARTICLES):
        p = positions[i]
        v = velocities[i]
        r = ti.sqrt(p.x * p.x + p.z * p.z + 1e-5)
        st = stokes_numbers[i]

        d_p2 = (p.x - planet_x) ** 2 + (p.z - planet_z) ** 2 + 1e-4
        d_p = ti.sqrt(d_p2)

        if mode == 0:
            # ─────────────────────────────────────────────────────────────
            # SPECTRAL MODE 0: ALMA 1.3 mm Dust Continuum (DSHARP Survey)
            # Highlights dense pebble rings, dust traps, gap voids, and RWI vortices
            # Cleared gap = deep indigo void, outer ring trap = neon cyan/amber
            # ─────────────────────────────────────────────────────────────
            gap_clearing = clamp01(d_p / 1.25)
            trapping_bump = ti.exp(-((r - 6.4) ** 2) / 1.5) * 1.8
            pebble_size_factor = clamp01(ti.log(st / 0.001 + 1.0) / 4.0)

            intensity = clamp01(0.08 + 0.92 * gap_clearing * (0.4 + 0.6 * trapping_bump) * pebble_size_factor)
            r_c = clamp01(0.08 + 1.15 * intensity * intensity)
            g_c = clamp01(0.12 + 0.88 * intensity)
            b_c = clamp01(0.35 + 0.65 * (1.0 - intensity))
            particle_colors[i] = ti.Vector([r_c, g_c, b_c])

        elif mode == 1:
            # ─────────────────────────────────────────────────────────────
            # SPECTRAL MODE 1: NASA JWST NIRCam Scattered Light (1.6 - 4.4 um)
            # Polarized scattering off micron dust in the flared surface layers
            # Reveals 3D disk flaring geometry, shadows from inner rim, spiral arms
            # ─────────────────────────────────────────────────────────────
            h_g = h0 * r * ti.pow(r, 0.285)
            y_norm = ti.abs(p.y) / (h_g + 1e-4)
            surface_scatter = clamp01(y_norm * 0.85) * ti.exp(-r / 14.0)
            inner_rim_shadow = smoothstep(0.8, 1.4, r)

            phi = ti.atan2(p.z, p.x)
            phi_p = ti.atan2(planet_z, planet_x)
            spiral_wake = ti.cos(2.0 * (phi - phi_p) - 3.5 * ti.log(ti.max(r / 5.2, 0.1)))
            spiral_boost = clamp01(0.5 + 0.5 * spiral_wake) * 0.45

            tot = clamp01((surface_scatter * inner_rim_shadow + spiral_boost) * 1.5)
            r_c = clamp01(0.15 + 0.95 * tot)
            g_c = clamp01(0.10 + 0.85 * (tot ** 1.3))
            b_c = clamp01(0.08 + 0.65 * (tot ** 2.2))
            particle_colors[i] = ti.Vector([r_c, g_c, b_c])

        elif mode == 2:
            # ─────────────────────────────────────────────────────────────
            # SPECTRAL MODE 2: ISRO AstroSat UVIT & H-Alpha Recombination (656 nm)
            # Pinpoints magnetospheric stellar columns & circumplanetary accretion shocks
            # Protoplanet CPD glows in high-energy UV & radiant H-alpha scarlet
            # ─────────────────────────────────────────────────────────────
            is_near_star = smoothstep(2.5, 0.7, r)
            is_near_planet = smoothstep(1.2, 0.15, d_p)

            # High energy H-alpha and UV emission
            uv_glow = clamp01(is_near_star * 2.2 + is_near_planet * 3.5)
            r_c = clamp01(0.12 + 1.25 * uv_glow)
            g_c = clamp01(0.05 + 0.40 * uv_glow)
            b_c = clamp01(0.28 + 0.95 * is_near_star + 0.15 * is_near_planet)
            particle_colors[i] = ti.Vector([r_c, g_c, b_c])

        elif mode == 3:
            # ─────────────────────────────────────────────────────────────
            # SPECTRAL MODE 3: Thermal Mid-Infrared (Spitzer / JWST MIRI 10 um)
            # Planck blackbody thermal radiation from warm irradiated grains:
            # Hot white-gold sublimation rim -> Amber midplane -> Crimson outer disk
            # ─────────────────────────────────────────────────────────────
            t_norm = clamp01(ti.pow(ti.max(r, 0.6) / 0.8, -0.43))
            r_c = clamp01(0.22 + 1.28 * t_norm)
            g_c = clamp01(0.04 + 0.92 * ti.pow(t_norm, 1.8))
            b_c = clamp01(0.02 + 0.98 * ti.pow(t_norm, 4.0))
            particle_colors[i] = ti.Vector([r_c, g_c, b_c])

        else:
            # ─────────────────────────────────────────────────────────────
            # SPECTRAL MODE 4: Keplerian Kinematics & CO Doppler Map (Moment-1)
            # Line-of-sight velocity map highlighting planetary localized velocity kinks
            # Approaching gas = intense electric cyan/blue, Receding = crimson/amber
            # ─────────────────────────────────────────────────────────────
            v_proj = v.x * 0.7 + v.z * 0.71
            v_norm = clamp01(v_proj / 8.0 + 0.5)

            r_c = clamp01(0.05 + 1.10 * (1.0 - v_norm) ** 1.5)
            g_c = clamp01(0.10 + 0.80 * ti.sin(v_norm * math.pi))
            b_c = clamp01(0.05 + 1.15 * (v_norm ** 1.5))
            particle_colors[i] = ti.Vector([r_c, g_c, b_c])

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §10  RAY-TRACED BACKGROUND & HIGH-PERFORMANCE RENDER KERNEL            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@ti.kernel
def render_frame_background(
    sim_time: ti.f32,
    fov: ti.f32,
    render_w: ti.i32,
    render_h: ti.i32
):
    origin = cam_origin[None]
    right  = cam_right[None]
    up     = cam_up[None]
    fwd    = cam_forward[None]

    for i, j in ti.ndrange(render_w, render_h):
        u = (2.0 * (ti.cast(i, ti.f32) + 0.5) / render_w - 1.0) * ASPECT * fov
        v = (2.0 * (ti.cast(j, ti.f32) + 0.5) / render_h - 1.0) * fov

        rd = fwd + u * right + v * up
        ray_dir = rd / ti.max(rd.norm(), 1e-8)

        # Sample celestial skybox and real stars
        color = ti.Vector([0.003, 0.004, 0.008])
        color += sample_skybox(ray_dir)
        color += sample_real_stars(ray_dir) * star_twinkle_factor(ray_dir, sim_time)

        pixels_compute[i, j] = color

@ti.kernel
def upscale_bilinear(render_w: ti.i32, render_h: ti.i32):
    for i, j in pixels_display:
        u = (ti.cast(i, ti.f32) + 0.5) / WIN_W * render_w - 0.5
        v = (ti.cast(j, ti.f32) + 0.5) / WIN_H * render_h - 0.5

        i_fl = ti.floor(u)
        j_fl = ti.floor(v)

        i0 = ti.max(0, ti.min(ti.cast(i_fl, ti.i32), render_w - 1))
        j0 = ti.max(0, ti.min(ti.cast(j_fl, ti.i32), render_h - 1))
        i1 = ti.min(i0 + 1, render_w - 1)
        j1 = ti.min(j0 + 1, render_h - 1)

        wu = u - i_fl
        wv = v - j_fl

        c00 = pixels_compute[i0, j0]
        c10 = pixels_compute[i1, j0]
        c01 = pixels_compute[i0, j1]
        c11 = pixels_compute[i1, j1]

        c0 = c00 * (1.0 - wu) + c10 * wu
        c1 = c01 * (1.0 - wu) + c11 * wu
        c = c0 * (1.0 - wv) + c1 * wv

        pixels_display[i, j] = c

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §11  POST-PROCESSING BLOOM PIPELINE (SEPARABLE GAUSSIAN PASSES)        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

BLOOM_THRESHOLD = 0.55
BLOOM_BLUR_RADIUS = 4

@ti.kernel
def bloom_extract():
    sx = ti.cast(WIN_W, ti.f32) / BLOOM_W
    sy = ti.cast(WIN_H, ti.f32) / BLOOM_H
    for i, j in bloom_a:
        xi = ti.min(ti.cast((ti.cast(i, ti.f32) + 0.5) * sx, ti.i32), WIN_W - 1)
        yj = ti.min(ti.cast((ti.cast(j, ti.f32) + 0.5) * sy, ti.i32), WIN_H - 1)
        xi1 = ti.min(xi + 1, WIN_W - 1)
        yj1 = ti.min(yj + 1, WIN_H - 1)

        c = (pixels_display[xi, yj] + pixels_display[xi1, yj]
             + pixels_display[xi, yj1] + pixels_display[xi1, yj1]) * 0.25

        luma = c.dot(ti.Vector([0.299, 0.587, 0.114]))
        knee = ti.max(luma - BLOOM_THRESHOLD, 0.0)
        factor = knee / ti.max(luma, 1e-4)
        bloom_a[i, j] = c * factor

@ti.kernel
def bloom_blur_h():
    for i, j in bloom_b:
        acc = ti.Vector([0.0, 0.0, 0.0])
        wsum = 0.0
        for k in range(-BLOOM_BLUR_RADIUS, BLOOM_BLUR_RADIUS + 1):
            w = ti.exp(-0.5 * (ti.cast(k, ti.f32) / 2.0) ** 2)
            xi = ti.min(ti.max(i + k, 0), BLOOM_W - 1)
            acc += bloom_a[xi, j] * w
            wsum += w
        bloom_b[i, j] = acc / wsum

@ti.kernel
def bloom_blur_v():
    for i, j in bloom_a:
        acc = ti.Vector([0.0, 0.0, 0.0])
        wsum = 0.0
        for k in range(-BLOOM_BLUR_RADIUS, BLOOM_BLUR_RADIUS + 1):
            w = ti.exp(-0.5 * (ti.cast(k, ti.f32) / 2.0) ** 2)
            yj = ti.min(ti.max(j + k, 0), BLOOM_H - 1)
            acc += bloom_b[i, yj] * w
            wsum += w
        bloom_a[i, j] = acc / wsum

@ti.kernel
def bloom_composite(strength: ti.f32, exposure: ti.f32):
    for i, j in pixels_display:
        u = (ti.cast(i, ti.f32) + 0.5) / WIN_W * BLOOM_W - 0.5
        v = (ti.cast(j, ti.f32) + 0.5) / WIN_H * BLOOM_H - 0.5

        i_fl = ti.floor(u)
        j_fl = ti.floor(v)
        i0 = ti.max(0, ti.min(ti.cast(i_fl, ti.i32), BLOOM_W - 1))
        j0 = ti.max(0, ti.min(ti.cast(j_fl, ti.i32), BLOOM_H - 1))
        i1 = ti.min(i0 + 1, BLOOM_W - 1)
        j1 = ti.min(j0 + 1, BLOOM_H - 1)

        wu = u - i_fl
        wv = v - j_fl

        c00 = bloom_a[i0, j0]
        c10 = bloom_a[i1, j0]
        c01 = bloom_a[i0, j1]
        c11 = bloom_a[i1, j1]

        c0 = c00 * (1.0 - wu) + c10 * wu
        c1 = c01 * (1.0 - wu) + c11 * wu
        glow = c0 * (1.0 - wv) + c1 * wv

        c_orig = pixels_display[i, j] + glow * strength
        pixels_display[i, j] = tone_map_aces(c_orig, exposure)

def apply_bloom(strength: float = 0.65, exposure: float = 1.0):
    bloom_extract()
    bloom_blur_h()
    bloom_blur_v()
    bloom_composite(strength, exposure)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §12  ORBITAL CAMERA WITH EASED FLY-TO TRANSITIONS                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class OrbitalCamera:
    def __init__(self):
        self.azimuth   = 0.0
        self.elevation = 0.65
        self.distance  = 22.0
        self.fov       = 0.45
        self.sens      = 3.5
        self.zoom_sens = 25.0
        self.min_dist  = 1.5
        self.max_dist  = 65.0
        self._prev     = None

        self.transitioning = False
        self._t_start    = None
        self._t_target   = None
        self._t_daz      = 0.0
        self._t_elapsed  = 0.0
        self._t_duration = 2.2

    def update(self, window):
        cur = window.get_cursor_pos()
        mx, my = cur[0], cur[1]

        lmb = window.is_pressed(ti.ui.LMB)
        rmb = window.is_pressed(ti.ui.RMB)
        if (lmb or rmb) and self.transitioning:
            self.transitioning = False

        if self._prev is not None and not self.transitioning:
            dx = mx - self._prev[0]
            dy = my - self._prev[1]

            if lmb and mx > 0.22:
                self.azimuth   -= dx * self.sens
                self.elevation += dy * self.sens
                self.elevation  = max(-1.5, min(1.5, self.elevation))

            if rmb:
                self.distance -= dy * self.zoom_sens
                self.distance  = max(self.min_dist, min(self.max_dist, self.distance))

        self._prev = (mx, my)

        if not self.transitioning:
            if window.is_pressed('w'):
                self.distance = max(self.min_dist, self.distance - 0.35)
            if window.is_pressed('s'):
                self.distance = min(self.max_dist, self.distance + 0.35)
            if window.is_pressed('a'):
                self.azimuth += 0.03
            if window.is_pressed('d'):
                self.azimuth -= 0.03

    def fly_to(self, bookmark, duration=2.2):
        self._t_start = {
            "azimuth": self.azimuth, "elevation": self.elevation,
            "distance": self.distance, "fov": self.fov,
        }
        daz = bookmark["azimuth"] - self.azimuth
        daz = (daz + math.pi) % (2.0 * math.pi) - math.pi
        self._t_daz      = daz
        self._t_target   = bookmark
        self._t_elapsed  = 0.0
        self._t_duration = max(0.05, duration)
        self.transitioning = True

    def update_transition(self, dt):
        if not self.transitioning:
            return
        self._t_elapsed += dt
        t = min(1.0, self._t_elapsed / self._t_duration)
        e = t * t * (3.0 - 2.0 * t)   # Smooth cubic easing
        s, tg = self._t_start, self._t_target
        self.azimuth   = s["azimuth"] + self._t_daz * e
        self.elevation = s["elevation"] + (tg["elevation"] - s["elevation"]) * e
        self.distance  = s["distance"]  + (tg["distance"]  - s["distance"])  * e
        self.fov       = s["fov"]       + (tg["fov"]       - s["fov"])       * e
        if t >= 1.0:
            self.transitioning = False

    def upload(self):
        ce = np.cos(self.elevation);  se = np.sin(self.elevation)
        ca = np.cos(self.azimuth);    sa = np.sin(self.azimuth)

        pos = np.array([
            self.distance * ce * sa,
            self.distance * se,
            self.distance * ce * ca
        ], dtype=np.float32)

        fwd = -pos / (np.linalg.norm(pos) + 1e-12)
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(fwd, world_up)
        rn = np.linalg.norm(right)
        if rn < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            right /= rn

        up = np.cross(right, fwd)
        up /= (np.linalg.norm(up) + 1e-12)

        cam_origin[None]  = pos.tolist()
        cam_right[None]   = right.tolist()
        cam_up[None]      = up.tolist()
        cam_forward[None] = fwd.tolist()

CAMERA_BOOKMARKS = [
    {
        "name": "1. ALMA DSHARP Perspective (Macro)",
        "azimuth": 0.42, "elevation": 0.65, "distance": 25.0, "fov": 0.42,
    },
    {
        "name": "2. Protoplanet Hill Sphere & CPD",
        "azimuth": 0.85, "elevation": 0.22, "distance": 6.8, "fov": 0.28,
    },
    {
        "name": "3. Inner Magnetosphere (ISRO View)",
        "azimuth": 1.55, "elevation": 0.18, "distance": 3.8, "fov": 0.22,
    },
    {
        "name": "4. Edge-on Flared Stratification",
        "azimuth": 0.00, "elevation": 0.03, "distance": 26.0, "fov": 0.35,
    },
    {
        "name": "5. Top-Down Kinematic Velocity Map",
        "azimuth": 0.00, "elevation": 1.48, "distance": 32.0, "fov": 0.50,
    },
]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §13  CLICK-TO-IDENTIFY STAR & DISK REGION INSPECTOR                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def pick_star(mx: float, my: float, fov: float, max_angle_deg: float = None):
    right = np.array(cam_right[None],   dtype=np.float64)
    up    = np.array(cam_up[None],      dtype=np.float64)
    fwd   = np.array(cam_forward[None], dtype=np.float64)

    u = (2.0 * mx - 1.0) * ASPECT * fov
    v = (2.0 * my - 1.0) * fov

    rd = fwd + u * right + v * up
    rd_norm = np.linalg.norm(rd)
    if rd_norm < 1e-12:
        return None
    ray_dir = rd / rd_norm

    dots = STAR_DIRS @ ray_dir
    idx = int(np.argmax(dots))
    best_dot = float(np.clip(dots[idx], -1.0, 1.0))
    angle_deg = math.degrees(math.acos(best_dot))

    if max_angle_deg is None:
        max_angle_deg = math.degrees(max(0.010, fov * 0.08))

    if angle_deg <= max_angle_deg:
        hr, name, ra_h, dec_deg, vmag, bv = REAL_STARS[idx]
        return {
            "hr": hr, "name": name, "ra_h": ra_h, "dec_deg": dec_deg,
            "vmag": vmag, "bv": bv,
            "temp_k": bv_to_temperature(bv),
            "angle_deg": angle_deg,
        }
    return None

def inspect_disk_at_screen_pos(mx: float, my: float, fov: float):
    # Intersect view ray with disk midplane (y=0)
    origin = np.array(cam_origin[None],   dtype=np.float64)
    right  = np.array(cam_right[None],   dtype=np.float64)
    up     = np.array(cam_up[None],      dtype=np.float64)
    fwd    = np.array(cam_forward[None], dtype=np.float64)

    u = (2.0 * mx - 1.0) * ASPECT * fov
    v = (2.0 * my - 1.0) * fov
    rd = fwd + u * right + v * up
    rd /= np.linalg.norm(rd)

    if abs(rd[1]) < 1e-6:
        return None
    t = -origin[1] / rd[1]
    if t <= 0:
        return None

    hit = origin + t * rd
    r = math.hypot(hit[0], hit[2])
    if r < 0.2 or r > 25.0:
        return None

    return float(r)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §14  ASTROPHYSICS EQUATIONS & TELEMETRY ENGINE (CPU DIAGNOSTICS)      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def get_astrophysics_diagnostics(
    planet_m_solar, planet_radius, h0, alpha_turb, stokes_num, sigma0_cgs
):
    """
    Computes rigorous physical metrics for the protostar, protoplanet, and disk,
    incorporating peer-reviewed equations (Chiang-Goldreich, Crida et al., Toomre).
    """
    r = max(float(planet_radius), 0.1)
    mp = max(float(planet_m_solar), 1e-8)

    # 1. Protostellar Parameters (PDS 70 / TW Hydrae analog)
    T_star = 4000.0                       # Effective temperature (Kelvin)
    R_star_m = 2.0 * R_SUN_M              # Protostellar radius (meters)
    L_star_W = 4.0 * math.pi * (R_star_m ** 2) * SIGMA_SB * (T_star ** 4)
    L_star_solar = L_star_W / L_SUN_W

    # Dust Sublimation Radius (T_sub = 1500 K)
    T_sub = 1500.0
    r_sub_m = math.sqrt(L_star_W / (4.0 * math.pi * SIGMA_SB * (T_sub ** 4)))
    r_sub_au = r_sub_m / AU_M

    # 2. Midplane Temperature Profile (Chiang & Goldreich 1997 flared irradiated disk)
    T_mid = T0_MID * (r ** -Q_TEMP)

    # 3. Isothermal Sound Speed c_s = sqrt(gamma * k_B * T / (mu * m_H))
    gamma_gas = 1.4                       # Diatomic adiabatic index
    cs_m_s = math.sqrt((gamma_gas * K_B * T_mid) / (MU * M_H))
    cs_km_s = cs_m_s / 1000.0

    # 4. Keplerian Angular Frequency Omega_K (rad/yr) & Orbital Velocity (km/s)
    omega_k_yr = math.sqrt(G * STAR_MASS / (r ** 3))
    v_k_au_yr = math.sqrt(G * STAR_MASS / r)
    v_k_km_s = v_k_au_yr * AU_PER_YEAR_KM_S

    # 5. Planetary Orbital Period (Earth years)
    period_yr = 2.0 * math.pi / omega_k_yr

    # 6. Gas Scale Height H = c_s / Omega_K (AU) & Aspect Ratio H/r
    cs_au_yr = cs_km_s / AU_PER_YEAR_KM_S
    H_au = cs_au_yr / omega_k_yr
    aspect_ratio = H_au / r

    # 7. Surface Density & Midplane Volume Density
    # Sigma(r) = Sigma0 * (r/1AU)^-1 * exp(-r / R_crit)
    sigma_cgs = sigma0_cgs * (r ** -P_SIGMA) * math.exp(-((r / R_CRITICAL) ** (2.0 - P_SIGMA)))
    # Conversion: 1 g/cm^2 = 10 kg/m^2; rho = Sigma / (sqrt(2*pi)*H)
    H_m = H_au * AU_M
    sigma_si = sigma_cgs * 10.0
    rho_mid_si = sigma_si / (math.sqrt(2.0 * math.pi) * max(H_m, 1e-3))

    # 8. Shakura & Sunyaev (1973) Kinematic Viscosity nu = alpha * c_s * H
    nu_si = alpha_turb * cs_m_s * H_m
    nu_au_yr = alpha_turb * cs_au_yr * H_au

    # Reynolds Number Re = v_K * r / nu
    r_m = r * AU_M
    v_k_m_s = cs_km_s * 1000.0 / cs_au_yr * v_k_au_yr
    reynolds_num = (v_k_m_s * r_m) / max(nu_si, 1e-10)

    # 9. Planetary Hill Radius R_H = a * (M_p / (3*M_*))^(1/3)
    mass_ratio = mp / STAR_MASS
    R_H_au = r * ((mass_ratio / 3.0) ** (1.0 / 3.0))
    R_H_million_km = R_H_au * (AU_KM / 1e6)

    # 10. Roche Lobe Radius R_Roche (Eggleton 1983 approximation)
    q13 = mass_ratio ** (1.0 / 3.0)
    q23 = mass_ratio ** (2.0 / 3.0)
    R_roche_au = r * (0.49 * q23) / (0.6 * q23 + math.log(1.0 + q13))

    # 11. Crida, Morbidelli & Masset (2006) Gap-Opening Parameter P_gap
    # Deep annular gap opens when P_gap <= 1.0
    p_thermal = 0.75 * (H_au / max(R_H_au, 1e-5))
    p_viscous = (50.0 * nu_au_yr) / (mass_ratio * (r ** 2) * omega_k_yr + 1e-7)
    crida_p = p_thermal + p_viscous

    # Kanagawa et al. (2015) empirical gap depth Sigma_gap / Sigma_0
    k_prime = (mass_ratio ** 2) / (max(aspect_ratio, 1e-3) ** 5 * max(alpha_turb, 1e-5))
    gap_depth_ratio = 1.0 / (1.0 + 0.04 * k_prime)

    # 12. Gravitational Stability: Toomre Q parameter Q = c_s * Omega / (pi * G * Sigma)
    G_CGS = 6.67430e-8
    omega_k_s = omega_k_yr / YEAR_S
    toomre_q = (cs_m_s * 100.0 * omega_k_s) / (math.pi * G_CGS * max(sigma_cgs, 1e-3))

    # 13. Dust Aerodynamics & Radial Drift Barrier (Epstein Regime)
    eta = 1.375 * (aspect_ratio ** 2)
    tau_s_yr = stokes_num / omega_k_yr
    # Maximum radial drift speed v_r = -2 * eta * v_K * St / (1 + St^2)
    v_drift_au_yr = -2.0 * eta * v_k_au_yr * (stokes_num / (1.0 + stokes_num ** 2))
    v_drift_century = v_drift_au_yr * 100.0
    drift_timescale_yr = r / max(abs(v_drift_au_yr), 1e-6)

    # 14. Circumplanetary Disk (CPD) Accretion Luminosity & H-alpha Flux
    # R_CPD ~ R_H / 3
    r_cpd_au = R_H_au / 3.0
    r_planet_m = 1.0e8                    # Approximate giant planet radius (~1.4 R_Jup)
    # Empirical accretion rate ~ 10^-8 M_Jup/yr
    m_dot_acc_kg_s = 1.0e-8 * (JUPITER_MASS * M_SUN_KG) / YEAR_S
    m_p_kg = mp * M_SUN_KG
    l_acc_w = (G_SI * m_p_kg * m_dot_acc_kg_s) / r_planet_m
    l_halpha_w = l_acc_w * 0.015          # H-alpha branching fraction ~1.5%

    return {
        "star_lum_solar": L_star_solar,
        "sublimation_radius_au": r_sub_au,
        "temperature_K": T_mid,
        "sound_speed_km_s": cs_km_s,
        "orbital_speed_km_s": v_k_km_s,
        "orbital_period_yr": period_yr,
        "scale_height_AU": H_au,
        "aspect_ratio": aspect_ratio,
        "surface_density_cgs": sigma_cgs,
        "midplane_density_si": rho_mid_si,
        "viscosity_si": nu_si,
        "reynolds_number": reynolds_num,
        "hill_radius_AU": R_H_au,
        "hill_radius_Mkm": R_H_million_km,
        "roche_lobe_AU": R_roche_au,
        "crida_p": crida_p,
        "gap_depth_ratio": gap_depth_ratio,
        "toomre_q": toomre_q,
        "stopping_time_yr": tau_s_yr,
        "drift_speed_century": v_drift_century,
        "drift_timescale_yr": drift_timescale_yr,
        "cpd_radius_au": r_cpd_au,
        "halpha_lum_w": l_halpha_w,
    }

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  §15  MAIN SIMULATION ENGINE & INTERACTIVE CONTROL DECK                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print()
    print("=" * 72)
    print("  ASTRA-V7: PROTOPLANETARY GAS & DUST DISK SIMULATION DECK")
    print("  Target Class: PDS 70 / HL Tauri Protostellar Multi-Fluid Systems")
    print("  Observational Baselines: NASA JWST/HST, ALMA DSHARP, ISRO AstroSat")
    print("=" * 72)
    print("  Controls:")
    print("  [SPACE]      : Play / Pause simulation")
    print("  [R]          : Reset disk particles to initial equilibrium")
    print("  [1] - [5]    : Switch spectral views (ALMA, JWST, UVIT, IR, CO)")
    print("  [H]          : Toggle telemetry HUD & control panel")
    print("  [G]          : Toggle gas aerodynamic drag")
    print("  [+] / [-]    : Accelerate / decelerate simulation time warp")
    print("  [ [ ] / ] ]  : Step planet orbital radius inward / outward")
    print("  [M] / [N]    : Increase / decrease planet mass")
    print("  Left Drag    : Orbit camera in 3D space")
    print("  Right Drag   : Zoom camera in / out")
    print("=" * 72)
    print()

    # Initialize Taichi 3D Window & Canvas
    window = ti.ui.Window(
        "ASTRA-V7 — Protoplanetary Gas & Dust Disk Control Deck",
        (WIN_W, WIN_H),
        vsync=True
    )

    # Windows DWM Dark Mode Styling (consistent with ypae.1)
    try:
        import ctypes
        hwnd = ctypes.windll.user32.FindWindowW(None, "ASTRA-V7 — Protoplanetary Gas & Dust Disk Control Deck")
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 3)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 20, ctypes.byref(ctypes.c_int(1)), 4)
            ctypes.windll.dwmapi.DwmSetWindowAttribute(hwnd, 35, ctypes.byref(ctypes.c_int(0x00100C0B)), 4)
    except Exception:
        pass

    canvas = window.get_canvas()
    scene  = window.get_scene()
    gui    = window.get_gui()
    camera = OrbitalCamera()

    # Camera Preset Bookmarks
    auto_orbit = False

    # Simulation State
    paused = False
    simulation_speed = 1.0
    substeps = 3
    simulation_time = 0.0

    # Planetary Parameters
    planet_mass_mjup = 1.0                # Jupiter masses
    planet_radius = 5.2                   # Semi-major axis (AU)
    planet_eccentricity = 0.0             # Orbital eccentricity
    planet_angle = 0.0                    # Orbital phase / true anomaly
    enable_migration = False              # Type I / Type II inward migration
    migration_rate = 0.005                # AU per year

    # Gas Disk Hydrodynamics & Aerodynamics
    enable_gas_drag = True
    stokes_number_scale = 0.06            # Stokes number baseline
    aspect_ratio_h0 = 0.040               # Midplane aspect ratio H/r at 1 AU
    gas_sigma0 = 1700.0                   # Base surface density (g/cm^2)
    alpha_turb = 0.001                    # Shakura-Sunyaev turbulent viscosity parameter

    # Celestial Skybox & Stars
    mw_tilt_deg = 62.0
    mw_intensity = 0.65
    mw_hue = 0.62
    mw_tint = 0.0

    # Visual & Post-Processing Parameters
    color_palette_idx = 0                 # 0: ALMA, 1: JWST, 2: UVIT, 3: IR, 4: CO
    particle_display_size = 0.007
    show_ui = True
    show_guide_rings = True
    show_lagrange_points = True
    render_scale = 1.0
    exposure = 1.05
    bloom_strength = 0.70

    # Spectral Mode Names
    spectral_names = [
        "1. ALMA 1.3 mm Dust Continuum (DSHARP)",
        "2. NASA JWST NIRCam Scattered Light (1.6-4.4 um)",
        "3. ISRO AstroSat UVIT & H-Alpha (656 nm)",
        "4. Thermal Mid-Infrared (JWST MIRI 10 um)",
        "5. Keplerian Kinematics & CO Doppler Map",
    ]

    # Initialize disk particles on GPU
    planet_x_init = planet_radius * math.cos(planet_angle)
    planet_z_init = planet_radius * math.sin(planet_angle)
    initialize_disk(aspect_ratio_h0, alpha_turb, planet_x_init, planet_z_init)

    # Initialize Skybox & Yale Stars
    bake_real_stars()

    # Performance Timing
    fps_smooth = 60.0
    last_perf = _clock.perf_counter()

    # Click-to-identify tracking
    selected_star = None
    inspected_radius_au = None
    lmb_was_down = False
    lmb_down_pos = (0.0, 0.0)

    # Main Application Event Loop
    while window.running:
        t_now = _clock.perf_counter()
        dt_frame = t_now - last_perf
        last_perf = t_now
        if dt_frame > 0.0:
            fps_smooth = fps_smooth * 0.92 + (1.0 / dt_frame) * 0.08
        frame_latency_ms = 1000.0 / max(fps_smooth, 1.0)

        # ----------------------------------------------------
        # KEYBOARD EVENT HANDLING
        # ----------------------------------------------------
        while window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.SPACE:
                paused = not paused
            elif window.event.key == "r":
                initialize_disk(aspect_ratio_h0, alpha_turb, planet_x_init, planet_z_init)
                simulation_time = 0.0
                planet_angle = 0.0
            elif window.event.key == "h":
                show_ui = not show_ui
            elif window.event.key == "g":
                enable_gas_drag = not enable_gas_drag
            elif window.event.key == "1":
                color_palette_idx = 0
            elif window.event.key == "2":
                color_palette_idx = 1
            elif window.event.key == "3":
                color_palette_idx = 2
            elif window.event.key == "4":
                color_palette_idx = 3
            elif window.event.key == "5":
                color_palette_idx = 4
            elif window.event.key in ["+", "="]:
                simulation_speed = min(simulation_speed * 1.25, 10.0)
            elif window.event.key in ["-", "_"]:
                simulation_speed = max(simulation_speed / 1.25, 0.05)
            elif window.event.key == "[":
                planet_radius = max(1.2, planet_radius - 0.25)
            elif window.event.key == "]":
                planet_radius = min(14.0, planet_radius + 0.25)
            elif window.event.key == "m":
                planet_mass_mjup = min(15.0, planet_mass_mjup * 1.25)
            elif window.event.key == "n":
                planet_mass_mjup = max(0.005, planet_mass_mjup / 1.25)
            elif window.event.key == ti.ui.ESCAPE:
                window.running = False

        # Mouse Picking & Star/Disk Identification
        cur_pos = window.get_cursor_pos()
        lmb_down = window.is_pressed(ti.ui.LMB)
        if lmb_down and not lmb_was_down:
            lmb_down_pos = cur_pos
        if (not lmb_down) and lmb_was_down:
            ddx = cur_pos[0] - lmb_down_pos[0]
            ddy = cur_pos[1] - lmb_down_pos[1]
            if math.hypot(ddx, ddy) < 0.004 and lmb_down_pos[0] > 0.22:
                # First attempt star identification
                picked = pick_star(lmb_down_pos[0], lmb_down_pos[1], camera.fov)
                if picked is not None:
                    selected_star = picked
                    inspected_radius_au = None
                else:
                    # Otherwise inspect disk midplane properties at click
                    r_hit = inspect_disk_at_screen_pos(lmb_down_pos[0], lmb_down_pos[1], camera.fov)
                    if r_hit is not None:
                        inspected_radius_au = r_hit
                        selected_star = None
        lmb_was_down = lmb_down

        # Camera Update & Eased Transitions
        camera.update(window)
        camera.update_transition(dt_frame)
        if auto_orbit and not camera.transitioning:
            camera.azimuth += 0.004
        camera.upload()

        # Convert user planet mass from Jupiter masses to Solar masses
        planet_m_solar = planet_mass_mjup * JUPITER_MASS

        # ----------------------------------------------------
        # NUMERICAL PHYSICS ADVANCE (SUBSTEPPED OPERATOR SPLIT)
        # ----------------------------------------------------
        if not paused:
            dt_step = (BASE_DT * simulation_speed) / float(substeps)

            for _ in range(substeps):
                # Keplerian angular motion
                omega_p = math.sqrt(G * (STAR_MASS + planet_m_solar) / (planet_radius ** 3))
                planet_angle += omega_p * dt_step

                # Inward orbital migration (Type I / Type II)
                if enable_migration and planet_radius > 1.2:
                    planet_radius = max(1.2, planet_radius - migration_rate * dt_step)

                # Elliptical orbit geometry
                if planet_eccentricity > 0.001:
                    r_curr = planet_radius * (1.0 - planet_eccentricity ** 2) / (
                        1.0 + planet_eccentricity * math.cos(planet_angle)
                    )
                else:
                    r_curr = planet_radius

                planet_x = r_curr * math.cos(planet_angle)
                planet_z = r_curr * math.sin(planet_angle)

                # Advance particles via operator-split symplectic Verlet with Epstein drag
                advance_particles_symplectic(
                    dt_step,
                    planet_m_solar,
                    planet_x,
                    planet_z,
                    stokes_number_scale,
                    aspect_ratio_h0,
                    alpha_turb,
                    1 if enable_gas_drag else 0
                )
                simulation_time += dt_step
        else:
            if planet_eccentricity > 0.001:
                r_curr = planet_radius * (1.0 - planet_eccentricity ** 2) / (
                    1.0 + planet_eccentricity * math.cos(planet_angle)
                )
            else:
                r_curr = planet_radius
            planet_x = r_curr * math.cos(planet_angle)
            planet_z = r_curr * math.sin(planet_angle)

        # ----------------------------------------------------
        # GPU GUIDE OVERLAYS & SPECTRAL SHADER UPDATE
        # ----------------------------------------------------
        mass_ratio = planet_m_solar / STAR_MASS
        hill_r = planet_radius * ((mass_ratio / 3.0) ** (1.0 / 3.0))

        # Protostellar sublimation radius
        r_sublim = 0.077 * math.sqrt(STAR_MASS)

        if show_guide_rings:
            update_guide_overlays(planet_radius, planet_x, planet_z, hill_r, r_sublim)

        compute_shader_colors(
            color_palette_idx,
            planet_x,
            planet_z,
            planet_m_solar,
            stokes_number_scale,
            aspect_ratio_h0
        )

        # ----------------------------------------------------
        # SCENE RENDERING PIPELINE
        # ----------------------------------------------------
        cam_pos = np.array(cam_origin[None])
        cam_target = np.array([0.0, 0.0, 0.0])
        cam_up_v = np.array(cam_up[None])

        # Taichi GGUI Camera
        t_cam = ti.ui.Camera()
        t_cam.position(cam_pos[0], cam_pos[1], cam_pos[2])
        t_cam.lookat(cam_target[0], cam_target[1], cam_target[2])
        t_cam.up(cam_up_v[0], cam_up_v[1], cam_up_v[2])
        scene.set_camera(t_cam)

        # Protostellar & Ambient Illumination
        scene.ambient_light((0.28, 0.28, 0.32))
        scene.point_light(pos=(0.0, 0.8, 0.0), color=(1.0, 0.92, 0.75))

        # 1. Active Multi-Grain Dust Particles (100,000 GPU Particles)
        scene.particles(
            positions,
            radius=particle_display_size,
            per_vertex_color=particle_colors
        )

        # 2. Central Protostar
        scene.particles(
            star_pos,
            radius=0.25,
            per_vertex_color=star_color
        )

        # 3. Protoplanet
        scene.particles(
            planet_pos,
            radius=0.12,
            per_vertex_color=planet_color
        )

        # 4. Lagrange Points L1-L5
        if show_lagrange_points:
            scene.particles(
                lagrange_pos,
                radius=0.06,
                color=(0.95, 0.75, 0.15)
            )

        # 5. Guide Rings & Resonances
        if show_guide_rings:
            scene.lines(orbit_guide_lines, width=1.4, color=(0.20, 0.55, 0.90))
            scene.lines(hill_guide_lines, width=1.6, color=(0.30, 0.95, 0.55))
            scene.lines(inner_res_lines, width=1.0, color=(0.85, 0.35, 0.75))
            scene.lines(outer_res_lines, width=1.0, color=(0.85, 0.35, 0.75))
            scene.lines(sublim_guide_lines, width=1.2, color=(1.00, 0.65, 0.25))

        canvas.set_background_color((0.005, 0.006, 0.012))
        canvas.scene(scene)

        # ----------------------------------------------------
        # REAL-TIME ASTROPHYSICAL DIAGNOSTICS
        # ----------------------------------------------------
        diag_r = inspected_radius_au if inspected_radius_au is not None else planet_radius
        diag = get_astrophysics_diagnostics(
            planet_m_solar,
            diag_r,
            aspect_ratio_h0,
            alpha_turb,
            stokes_number_scale,
            gas_sigma0
        )

        # ----------------------------------------------------
        # IMGUI SCIENTIFIC TELEMETRY DASHBOARD
        # ----------------------------------------------------
        if show_ui:
            # ───────────────── LEFT CONTROL PANEL ─────────────────
            with gui.sub_window("ASTRA-V7: Controls & Physics", 0.01, 0.015, 0.21, 0.97):
                gui.text(f"FPS: {fps_smooth:4.0f} ({frame_latency_ms:4.1f} ms)")
                gui.text(f"Particles: {N_PARTICLES:,} (Multi-Grain)")
                gui.text("")

                gui.text("=== SIMULATION CONTROLS ===")
                btn_pause = ">> Resume (SPACE)" if paused else "|| Pause (SPACE)"
                if gui.button(btn_pause):
                    paused = not paused
                if gui.button("Reset Disk Particles (R)"):
                    initialize_disk(aspect_ratio_h0, alpha_turb, planet_x, planet_z)
                    simulation_time = 0.0
                    planet_angle = 0.0

                simulation_speed = gui.slider_float("Time Warp", simulation_speed, 0.1, 8.0)
                substeps = gui.slider_int("Substeps/Frame", substeps, 1, 8)
                gui.text("")

                gui.text("=== SPECTRAL OBSERVATIONS ===")
                for s_idx, s_name in enumerate(spectral_names):
                    if gui.button(s_name):
                        color_palette_idx = s_idx
                gui.text(f"Active: {spectral_names[color_palette_idx]}")
                gui.text("")

                gui.text("=== PROTOPLANETARY PARAMETERS ===")
                planet_mass_mjup = gui.slider_float("Mass (M_Jup)", planet_mass_mjup, 0.0, 12.0)
                planet_radius = gui.slider_float("Orbit Radius a (AU)", planet_radius, 1.2, 13.0)
                planet_eccentricity = gui.slider_float("Eccentricity e", planet_eccentricity, 0.0, 0.5)
                enable_migration = gui.checkbox("Inward Migration (da/dt)", enable_migration)
                if enable_migration:
                    migration_rate = gui.slider_float("Migration Rate (AU/yr)", migration_rate, 0.001, 0.05)
                gui.text("")

                gui.text("=== GAS DISK & AERODYNAMICS ===")
                enable_gas_drag = gui.checkbox("Aerodynamic Drag (G)", enable_gas_drag)
                stokes_number_scale = gui.slider_float("Stokes No. Scale", stokes_number_scale, 0.005, 0.8)
                aspect_ratio_h0 = gui.slider_float("Aspect Ratio (H/r)_0", aspect_ratio_h0, 0.02, 0.08)
                alpha_turb = gui.slider_float("Turbulence alpha", alpha_turb, 0.0001, 0.008)
                gas_sigma0 = gui.slider_float("Base Sigma_0", gas_sigma0, 400.0, 4000.0)
                gui.text("")

                gui.text("=== CAMERA & VISUALS ===")
                particle_display_size = gui.slider_float("Particle Size", particle_display_size, 0.002, 0.016)
                show_guide_rings = gui.checkbox("Show Orbit & Hill Rings", show_guide_rings)
                show_lagrange_points = gui.checkbox("Show Lagrange Points L1-L5", show_lagrange_points)
                if gui.button("Toggle Cinematic Auto-Orbit"):
                    auto_orbit = not auto_orbit
                gui.text("")

                gui.text("--- CAMERA BOOKMARKS (Eased Fly-To) ---")
                for bm in CAMERA_BOOKMARKS:
                    if gui.button(bm["name"]):
                        camera.fly_to(bm, duration=2.2)
                gui.text("")

                gui.text("\n[H] Toggle HUD  |  LMB/RMB Rotate/Zoom")

            # ───────────────── RIGHT TELEMETRY HUD ─────────────────
            with gui.sub_window("ASTRA-V7: Astrophysics Telemetry Deck", 0.78, 0.015, 0.21, 0.97):
                orbits_done = simulation_time / max(diag['orbital_period_yr'], 1e-4)
                gui.text(f"Sim Time: {simulation_time:6.2f} yr ({orbits_done:5.2f} orbits)")
                gui.text("")

                gui.text("--- PROTOSTAR (PDS 70 / TW Hydrae) ---")
                gui.text(f"Mass M_*: {STAR_MASS:.2f} M_sun (T Tauri class)")
                gui.text(f"Luminosity L_*: {diag['star_lum_solar']:.2f} L_sun")
                gui.text(f"Dust Sublimation Wall: {diag['sublimation_radius_au']:.3f} AU")
                gui.text("ISRO AstroSat: Magnetospheric UV Shock Active")
                gui.text("")

                gui.text("--- PROTOPLANET & CPD ACCRETION ---")
                m_earth = planet_mass_mjup * (JUPITER_MASS / EARTH_MASS)
                gui.text(f"Mass: {planet_mass_mjup:.3f} M_J ({m_earth:.1f} M_Earth)")
                gui.text(f"Orbit Radius a: {planet_radius:.2f} AU")
                gui.text(f"Orbital Speed v_K: {diag['orbital_speed_km_s']:.2f} km/s")
                gui.text(f"Orbital Period P: {diag['orbital_period_yr']:.2f} Earth yr")
                gui.text(f"Hill Radius R_H: {diag['hill_radius_AU']:.3f} AU ({diag['hill_radius_Mkm']:.1f} M km)")
                gui.text(f"Roche Lobe R_L: {diag['roche_lobe_AU']:.3f} AU")
                gui.text(f"CPD Disk Radius: {diag['cpd_radius_au']:.3f} AU")
                gui.text(f"H-Alpha Line Lum: {diag['halpha_lum_w']:.2e} W")
                gui.text("")

                gui.text("--- DISK ENVIRONMENT & DYNAMICS ---")
                if inspected_radius_au is not None:
                    gui.text(f">> INSPECTING RADIUS: {inspected_radius_au:.2f} AU <<")
                else:
                    gui.text(f">> LOCAL TO PLANET: {planet_radius:.2f} AU <<")
                gui.text(f"Midplane Temp: {diag['temperature_K']:.1f} K")
                gui.text(f"Sound Speed c_s: {diag['sound_speed_km_s']:.3f} km/s")
                gui.text(f"Gas Scale Height H: {diag['scale_height_AU']:.3f} AU")
                gui.text(f"Aspect Ratio H/r: {diag['aspect_ratio']:.3f}")
                gui.text(f"Gas Surface Density: {diag['surface_density_cgs']:.1f} g/cm^2")
                gui.text(f"Midplane Density: {diag['midplane_density_si']:.2e} kg/m^3")
                gui.text(f"Kinematic Viscosity: {diag['viscosity_si']:.2e} m^2/s")
                gui.text(f"Reynolds Number Re: {diag['reynolds_number']:.1e}")
                gui.text("")

                gui.text("--- AERODYNAMICS & GAP OPENING ---")
                crida = diag["crida_p"]
                gap_str = "CLEARS DEEP GAP" if crida <= 1.0 else "SHALLOW WAKE"
                gui.text(f"Crida Gap Metric: {crida:.3f} [{gap_str}]")
                gui.text(f"Gap Depth (Sigma/Sigma0): {diag['gap_depth_ratio']:.3f}")
                toomre = diag["toomre_q"]
                toomre_str = "STABLE" if toomre > 1.4 else "GRAV. UNSTABLE"
                gui.text(f"Toomre Stability Q: {toomre:.1f} [{toomre_str}]")
                gui.text(f"Stopping Time: {diag['stopping_time_yr']:.3f} yr")
                gui.text(f"Max Pebble Drift: {diag['drift_speed_century']:.2f} AU/century")
                gui.text(f"Drift Timescale: {diag['drift_timescale_yr']:,.0f} yr")
                gui.text("")

                gui.text("--- CLICK-TO-INSPECT IDENTIFIER ---")
                if selected_star is not None:
                    s = selected_star
                    gui.text(f"Star: {s['name']} (HR {s['hr']})")
                    gui.text(f"RA {s['ra_h']:.4f}h | Dec {s['dec_deg']:+.4f} deg")
                    gui.text(f"V mag: {s['vmag']:+.2f}")
                    if s['temp_k'] is not None:
                        gui.text(f"Temperature: {s['temp_k']:,.0f} K")
                    if gui.button("Clear Star Selection"):
                        selected_star = None
                elif inspected_radius_au is not None:
                    gui.text("Inspecting Disk Annulus.")
                    if gui.button("Reset Inspection to Planet"):
                        inspected_radius_au = None
                else:
                    gui.text("Click any background star to identify.")
                    gui.text("Click anywhere in disk to inspect.")
                gui.text("")

                gui.text("--- MISSION BASES & CITATIONS ---")
                gui.text("NASA: JWST NIRCam, HST, Kepler")
                gui.text("ALMA: DSHARP Survey (Andrews+ 2018)")
                gui.text("ISRO: AstroSat UVIT YSO Studies")
                gui.text("Theory: Crida+ (2006), Chiang+ (1997)")

        window.show()

    print("ASTRA-V7 simulation terminated cleanly.")

if __name__ == "__main__":
    main()