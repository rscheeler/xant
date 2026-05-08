# 📡 xant

**xant** is a Python library for spatial antenna analysis. Define antenna patterns analytically or from measured data, position them in 3D space, combine them into phased arrays, and evaluate gain, beamwidth, and directivity — all coordinate-frame-aware and unit-safe via [pint](https://pint.readthedocs.io/) and [xarray](https://docs.xarray.dev/).

Built on top of [hics](https://github.com/rscheeler/hics), [xarray](https://docs.xarray.dev/), and [scipy](https://scipy.org/).

## Installation

```bash
uv add xant
```

Or with pip:

```bash
pip install xant
```

## Features

- **Lazy antenna math** — add, subtract, multiply, and divide patterns using Python operators; the expression tree is only evaluated when data is requested
- **Arbitrary coordinate frames** — request pattern data in `phitheta`, `azel`, `elaz`, `uv`, `uvw`, `llh`, `ecef`, or `h3`; xant handles all rotation and interpolation automatically
- **Spatial positioning** — place any antenna or array in 3D space using [hics](https://github.com/rscheeler/hics) hierarchical coordinate systems; relative rotations are resolved automatically
- **Phased arrays** — build rectangular, triangular, or arbitrary lattice arrays; steer dispersively by phase center; apply amplitude tapers; visualize element layout
- **Hierarchical arrays** — compose arrays of arrays for subarray-based architectures
- **Spatial RF link analysis** — compute received power between two spatially positioned antennas with automatic antenna pointing, polarization projection, and horizon angle resolution
- **Transmit power density mapping** — project EIRP onto a lat/lon/alt grid and plot coverage maps over satellite imagery via cartopy
- **Propagation models** — free-space path loss (FSPL) and terrain-aware ITM (Irregular Terrain Model) with polarization-resolved H/V loss, optional clutter models (ITU-R P.1812, ITU-R P.2108), and configurable reliability/confidence percentiles
- **Analytical models** — isotropic, dipole, cosine, cardioid, aperture, and more built-in element patterns
- **Data patterns** — load and interpolate measured or simulated antenna data from `.xant` (NetCDF/HDF5) files using spline interpolation
- **Polarization support** — various polarizations support in different coordinate systems; apolar patterns for total power
- **Array concatenation** — stack antennas along arbitrary named dimensions for channel or port analysis
- **Antenna metrics** — beamwidth, directivity (D0), and total radiated power (TRP)
- **Unit safety** — all quantities carry pint units; degrees, radians, Hz, and meter conversions are handled automatically

## Quick Start

### Define and query an antenna

```python
import numpy as np
from xrench.units import ureg
from xant.antenna.common import Dipole

freq  = np.array([2.4e9]) * ureg.Hz
theta = np.linspace(0, 180, 181) * ureg.degree
phi   = np.arange(-180, 180, 5) * ureg.degree

dip  = Dipole(l=0.5 * ureg.m, frequency=freq)
data = dip.request_data(theta=theta, phi=phi, coordinate_frame="phitheta")
# Returns xr.DataArray with dims: (polarization, frequency, phi, theta)
```

### Position an antenna in space

```python
from hics import HCS
from scipy.spatial.transform import Rotation

# Tilt 30° around Y, offset 1 m above origin
cs  = HCS((0, 0, 1) * ureg.m, rotation=Rotation.from_euler("Y", 30, degrees=True))
ant = Dipole(l=0.5 * ureg.m, frequency=freq, hcs=cs)
```

### Antenna math

Operators are lazy — the expression tree is evaluated only when `request_data()` is called.

```python
from xant.antenna.common import Isotropic

iso      = Isotropic(frequency=freq)
combined = iso + dip       # coherent sum
scaled   = dip * 0.5       # amplitude scale
ratio    = dip / iso       # pattern ratio
```

### Phased array

```python
from xant.antenna.phasedarray import AntennaArray, steer_phase_centers
from hics import GLOBAL_CS

freq = np.array([3e9]) * ureg.Hz
lam  = (ureg.speed_of_light / freq).to("m")

element = Isotropic(frequency=freq)
array   = AntennaArray.rectangular(
    element, nx=8, dx=0.5*lam, ny=8, dy=0.5*lam, cs_reference=GLOBAL_CS
)

# Steer to theta=30°, phi=0°
steer_phase_centers(array, frequency=freq[0], theta=30*ureg.degree, phi=0*ureg.degree)

data = array.total.request_data(theta=theta, phi=phi, coordinate_frame="phitheta")
```

### Hierarchical subarray

```python
subarray = AntennaArray.rectangular(
    element, nx=2, dx=0.5*lam, ny=2, dy=0.5*lam, cs_reference=GLOBAL_CS
)
top = AntennaArray(subarray, [HCS((i * 2 * lam[0].magnitude, 0, 0) * ureg.m) for i in range(4)])
```

### Spatial RF link

Calculate received power between two geospatially positioned antennas. xant automatically resolves the pointing angles from each antenna's coordinate system toward the other, projects polarization, and applies the selected propagation model.

```python
from hics import HCS
from xant.propagation.rflink import calculate_spatial_link

# Position TX and RX using geodetic coordinates
tx_cs = HCS.from_crs((40.015 * ureg.degree, -105.27 * ureg.degree, 30 * ureg.m), hagl=True)
rx_cs = HCS.from_crs((40.020 * ureg.degree, -105.24 * ureg.degree,  10 * ureg.m), hagl=True)

tx = Isotropic(frequency=freq, hcs=tx_cs)
rx = Isotropic(frequency=freq, hcs=rx_cs)

rx_power, prop_loss, incident_pol, txcs, rxcs = calculate_spatial_link(
    tx, power=1 * ureg.watt, rx=rx, propagation="fspl"
)
print(rx_power)  # received power in dBm
```

### Terrain-aware propagation with ITM

```python
rx_power, prop_loss, incident_pol, txcs, rxcs = calculate_spatial_link(
    tx,
    power=1 * ureg.watt,
    rx=rx,
    propagation="itm_rflink",
    gnd="good",           # ground conductivity: "poor", "average", "good", "fresh_water"
    time=[50],            # reliability percentile
    situation=[50],       # confidence percentile
)
```

### Transmit power density map

Project EIRP across a lat/lon grid and visualize coverage over a map.

```python
from xant.propagation.rflink import transmit_power_density, plot_transmit_pd

lats = np.linspace(40.00, 40.04, 50) * ureg.degree
lons = np.linspace(-105.30, -105.20, 50) * ureg.degree
hs   = [0] * ureg.m

pd_map = transmit_power_density(
    tx, power=1 * ureg.watt,
    lats=lats, lons=lons, hs=hs,
    propagation="fspl",
)

plot_transmit_pd(pd_map)
```

### Link horizon visualization

Visualize the link geometry, terrain profile, Fresnel zones, and antenna gain patterns in a single cross-section plot.

```python
from xant.propagation.rflink import view_link_horizon

ax = view_link_horizon(tx, rx, rx_power, incident_pol, prop_loss,gsize=.02)
```

### Load and export measured patterns

```python
import xant

ant  = xant.Antenna("measured_pattern.xant")
ant.export("my_antenna.xant")
```

### Metrics

```python
bw  = ant.beamwidth(dim="phi", coordinate_frame="phitheta")
d0  = ant.d0()
trp = ant.trp()
```

## API Reference

### `Antenna`

The base pattern container. Wraps either an `xr.DataArray` of measured data or an `AntennaFunction` for analytical/lazy patterns.

| Parameter | Description |
|----------:|:------------|
| `data` | Path to `.xant` file, `xr.DataArray`, or `AntennaFunction` |
| `hcs` | `HCS` coordinate system defining position and orientation |

Key methods: `.request_data()`, `.move()`, `.export()`, `.copy()`, `.beamwidth()`, `.d0()`, `.trp()`

Operators: `+`, `-`, `*`, `/`, `**`, unary `-`, `.sum(dim)`

### `request_data()`

```python
ant.request_data(
    coordinate_frame="phitheta",   # output coordinate frame
    hcs=None,                      # viewpoint coordinate system
    theta=...,                     # spatial coordinate kwargs (Quantity or DataArray)
    phi=...,
)
```

Returns an `xr.DataArray` with dims `(polarization, frequency, ...)`. All coordinate frame transforms and spatial rotations are applied automatically.

### `AntennaArray`

A phased array of `Antenna` or nested `AntennaArray` elements.

| Constructor | Description |
|------------:|:------------|
| `AntennaArray(element, coordinate_systems)` | Arbitrary element positions |
| `AntennaArray.rectangular(element, nx, dx, ny, dy, cs_reference)` | Rectangular lattice |
| `AntennaArray.triangular(element, nx, ny, dx, cs_reference)` | Triangular lattice |

| Property | Description |
|---------:|:------------|
| `.total` | Sum of translated element patterns weighted by excitation |
| `.af` | Array factor (translated phase × excitation) |
| `.elements` | Per-element patterns with translation applied |
| `.excitation` | Complex excitation weights (`taper × steering_vector`) |
| `.positions(reference)` | Element positions relative to a reference HCS |

Visualization: `.show()` (3D), `.showxy()` (2D top-down with optional phase/magnitude/subarray coloring)

### Steering and Tapering

```python
from xant.antenna.phasedarray import steer_phase_centers, apply_taper

steer_phase_centers(array, frequency=freq[0], theta=30*ureg.degree, phi=0*ureg.degree)
apply_taper(array, window=np.kaiser(N, beta=6), dim="x")
```

### Propagation (`xant.propagation`)

| Function | Description |
|---------:|:------------|
| `calculate_spatial_link(tx, power, rx, propagation)` | Received power, path loss, and pointing angles between two `Antenna` objects |
| `transmit_power_density(tx, power, lats, lons, hs, propagation)` | EIRP projected onto a lat/lon/alt grid in dBm |
| `plot_transmit_pd(data)` | Coverage map over satellite imagery; supports animation along a time dimension |
| `plot_link(data)` | Scatter plot of received power on a map projection |
| `view_link_horizon(tx, rx, res, incident_pol, prop_loss)` | Terrain cross-section with antenna patterns, LOS, and Fresnel zones overlaid |

| Propagator | Description |
|-----------:|:------------|
| `fspl` | Free-space path loss |
| `itm_rflink` | ITM terrain propagation, H/V polarization-resolved, configurable reliability/confidence percentiles and ground type |
| `mask_los` | Mask path loss beyond the radio horizon |

ITM ground types: `"poor"`, `"average"`, `"good"`, `"fresh_water"`

ITM clutter methods: `ClutterMethods.NONE`, `ClutterMethods.ITURP1812`, `ClutterMethods.ITURP2108`

### Built-in Antenna Models (`xant.antenna.common`)

| Class | Description |
|------:|:------------|
| `Isotropic` | Uniform gain in all directions |
| `Dipole` | Half-wave dipole with theta/phi polarization |
| `Cardioid` | Cardioid pattern |
| `Cosine` | Cosine^n element pattern |
| `Hemispherical` | Upper-hemisphere only |
| `ElementWeight` | Slope-vs-theta weighting function |
| `RectangularAperture` | Uniform rectangular aperture (Balanis Ch. 12) |
| `TE10Aperture` | TE₁₀-mode waveguide aperture |
| `CircularAperture` | Uniform circular aperture (Balanis Ch. 12) |
| `DipoleAboveGround` | Dipole over a ground plane via image theory |

### Coordinate Frames

| Frame | Dims | Notes |
|------:|:-----|:------|
| `phitheta` | `phi`, `theta` | Standard spherical |
| `azel` | `azimuth`, `elevation` | Azimuth over elevation |
| `elaz` | `elevation`, `azimuth` | Elevation over azimuth |
| `uv` | `u`, `v` | Sin-space projection |
| `uvw` | `u`, `v`, `w` | Direction cosines |
| `llh` | `lat`, `lon`, `h` | Geodetic — requires HCS |
| `ecef` | `x`, `y`, `z` | Earth-centered — requires HCS |
| `h3` | `i`, `j`, `h` | H3 hexagonal grid — requires HCS |

### Concatenation

```python
from xant.antenna.core import concat

multi = concat([ant1, ant2, ant3], coord={"channel": [0, 1, 2]})
data  = multi.request_data(channel=[0, 2], theta=theta, phi=phi, coordinate_frame="phitheta")
```

### Grating Lobe Diagram

```python
from xant.antenna.phasedarray import plot_grating_lobe_diagram

plot_grating_lobe_diagram(
    dx=0.5, dy=0.5,
    steering_angles=(0*ureg.degree, 30*ureg.degree),
    lattice_type="rectangular",
    max_scan=60*ureg.degree,
)
```

## Dependencies

- [hics](https://github.com/rscheeler/hics) — hierarchical coordinate systems and geospatial utilities
- [xrench](https://github.com/rscheeler/xrench) — xarray + pint integration utilities
- [xarray](https://docs.xarray.dev/) — labeled N-D arrays
- [pint](https://pint.readthedocs.io/) — unit-aware quantities
- numpy, scipy, matplotlib
- cartopy, shapely *(propagation and mapping)*

## License

MIT

## Author

Created by [Rob Scheeler](https://github.com/rscheeler)
