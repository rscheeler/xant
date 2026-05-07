# EMerge → xant Integration

Export far-field antenna patterns from [EMerge](https://www.emerge-em.com/) simulations into xant `Antenna` objects for spatial analysis, phased array modeling, and RF link calculations.

## Requirements

- `xant`
- `emerge` Python package (installed with EMerge) — only required at runtime, not at import time

The script uses `TYPE_CHECKING` guards so that EMerge's types are used for static analysis and editor autocompletion without creating a hard import dependency. You can import `emerge2xant` in any environment; EMerge only needs to be present when you actually call the functions.

## Usage

Copy `emerge2xant.py` into your project and import directly.

### Single pattern export

```python
import numpy as np
from xrench.units import ureg
from emerge import Simulation
from emerge._emerge.physics.microwave.microwave_data import MWData

from emerge2xant import emerge2xr

model  = Simulation("my_model")
mwdata = MWData(model)

theta = np.linspace(0, 180, 181) * ureg.degree
phi   = np.arange(-180, 180, 5) * ureg.degree

ant  = emerge2xr(model, mwdata, faces, theta, phi, element_pattern=False)
data = ant.request_data(theta=theta, phi=phi, coordinate_frame="phitheta")
```

### Embedded element pattern export

For phased array analysis, set `element_pattern=True` to export the embedded element pattern for each port — each element is excited in turn with all others terminated:

```python
ant = emerge2xr(model, mwdata, faces, theta, phi, element_pattern=True)
# Returns Antenna with dims: (port, polarization, frequency, phi, theta)

data = ant.request_data(port=0, theta=theta, phi=phi, coordinate_frame="phitheta")
```

### Export to file

```python
ant = emerge2xr(model, mwdata, faces, theta, phi, export=True)
# Saves to <model name>.EMResults/<model name>.xant

# Reload later without EMerge
import xant
ant = xant.Antenna("my_model.xant")
```

## Polarization

EMerge exports Cartesian `(x, y, z)` polarization components. xant natively supports this basis and will automatically project to `theta/phi`, `RHCP/LHCP`, Ludwig III, or `apolar` when `request_data()` is called. No manual conversion is needed.

## Normalization

Far-field data is normalized to gain units via:

```
E_gain = E_farfield * sqrt(2π / η₀)
```

where η₀ = sqrt(μ₀/ε₀) ≈ 376.73 Ω is the impedance of free space, derived from `scipy.constants`. This converts EMerge's complex far-field components to a dimensionless gain field consistent with xant's internal convention.