# xant Integrations

This folder contains integration scripts for exporting data from third-party EM simulation tools into xant. Each integration is a standalone script you can copy into your project — no additional installation is required beyond xant and the relevant simulator's Python API.

## Available Integrations

| Simulator | Folder | Description |
|----------:|:-------|:------------|
| [EMerge](https://www.emerge-em.com/) | [`emerge/`](emerge/) | Export far-field patterns and embedded element patterns from EMerge simulations |

## How to use

Copy the relevant script into your project and import it directly. Each integration folder contains its own README with usage examples and any simulator-specific requirements.

## Adding an integration

If you use xant with another simulator (CST, HFSS, FEKO, WIPL-D, etc.) and have an export script, contributions are welcome. The pattern is straightforward — convert the simulator's far-field output into an `xr.DataArray` with dims `(polarization, frequency, phi, theta)` and `attrs=dict(coordinate_frame="phitheta")`, then wrap it in `xant.Antenna`. See the EMerge integration for a worked example.