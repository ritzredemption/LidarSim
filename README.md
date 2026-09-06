# LidarSim Synthetic LiDAR Demo

A packaged Unreal Engine demo that simulates an automotive LiDAR sensor in a
city environment, exports annotated point clouds, and a Python workbench for
launching the simulation and inspecting the results live.

Built with the **AutoSimGen** sensor simulation plugin for Unreal Engine 5.

> **This repository contains the tooling and documentation only.**
> The packaged simulation build is several gigabytes and is distributed
> separately — see [Download the simulation](#2-download-the-simulation).

---

## Contents

1. [What the simulation contains](#1-what-the-simulation-contains)
2. [Download the simulation](#2-download-the-simulation)
3. [Repository layout](#3-repository-layout)
4. [Requirements](#4-requirements)
5. [Quick start](#5-quick-start)
6. [Simulation controls](#6-simulation-controls)
7. [The Python workbench](#7-the-python-workbench)
8. [Exported file format](#8-exported-file-format)
9. [Configuring the LiDAR](#9-configuring-the-lidar)
10. [Troubleshooting](#10-troubleshooting)
11. [License](#11-license)

---

## 1. What the simulation contains

The demo is a drivable small  scene with a LiDAR sensor mounted on the vehicle.
It is built to show what the sensor model produces rather than to be a game, so
everything is arranged around inspecting sensor output.

### LiDAR scanning

A rotating multi-channel sensor scans continuously as you drive. Returns carry
more than geometry:

- **Range and position** for every return, with a blind zone near the housing
  and a maximum range, as a real unit has
- **Intensity**, derived from surface incidence angle and range falloff, so
  surfaces angled away from the sensor return weaker
- **Ring index**, identifying which vertical channel produced each point, the
  same way a real driver reports it
- **Per-point timing** within the frame, which is what makes motion distortion
  reproducible a rotating sensor samples different directions at different
  instants
- **Semantic labels** for vehicles and pedestrians, so exports are usable as
  annotated data without a separate labelling pass

### Bounding box annotation

3D bounding boxes can be drawn around objects in the scene. These come from the
scene itself rather than from inference, so they act as ground truth: you can
see exactly which objects a perception system ought to find, and compare that
against what the point cloud actually resolves.

### Adverse weather

Weather degrades the scene the way it degrades real sensors. Rain, fog and snow
reduce effective range, thin out returns at distance, and cause
low-reflectivity targets to drop out first.

Running the bounding box overlay under adverse weather shows the gap opening up
between what is present in the scene and what remains detectable in the
returns. That gap is the case that matters most when validating a perception
stack, and it is the main thing this demo exists to show.

### Levels

Several levels are reachable from the in-simulation menu. Some are arranged to
demonstrate LiDAR scanning across different environments and geometry; others
focus on bounding box annotation and the effect of weather. You can move
between them freely without restarting.

---

## 2. Download the simulation

The packaged Windows build is hosted externally because of its size.

**Download:** *[https://drive.google.com/file/d/1xn5-PIi2vCpeT7G4sycFiuqEGzJhOkj_/view?usp=drive_link]*

Extract it so that the `LidarExe` folder sits next to the Python scripts:

```
LidarSim/
    AutoSimExe.py
    lidar_viewer.py
    README.md
    LidarExe/                     <- extracted here
        Windows/
            AutoSimGen.exe
            PluginTest55/
```

Nothing needs configuring. Both the executable and the exported point cloud are
located relative to the script, so the folder can live anywhere on any machine.

Demo videos of the simulation and of the weather interaction model are included
in the download.

---

## 3. Repository layout

| File | Purpose |
|------|---------|
| `AutoSimExe.py` | Full workbench launches and embeds the simulation, with three live point cloud views alongside it |
| `lidar_viewer.py` | Standalone viewer opens and inspects a `.ply` without running the simulation |
| `README.md` | This file |
| `LICENSE` | License terms |
| `.gitignore` | Excludes the build, media and generated point clouds |

The build, demo videos and exported point clouds are deliberately absent. They
are either too large for a repository or regenerated on every run.

---

## 4. Requirements

**Simulation**

- Windows 10 or 11, 64-bit
- A discrete GPU is strongly recommended. The sensor casts thousands of rays per
  frame; an RTX card handles this comfortably. It will run on integrated
  graphics, but expect low frame rates and a visibly sparser scan.
- Roughly 20 GB free disk space

**Python workbench**

- Python 3.9 or newer
- Three packages:

  ```
  pip install PyQt6 open3d numpy
  ```

---

## 5. Quick start

```
python AutoSimExe.py
```

1. Click **Launch Simulation**. The Unreal window opens inside the workbench.
2. Drive around. The LiDAR scans continuously.
3. Press **P** to export the current point cloud.
4. All three point cloud views update within half a second.

Press **P** as often as you like each export replaces the previous file and
the views follow, so it behaves as a live feed.

To inspect an existing capture without running the simulation:

```
python lidar_viewer.py                      # loads the most recent export
python lidar_viewer.py path\to\file.ply
```

---

## 6. Simulation controls

| Key | Action |
|-----|--------|
| `P` | Export the current point cloud to `PointCloudOutput.ply` |
| `M` | Open the menu |

**`P` export.** Writes the point cloud next to the executable. Each press
overwrites the previous file, so copy it under another name to keep a
particular capture.

**`M` menu.** Switch levels and toggle the bounding box overlay. Levels are
grouped by what they demonstrate: LiDAR scanning, or bounding boxes under
adverse conditions.

Click the simulation pane before pressing keys keyboard focus follows the
last pane you clicked.

---

## 7. The Python workbench

`AutoSimExe.py` puts the simulation and its output side by side: controls on the
left, the simulation in the middle, and a column of three point cloud views on
the right, each locked to a different angle.

### Live reload

On by default. The workbench watches the export file and reloads whenever the
simulation rewrites it. Camera angles, zoom, colour mode, point size and filters
all survive a reload, so a view set up once stays put across captures. If an
export is caught mid-write, the previous cloud stays on screen and the load is
retried rather than flashing a truncated frame.

### Colour channels

| Channel | Shows |
|---------|-------|
| Intensity | Return strength |
| Ring / line | Which emitter channel produced each point |
| Time in frame | When during the sweep each point was sampled |
| Height (Z) | Vertical position |
| Distance | Range from the sensor |
| Semantic label | Vehicle, pedestrian, background |

*Time in frame* is the interesting one: it makes the scan pattern visible. A
rotating sensor produces a gradient sweeping around the azimuth, because it
samples one azimuth at a time rather than the whole scene at once.

### Filters

Minimum and maximum range isolate a distance band. Semantic class checkboxes
hide whole classes unticking *background* leaves only labelled objects, which
is the quickest way to see how much of a vehicle or pedestrian actually survives
at range or in bad weather.

### Statistics

Point count, range span, intensity span, ring count, frame duration, and a
per-class breakdown, updated on every reload.

---

## 8. Exported file format

`PointCloudOutput.ply` is ASCII PLY with these per-point properties:

| Property | Meaning |
|----------|---------|
| `x`, `y`, `z` | Position in Unreal world space, centimetres |
| `intensity` | Return strength, 0 to 1 |
| `ring` | Emitter channel index |
| `time` | Seconds since the start of the frame |
| `label` | 0 background, 1 vehicle, 2 pedestrian |

Any PLY tool reads the positions. Note that most readers including Open3D's
own keep only `x`, `y`, `z` and silently discard the rest, which is why the
tools here parse the file directly.

If you process this data further, use the per-point `time` rather than a single
frame timestamp when compensating for platform motion. A rotating sensor skews a
moving scene across the frame, and that skew is reproduced here.

Coordinates are left-handed with Z up, in centimetres. Converting to a
right-handed metre frame is a Y flip and a divide by 100; the viewer has a
toggle for it.

---

## 9. Configuring the LiDAR

**The demo runs a fixed sensor configuration and cannot be reconfigured.**

It is a compiled build, so the sensor's Blueprint properties are not reachable
from outside it. What you see is one preset.

The **AutoSimGen plugin** exposes the sensor as a Blueprint component with
editable properties: channel count, horizontal and vertical field of view,
angular resolution, minimum and maximum range, rotation rate, range noise,
semantic class rules and export settings. Weather and bounding box generation
are configurable the same way.

To use the sensor in your own scenes, get the plugin:

**Fab marketplace:** *[add your Fab listing link here]*
Publisher: **ritzredemption**

---

## 10. Troubleshooting

**The Launch button is greyed out.** No executable was found. Check that
`LidarExe` was extracted next to the Python scripts, or use **Browse for
executable ** to point at it directly.

**The simulation opens in its own window instead of inside the workbench.**
Embedding is a Windows-only convenience and does not always succeed. The header
above the pane reports why. Everything else works normally; the two windows are
simply not joined.

**The simulation runs slowly.** Ray count dominates the cost, and it scales with
channel count multiplied by angular resolution. Lowering either requires the
plugin, as above. Closing other GPU applications helps.

**`P` does not produce a file.** The export is written next to the executable,
not the folder you launched from. If the sensor has not returned anything yet
the export is skipped, so move through the scene first.

**The viewer says it is waiting for the file.** Nothing has been exported yet.
Press `P` in the simulation and it will be picked up automatically.

**A reload shows nothing.** Range filters or class checkboxes may exclude
everything in the new capture. Reset the maximum range and re-tick the classes.

**`ModuleNotFoundError` on start.**

  ```
  pip install PyQt6 open3d numpy
  ```

---

## 11. License

See [LICENSE](LICENSE).

The plugin used to build this demo is a separate commercial product with its own
terms; see the Fab listing.
