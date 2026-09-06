# LidarExe — simulation build

**This folder is intentionally empty in the repository.**

The packaged Unreal build is several gigabytes, past GitHub's file size limits,
so it is distributed separately.

## Download

**[Download the simulation build]()**  ← add your Drive link

## Install

Extract the archive so its contents land directly in this folder:

```
LidarSim/
    AutoSimExe.py
    lidar_viewer.py
    README.md
    LidarExe/
        README.md              <- this file
        Windows/               <- extracted here
            AutoSimGen.exe
            PluginTest55/
```

The Python tools locate the executable and the exported point cloud relative to
the script, so nothing needs configuring once the archive is in place. If your
build sits somewhere else, use **Browse for executable…** in the workbench.

Return to the [main README](../README.md) for usage.
