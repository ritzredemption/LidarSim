"""
AutoSensData — LiDAR live viewer.

Launches the packaged Unreal demo and shows the exported point cloud in three
simultaneous viewports, each locked to a different angle. Press P in the demo
and every view updates within half a second.

Both the executable and the point cloud are resolved relative to this script,
so the whole folder can be copied to any machine and run without editing paths:

    <script dir>/
        lidar_viewer.py
        LidarExe/Windows/AutoSensData.exe
        LidarExe/Windows/PluginTest55/PointCloudOutput.ply

The 3D views use Open3D's legacy Visualizer, embedded into this window as child
windows. That renderer is used deliberately: pyqtgraph's GL scatter and
Open3D's Filament-based gui module both fail on some Windows GPU setups.

    pip install PyQt6 open3d numpy
    python lidar_viewer.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------- layout

SCRIPT_DIR = Path(__file__).resolve().parent

RELATIVE_EXE = Path("LidarExe") / "Windows" / "AutoSensData.exe"
RELATIVE_PLY = Path("LidarExe") / "Windows" / "PluginTest55" / "PointCloudOutput.ply"

EXE_FILENAME = RELATIVE_EXE.name
PLY_FILENAME = RELATIVE_PLY.name


def _find_relative(expected: Path, filename: str, max_depth: int = 6) -> Path:
    """
    Resolve a file relative to the script directory.

    Falls back to a shallow search so a build placed in a differently named
    folder is still found. If nothing exists yet the expected path is returned,
    which lets the file watcher pick the file up once it appears.
    """
    if expected.exists():
        return expected

    for depth in range(1, max_depth + 1):
        pattern = "/".join(["*"] * depth) + "/" + filename
        for candidate in sorted(SCRIPT_DIR.glob(pattern)):
            return candidate

    return expected


def resolve_exe() -> Path:
    return _find_relative(SCRIPT_DIR / RELATIVE_EXE, EXE_FILENAME)


def resolve_ply() -> Path:
    return _find_relative(SCRIPT_DIR / RELATIVE_PLY, PLY_FILENAME)




SEMANTIC_LABELS = {0: "background", 1: "vehicle", 2: "pedestrian"}

COLOUR_MODES = [
    ("Intensity", "intensity"),
    ("Ring / line", "ring"),
    ("Time in frame", "time"),
    ("Height (Z)", "height"),
    ("Distance", "distance"),
    ("Semantic label", "label"),
]

# front = direction from the look-at target towards the camera.
VIEW_PRESETS = {
    "Top":   ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    "Front": ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    "Side":  ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    "Iso":   ([-0.6, -0.6, 0.5], [0.0, 0.0, 1.0]),
}

# The three panes, in order, with the angle each opens at.
PANES = [
    ("TOP",   "Top",   "o3d_pane_top"),
    ("FRONT", "Front", "o3d_pane_front"),
    ("ISO",   "Iso",   "o3d_pane_iso"),
]

DEFAULT_ZOOM = 0.35

_VIRIDIS = np.array([
    [0.267, 0.005, 0.329], [0.283, 0.141, 0.458], [0.254, 0.265, 0.530],
    [0.207, 0.372, 0.553], [0.164, 0.471, 0.558], [0.128, 0.567, 0.551],
    [0.135, 0.659, 0.518], [0.267, 0.749, 0.441], [0.478, 0.821, 0.318],
    [0.741, 0.873, 0.150], [0.993, 0.906, 0.144],
])

_TURBO = np.array([
    [0.190, 0.072, 0.232], [0.246, 0.386, 0.907], [0.180, 0.671, 0.973],
    [0.110, 0.880, 0.744], [0.400, 0.980, 0.416], [0.750, 0.965, 0.212],
    [0.947, 0.813, 0.196], [0.994, 0.577, 0.130], [0.936, 0.312, 0.048],
    [0.780, 0.126, 0.014], [0.480, 0.016, 0.011],
])

_LABEL_COLOURS = np.array([
    [0.85, 0.85, 0.88],   # 0 background
    [0.22, 1.00, 0.08],   # 1 vehicle
    [0.00, 0.47, 1.00],   # 2 pedestrian
    [1.00, 0.55, 0.10],
    [0.90, 0.20, 0.55],
    [0.55, 0.35, 0.95],
    [0.15, 0.85, 0.85],
    [0.95, 0.90, 0.25],
])

STYLESHEET = """
QWidget { background-color: #000000; color: #d6d8dd;
          font-family: 'Segoe UI', sans-serif; font-size: 12px; }
QGroupBox { border: 1px solid #23252b; border-radius: 5px;
            margin-top: 10px; padding-top: 10px; }
QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 5px;
                   color: #6f7681; font-size: 10px;
                   font-weight: 600; letter-spacing: 1px; }
QPushButton { background-color: #14151a; border: 1px solid #2b2e36;
              border-radius: 4px; padding: 6px 10px; }
QPushButton:hover { background-color: #1e2029; border-color: #3a3e49; }
QPushButton:pressed { background-color: #0d0e12; }
QPushButton:disabled { color: #4a4d55; border-color: #1c1e24; }
QPushButton#primary { background-color: #1b4d3a; border-color: #2a7355;
                      font-weight: 600; }
QPushButton#primary:hover { background-color: #226147; }
QPushButton#danger { background-color: #4a1f22; border-color: #7a3238; }
QPushButton#danger:hover { background-color: #5c272b; }
QComboBox, QDoubleSpinBox { background-color: #14151a; border: 1px solid #2b2e36;
                            border-radius: 4px; padding: 4px 6px; }
QComboBox::drop-down { border: none; width: 18px; }
QComboBox QAbstractItemView { background-color: #14151a; border: 1px solid #2b2e36;
                              selection-background-color: #226147; }
QSlider::groove:horizontal { height: 3px; background: #2b2e36; border-radius: 2px; }
QSlider::handle:horizontal { background: #3ec98a; width: 12px; height: 12px;
                             margin: -5px 0; border-radius: 6px; }
QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #3a3e49;
                       border-radius: 3px; background: #14151a; }
QCheckBox::indicator:checked { background: #3ec98a; border-color: #3ec98a; }
QLabel#paneTag { color: #3ec98a; font-size: 10px; font-weight: 700;
                 letter-spacing: 2px; padding: 3px 8px; }
QLabel#stats { font-family: Consolas, 'Courier New', monospace;
               font-size: 11px; color: #9aa0aa; }
QLabel#hint { color: #6f7681; font-size: 11px; }
QStatusBar { color: #6f7681; border-top: 1px solid #23252b; }
QSplitter::handle { background: #1a1c21; }
"""


# ---------------------------------------------------------------- parsing

def read_ply(path: Path) -> dict:
    """
    Parse the PLY written by the LiDAR component.

    Open3D's own reader keeps only XYZ and discards intensity, ring, time and
    label, which are exactly the channels worth colouring by.
    """
    names: list[str] = []
    header_lines = 0
    vertex_count = 0
    in_vertex = False

    with open(path, "r", errors="replace") as handle:
        if handle.readline().strip() != "ply":
            raise ValueError("not a PLY file")
        header_lines = 1

        for line in handle:
            header_lines += 1
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format" and parts[1] != "ascii":
                raise ValueError(f"'{parts[1]}' encoding; this reads ASCII PLY only")
            if parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                names.append(parts[2])
            elif parts[0] == "end_header":
                break
        else:
            raise ValueError("no end_header line")

    raw = np.loadtxt(path, skiprows=header_lines, max_rows=vertex_count or None, ndmin=2)

    if raw.shape[1] != len(names):
        raise ValueError(f"header declares {len(names)} properties, rows have {raw.shape[1]}")

    columns = {name: raw[:, index] for index, name in enumerate(names)}
    for axis in ("x", "y", "z"):
        if axis not in columns:
            raise ValueError(f"missing '{axis}' property")

    xyz = np.column_stack((columns["x"], columns["y"], columns["z"]))

    
    columns["ranges"] = np.linalg.norm(xyz, axis=1)
    columns["origin"] = xyz.mean(axis=0)
    columns["xyz"] = xyz - columns["origin"]

    return columns


# ---------------------------------------------------------------- colour

def _ramp(values: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    positions = np.clip(values, 0.0, 1.0) * (len(anchors) - 1)
    lower = np.floor(positions).astype(np.int32)
    upper = np.minimum(lower + 1, len(anchors) - 1)
    blend = (positions - lower)[:, None]
    return anchors[lower] * (1.0 - blend) + anchors[upper] * blend


def _normalise(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64)
    low, high = float(values.min()), float(values.max())
    if high - low <= 1e-9:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def make_colours(columns: dict, indices: np.ndarray, mode: str) -> np.ndarray:
    grey = np.tile([0.8, 0.8, 0.85], (len(indices), 1))

    if mode == "label":
        if "label" not in columns:
            return grey
        labels = columns["label"][indices].astype(np.int32)
        return _LABEL_COLOURS[labels % len(_LABEL_COLOURS)]

    if mode == "height":
        channel = columns["xyz"][indices, 2]
    elif mode == "distance":
        channel = columns["ranges"][indices]
    elif mode in columns:
        channel = columns[mode][indices]
    else:
        return grey

    anchors = _TURBO if mode in ("ring", "time") else _VIRIDIS
    return _ramp(_normalise(channel), anchors)




class RenderSurface(QWidget):
    """
    Hosts one Open3D window as a native child window.

    The legacy Visualizer creates its own top-level GLFW window with no API to
    reparent it, so on Windows it is located by title and adopted with
    SetParent. If that fails the Open3D window stays separate and everything
    else still works.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(150)
        self.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
        self.child_hwnd = None
        self._user32 = None

    def adopt(self, title: str) -> bool:
        if sys.platform != "win32":
            return False

        try:
            import ctypes
            from ctypes import wintypes

            user32 = ctypes.windll.user32
            user32.FindWindowW.restype = wintypes.HWND
            user32.FindWindowW.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR]
            user32.SetParent.restype = wintypes.HWND
            user32.SetParent.argtypes = [wintypes.HWND, wintypes.HWND]
            user32.GetWindowLongPtrW.restype = ctypes.c_longlong
            user32.GetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int]
            user32.SetWindowLongPtrW.restype = ctypes.c_longlong
            user32.SetWindowLongPtrW.argtypes = [wintypes.HWND, ctypes.c_int, ctypes.c_longlong]
            user32.MoveWindow.argtypes = [
                wintypes.HWND, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, wintypes.BOOL,
            ]

            hwnd = user32.FindWindowW(None, title)
            if not hwnd:
                return False

            GWL_STYLE = -16
            WS_CHILD = 0x40000000
            WS_POPUP = 0x80000000
            WS_CAPTION = 0x00C00000
            WS_THICKFRAME = 0x00040000
            WS_VISIBLE = 0x10000000

            style = user32.GetWindowLongPtrW(hwnd, GWL_STYLE)
            style &= ~(WS_POPUP | WS_CAPTION | WS_THICKFRAME)
            style |= WS_CHILD | WS_VISIBLE
            user32.SetWindowLongPtrW(hwnd, GWL_STYLE, style)

            if not user32.SetParent(hwnd, int(self.winId())):
                return False

            self.child_hwnd = hwnd
            self._user32 = user32
            self._fit_child()
            return True

        except Exception:
            return False

    def _fit_child(self) -> None:
        if self.child_hwnd and self._user32:
            self._user32.MoveWindow(
                self.child_hwnd, 0, 0, max(self.width(), 1), max(self.height(), 1), True
            )

    def resizeEvent(self, event) -> None:  # noqa: N802 — Qt naming
        super().resizeEvent(event)
        self._fit_child()


class ViewPane:
    """One labelled viewport: an Open3D visualizer plus its host widget."""

    def __init__(self, tag: str, preset: str, title: str) -> None:
        self.tag = tag
        self.preset = preset
        self.title = title
        self.zoom = DEFAULT_ZOOM
        self.geometry_added = False

        self.container = QFrame()
        self.container.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(self.container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setFixedHeight(22)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 6, 0)

        self.tag_label = QLabel(tag)
        self.tag_label.setObjectName("paneTag")
        header_layout.addWidget(self.tag_label)
        header_layout.addStretch()

        self.preset_combo = QComboBox()
        self.preset_combo.addItems(list(VIEW_PRESETS))
        self.preset_combo.setCurrentText(preset)
        self.preset_combo.setFixedWidth(78)
        header_layout.addWidget(self.preset_combo)

        layout.addWidget(header)

        self.surface = RenderSurface()
        layout.addWidget(self.surface, stretch=1)

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=title, width=760, height=300)

        options = self.vis.get_render_option()
        options.background_color = np.array([0.0, 0.0, 0.0])
        options.point_size = 2.0

    def apply_camera(self) -> None:
        front, up = VIEW_PRESETS[self.preset_combo.currentText()]
        control = self.vis.get_view_control()
        control.set_lookat([0.0, 0.0, 0.0])   # the cloud is centred there
        control.set_front(front)
        control.set_up(up)
        control.set_zoom(self.zoom)

    def set_point_size(self, size: float) -> None:
        self.vis.get_render_option().point_size = float(size)




class Viewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("AutoSensData — LiDAR Live Viewer")
        self.resize(1680, 980)
        self.setAcceptDrops(True)

        self.columns: dict | None = None
        self.source_path: Path | None = None
        self.exe_path = resolve_exe()
        self.metres = False
        self.last_mtime = 0.0
        self.reload_failures = 0
        self.process: subprocess.Popen | None = None

        self.geometry = o3d.geometry.PointCloud()
        self.panes = [ViewPane(tag, preset, title) for tag, preset, title in PANES]

        for pane in self.panes:
            pane.preset_combo.currentTextChanged.connect(
                lambda _text, p=pane: p.apply_camera()
            )

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._build_panel())

        self.view_splitter = QSplitter(Qt.Orientation.Vertical)
        for pane in self.panes:
            self.view_splitter.addWidget(pane.container)
        self.view_splitter.setSizes([320, 320, 320])
        layout.addWidget(self.view_splitter, stretch=1)

        self.setCentralWidget(central)

        # The Open3D windows must exist and Qt must have realised the hosts
        # before adoption can work, so this runs once the event loop is up.
        QTimer.singleShot(200, self._adopt_render_windows)

        self.render_timer = QTimer(self)
        self.render_timer.timeout.connect(self._pump)
        self.render_timer.start(16)

        # Polled separately from the render loop so a slow disk never stutters
        # the views.
        self.watch_timer = QTimer(self)
        self.watch_timer.timeout.connect(self._check_for_changes)
        self.watch_timer.start(500)

        self._set_controls_enabled(False)
        self._refresh_launch_state()

    # ---- panel ----

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(340)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(9)

        title = QLabel("AUTOSENSDATA")
        title.setStyleSheet(
            "color: #3ec98a; font-size: 15px; font-weight: 700; letter-spacing: 3px;"
        )
        layout.addWidget(title)

        subtitle = QLabel("LiDAR live viewer")
        subtitle.setObjectName("hint")
        layout.addWidget(subtitle)

        # --- simulation ---
        sim = QGroupBox("SIMULATION")
        sim_layout = QVBoxLayout(sim)

        self.launch_button = QPushButton("▶  Launch Unreal Demo")
        self.launch_button.setObjectName("primary")
        self.launch_button.setMinimumHeight(36)
        self.launch_button.clicked.connect(self.launch_demo)
        sim_layout.addWidget(self.launch_button)

        self.stop_button = QPushButton("■  Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.clicked.connect(self.stop_demo)
        self.stop_button.setEnabled(False)
        sim_layout.addWidget(self.stop_button)

        self.exe_status = QLabel("")
        self.exe_status.setObjectName("hint")
        self.exe_status.setWordWrap(True)
        sim_layout.addWidget(self.exe_status)

        hint = QLabel("Press  P  in the demo to export a scan.\nAll three views update automatically.")
        hint.setObjectName("hint")
        sim_layout.addWidget(hint)

        layout.addWidget(sim)

        # --- source ---
        source = QGroupBox("POINT CLOUD")
        source_layout = QVBoxLayout(source)

        self.open_button = QPushButton("Open a different PLY…")
        self.open_button.clicked.connect(self.open_file_dialog)
        source_layout.addWidget(self.open_button)

        self.live_check = QCheckBox("Live reload")
        self.live_check.setChecked(True)
        source_layout.addWidget(self.live_check)

        self.live_status = QLabel("waiting…")
        self.live_status.setObjectName("hint")
        self.live_status.setWordWrap(True)
        source_layout.addWidget(self.live_status)

        layout.addWidget(source)

        # --- appearance ---
        appearance = QGroupBox("APPEARANCE")
        appearance_layout = QVBoxLayout(appearance)

        appearance_layout.addWidget(QLabel("Colour by"))
        self.colour_combo = QComboBox()
        self.colour_combo.addItems([label for label, _ in COLOUR_MODES])
        self.colour_combo.currentIndexChanged.connect(self.refresh)
        appearance_layout.addWidget(self.colour_combo)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Point size"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(1, 10)
        self.size_slider.setValue(2)
        self.size_slider.valueChanged.connect(self._on_point_size)
        size_row.addWidget(self.size_slider)
        appearance_layout.addLayout(size_row)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(4, 200)              # zoom * 100
        self.zoom_slider.setValue(int(DEFAULT_ZOOM * 100))
        self.zoom_slider.setInvertedAppearance(True)   # drag right to zoom in
        self.zoom_slider.valueChanged.connect(self._on_zoom)
        zoom_row.addWidget(self.zoom_slider)
        appearance_layout.addLayout(zoom_row)

        reset_button = QPushButton("Reset all views")
        reset_button.clicked.connect(self.reset_views)
        appearance_layout.addWidget(reset_button)

        layout.addWidget(appearance)

        # --- filters ---
        filters = QGroupBox("FILTERS")
        filters_layout = QGridLayout(filters)

        filters_layout.addWidget(QLabel("Min range"), 0, 0)
        self.near_spin = QDoubleSpinBox()
        self.near_spin.setRange(0.0, 1e7)
        self.near_spin.setDecimals(1)
        self.near_spin.valueChanged.connect(self.refresh)
        filters_layout.addWidget(self.near_spin, 0, 1)

        filters_layout.addWidget(QLabel("Max range"), 1, 0)
        self.far_spin = QDoubleSpinBox()
        self.far_spin.setRange(0.0, 1e7)
        self.far_spin.setDecimals(1)
        self.far_spin.setValue(1e7)
        self.far_spin.valueChanged.connect(self.refresh)
        filters_layout.addWidget(self.far_spin, 1, 1)

        self.label_box = QVBoxLayout()
        self.label_checks: dict[int, QCheckBox] = {}
        filters_layout.addLayout(self.label_box, 2, 0, 1, 2)

        self.metres_check = QCheckBox("Metres instead of centimetres")
        self.metres_check.toggled.connect(self._on_units)
        filters_layout.addWidget(self.metres_check, 3, 0, 1, 2)

        layout.addWidget(filters)

        # --- stats ---
        self.stats_label = QLabel("No cloud loaded.")
        self.stats_label.setObjectName("stats")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.stats_label, stretch=1)

        return panel

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.colour_combo, self.size_slider, self.zoom_slider,
                       self.near_spin, self.far_spin, self.metres_check):
            widget.setEnabled(enabled)

    # ---- unreal process ----

    def _refresh_launch_state(self) -> None:
        self.exe_path = resolve_exe()
        running = self.process is not None and self.process.poll() is None

        if running:
            self.exe_status.setText("running")
            self.launch_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            return

        self.stop_button.setEnabled(False)

        if self.exe_path.exists():
            try:
                shown = self.exe_path.relative_to(SCRIPT_DIR)
            except ValueError:
                shown = self.exe_path
            self.exe_status.setText(str(shown))
            self.launch_button.setEnabled(True)
        else:
            self.exe_status.setText(f"{EXE_FILENAME} not found next to this script")
            self.launch_button.setEnabled(False)

    def launch_demo(self) -> None:
        exe = resolve_exe()
        if not exe.exists():
            QMessageBox.critical(self, "Not found", f"Could not find {exe}")
            return

        try:
            # cwd must be the executable's own folder: Unreal resolves its
            # content paths relative to it, and writes the export there too.
            self.process = subprocess.Popen([str(exe)], cwd=str(exe.parent))
        except Exception as error:  # noqa: BLE001 — surfaced to the user
            QMessageBox.critical(self, "Could not launch", f"{exe.name}\n\n{error}")
            return

        self.statusBar().showMessage(f"Launched {exe.name}")
        self._refresh_launch_state()

    def stop_demo(self) -> None:
        if self.process is None:
            return
        try:
            self.process.terminate()
        except Exception:
            pass
        self.process = None
        self.statusBar().showMessage("Demo stopped.")
        self._refresh_launch_state()

    # ---- embedding ----

    def _adopt_render_windows(self) -> None:
        adopted = sum(pane.surface.adopt(pane.title) for pane in self.panes)

        if adopted == len(self.panes):
            self.statusBar().showMessage("Ready.")
        else:
            self.statusBar().showMessage(
                f"{adopted}/{len(self.panes)} views embedded; "
                "the rest are separate windows."
            )

    def _pump(self) -> None:
        for pane in self.panes:
            if not pane.vis.poll_events():
                self.render_timer.stop()
                self.close()
                return
            pane.vis.update_renderer()

        if self.process is not None and self.process.poll() is not None:
            self.process = None
            self._refresh_launch_state()

    # ---- loading ----

    def open_file_dialog(self) -> None:
        start = str(self.source_path.parent) if self.source_path else str(SCRIPT_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self, "Open point cloud", start, "PLY files (*.ply);;All files (*)"
        )
        if path:
            self.load(Path(path))

    def load(self, path: Path) -> None:
        try:
            columns = read_ply(path)
        except Exception as error:  # noqa: BLE001 — surfaced to the user
            QMessageBox.critical(self, "Could not load", f"{path.name}\n\n{error}")
            return

        if columns["xyz"].shape[0] == 0:
            QMessageBox.warning(self, "Empty file", f"{path.name} contains no points.")
            return

        self.columns = columns
        self.source_path = path
        self.metres = False
        self.metres_check.blockSignals(True)
        self.metres_check.setChecked(False)
        self.metres_check.blockSignals(False)

        try:
            self.last_mtime = os.path.getmtime(path)
        except OSError:
            self.last_mtime = 0.0
        self.reload_failures = 0

        self._rebuild_label_filters()
        self._configure_ranges()
        self._set_controls_enabled(True)

        self.refresh(refit=True)
        self.reset_views()
        self.live_status.setText(f"{path.name} · {columns['xyz'].shape[0]:,} points")

    def _check_for_changes(self) -> None:
        """Reload when the watched file's timestamp moves."""
        if self.source_path is None or not self.live_check.isChecked():
            return

        try:
            mtime = os.path.getmtime(self.source_path)
        except OSError:
            self.live_status.setText(f"waiting for {self.source_path.name}…")
            return

        if mtime == self.last_mtime:
            return

        try:
            columns = read_ply(self.source_path)
        except Exception:
            # Unreal may be mid-write. Leave last_mtime alone and retry on the
            # next tick rather than showing a truncated cloud.
            self.reload_failures += 1
            self.live_status.setText(f"writing… ({self.reload_failures})")
            return

        if columns["xyz"].shape[0] == 0:
            return

        if self.columns is None:
            # First appearance of the file: full load, so the cameras get framed
            # and the class filters are built.
            self.load(self.source_path)
            return

        self.last_mtime = mtime
        self.reload_failures = 0

        if self.metres:
            for key in ("xyz", "ranges", "origin"):
                columns[key] = columns[key] * 0.01

        previous_labels = set(self.label_checks)
        self.columns = columns

        # Only rebuild the class checkboxes if the classes changed, so a reload
        # does not silently re-tick boxes you unticked.
        if "label" in columns:
            current_labels = set(np.unique(columns["label"].astype(np.int32)).tolist())
        else:
            current_labels = set()
        if current_labels != previous_labels:
            self._rebuild_label_filters()

        # Camera, colour mode, point size and filters are all left untouched.
        self.refresh()
        self.live_status.setText(f"updated · {columns['xyz'].shape[0]:,} points")

    def _on_units(self, metres: bool) -> None:
        if self.columns is None:
            return

        scale = 0.01 if metres else 100.0
        for key in ("xyz", "ranges", "origin"):
            self.columns[key] = self.columns[key] * scale
        self.metres = metres

        self._configure_ranges()
        self.refresh(refit=True)
        self.reset_views()

    def _rebuild_label_filters(self) -> None:
        while self.label_box.count():
            item = self.label_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.label_checks.clear()

        if self.columns is None or "label" not in self.columns:
            return

        labels = self.columns["label"].astype(np.int32)
        for value in np.unique(labels):
            value = int(value)
            name = SEMANTIC_LABELS.get(value, f"class {value}")
            count = int(np.count_nonzero(labels == value))

            check = QCheckBox(f"{name}  ({count:,})")
            check.setChecked(True)
            check.toggled.connect(self.refresh)
            self.label_box.addWidget(check)
            self.label_checks[value] = check

    def _configure_ranges(self) -> None:
        assert self.columns is not None
        far = float(self.columns["ranges"].max()) * 1.01

        for spin in (self.near_spin, self.far_spin):
            spin.blockSignals(True)
        self.near_spin.setValue(0.0)
        self.far_spin.setValue(far)
        step = max(far / 100.0, 0.01)
        self.near_spin.setSingleStep(step)
        self.far_spin.setSingleStep(step)
        for spin in (self.near_spin, self.far_spin):
            spin.blockSignals(False)

    # ---- display ----

    def _visible_indices(self) -> np.ndarray:
        assert self.columns is not None

        ranges = self.columns["ranges"]
        mask = (ranges >= self.near_spin.value()) & (ranges <= self.far_spin.value())

        if "label" in self.columns and self.label_checks:
            keep = [value for value, check in self.label_checks.items() if check.isChecked()]
            mask &= np.isin(self.columns["label"].astype(np.int32), keep)

        return np.flatnonzero(mask)

    def refresh(self, *_args, refit: bool = False) -> None:
        if self.columns is None:
            return

        indices = self._visible_indices()
        mode = COLOUR_MODES[self.colour_combo.currentIndex()][1]

        # One geometry object shared by all three visualizers, so the point data
        # is built once per update rather than three times.
        self.geometry.points = o3d.utility.Vector3dVector(self.columns["xyz"][indices])
        if len(indices):
            self.geometry.colors = o3d.utility.Vector3dVector(
                make_colours(self.columns, indices, mode)
            )

        for pane in self.panes:
            if not pane.geometry_added:
                pane.vis.add_geometry(self.geometry, reset_bounding_box=True)
                pane.geometry_added = True
                pane.apply_camera()
            elif refit:
                # Refit only for a new file or a unit change. Colour and filter
                # changes update in place, leaving each camera alone.
                pane.vis.clear_geometries()
                pane.vis.add_geometry(self.geometry, reset_bounding_box=True)
                pane.apply_camera()
            else:
                pane.vis.update_geometry(self.geometry)

        self.statusBar().showMessage(
            f"{len(indices):,} of {self.columns['xyz'].shape[0]:,} points shown"
        )
        self._update_stats(len(indices))

    def _on_point_size(self, value: int) -> None:
        for pane in self.panes:
            pane.set_point_size(value)

    def _on_zoom(self, value: int) -> None:
        zoom = value / 100.0
        for pane in self.panes:
            pane.zoom = zoom
            pane.vis.get_view_control().set_zoom(zoom)

    def reset_views(self) -> None:
        for pane in self.panes:
            pane.zoom = self.zoom_slider.value() / 100.0
            pane.apply_camera()

    # ---- stats ----

    def _update_stats(self, shown: int) -> None:
        if self.columns is None:
            return

        ranges = self.columns["ranges"]
        units = "m" if self.metres else "cm"

        lines = [
            f"points  {self.columns['xyz'].shape[0]:,}",
            f"shown   {shown:,}",
            f"units   {units}",
            "",
            f"range   {ranges.min():.1f} .. {ranges.max():.1f}",
        ]

        if "intensity" in self.columns:
            channel = self.columns["intensity"]
            lines.append(f"intens  {channel.min():.3f} .. {channel.max():.3f}")
        if "ring" in self.columns:
            rings = self.columns["ring"].astype(int)
            lines.append(f"rings   {rings.min()}..{rings.max()} ({len(np.unique(rings))})")
        if "time" in self.columns:
            span = (self.columns["time"].max() - self.columns["time"].min()) * 1000.0
            lines.append(f"frame   {span:.2f} ms")

        if "label" in self.columns:
            lines.append("")
            values, counts = np.unique(self.columns["label"].astype(int), return_counts=True)
            for value, count in zip(values, counts):
                name = SEMANTIC_LABELS.get(int(value), f"class {int(value)}")
                lines.append(f"  {name:<11} {count:,}")

        self.stats_label.setText("\n".join(lines))

    # ---- plumbing ----

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt naming
        self.render_timer.stop()
        self.watch_timer.stop()

        if self.process is not None:
            try:
                self.process.terminate()
            except Exception:
                pass

        for pane in self.panes:
            try:
                pane.vis.destroy_window()
            except Exception:
                pass

        event.accept()

    def dragEnterEvent(self, event) -> None:  # noqa: N802 — Qt naming
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 — Qt naming
        for url in event.mimeData().urls():
            path = Path(url.toLocalFile())
            if path.suffix.lower() == ".ply":
                self.load(path)
                break


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    viewer = Viewer()
    viewer.show()

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else resolve_ply()
    viewer.source_path = target

    if target.exists():
        viewer.load(target)
    else:
        # Not an error: the watcher picks the file up as soon as it appears.
        viewer.live_status.setText(f"waiting for {target.name}…")
        viewer.statusBar().showMessage(f"Waiting for {target}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()