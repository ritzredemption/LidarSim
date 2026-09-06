"""
LiDAR point cloud viewer.

One window: 3D view on the left, controls on the right. Everything is a button,
dropdown or slider.

The 3D view is Open3D's legacy Visualizer, embedded into the Qt window as a
child window. That combination is deliberate: pyqtgraph's GL scatter and
Open3D's Filament-based gui module both fail on some Windows GPU setups, while
the legacy Visualizer works. Embedding gets widgets without changing renderer.

    pip install PyQt6 open3d numpy
    python viewer.py [file.ply]

Self-contained: no other files needed.
"""

from __future__ import annotations

import os
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
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

# The point cloud is written by the packaged build, which lives in a fixed
# subdirectory next to this script. Resolving it relative to the script keeps
# the tool portable: it works from any checkout on any machine, and does not
# care what the working directory is when it is launched.
#
#     <script dir>/
#         viewer.py
#         Windows/PluginTest55/PointCloudOutput.ply
#
SCRIPT_DIR = Path(__file__).resolve().parent
PLY_FILENAME = "PointCloudOutput.ply"
RELATIVE_PLY = Path("Windows") / "PluginTest55" / PLY_FILENAME


def resolve_watch_path() -> Path:
    """
    Locate the point cloud relative to this script.

    Falls back to a shallow search of the script directory, so a build placed
    in a differently named folder is still found rather than silently watching
    a path that will never exist.
    """
    expected = SCRIPT_DIR / RELATIVE_PLY
    if expected.exists():
        return expected

    for depth in range(1, 5):
        pattern = "/".join(["*"] * depth) + "/" + PLY_FILENAME
        for candidate in sorted(SCRIPT_DIR.glob(pattern)):
            return candidate

    # Nothing on disk yet. Return the expected location so the watcher picks the
    # file up as soon as the build writes it.
    return expected


SEMANTIC_LABELS = {0: "background", 1: "vehicle", 2: "pedestrian"}

COLOUR_MODES = [
    ("Intensity", "intensity"),
    ("Ring / line", "ring"),
    ("Time in frame", "time"),
    ("Height (Z)", "height"),
    ("Distance", "distance"),
    ("Semantic label", "label"),
]

# front = direction from the look-at target towards the camera
VIEWS = {
    "Top":   ([0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    "Front": ([-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    "Side":  ([0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    "Iso":   ([-0.6, -0.6, 0.5], [0.0, 0.0, 1.0]),
}

WINDOW_TITLE = "o3d_render_surface"
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


# ---------------------------------------------------------------- parsing

def read_ply(path: Path) -> dict:
    """
    Parse the PLY written by SavePointCloudToPLY.

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

    # Centre on the cloud's own centroid. Exports are in world space and can sit
    # many thousands of units from the origin, which makes framing awkward. True
    # ranges are kept so distances are still reported from the sensor.
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


# ------------------------------------------------------- window embedding

class RenderSurface(QWidget):
    """
    Hosts the Open3D window as a native child window.

    Open3D's legacy Visualizer creates its own top-level GLFW window with no API
    to reparent it, so on Windows it is located by title and adopted with
    SetParent. If that fails the Open3D window simply stays separate and
    everything else still works.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(640, 480)
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


# ---------------------------------------------------------------- window

class Viewer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("LiDAR Point Cloud Viewer")
        self.resize(1600, 950)
        self.setAcceptDrops(True)

        self.columns: dict | None = None
        self.source_path: Path | None = None
        self.metres = False
        self.view_name = "Top"
        self.zoom = DEFAULT_ZOOM
        self.geometry = o3d.geometry.PointCloud()
        self.geometry_added = False
        self.last_mtime = 0.0
        self.reload_failures = 0

        self.surface = RenderSurface()

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.surface, stretch=1)
        layout.addWidget(self._build_panel())
        self.setCentralWidget(central)

        self.statusBar().showMessage("Open a .ply file to begin.")

        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name=WINDOW_TITLE, width=1100, height=900)

        options = self.vis.get_render_option()
        options.background_color = np.array([0.0, 0.0, 0.0])
        options.point_size = 2.0

        # The Open3D window must exist before it can be adopted, and Qt must
        # have realised the host widget, so this runs after the event loop
        # has had a chance to show both.
        QTimer.singleShot(120, self._adopt_render_window)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._pump)
        self.timer.start(16)

        # Polled separately from the render loop so a slow disk never stutters
        # the view.
        self.watch_timer = QTimer(self)
        self.watch_timer.timeout.connect(self._check_for_changes)
        self.watch_timer.start(500)

        self._set_controls_enabled(False)

        # Watch the configured path immediately. The file does not have to
        # exist yet: once Unreal writes it, the watcher picks it up.
        self.source_path = resolve_watch_path()
        self.live_status.setText(f"watching {self.source_path.name}")

    # ---- panel ----

    def _build_panel(self) -> QWidget:
        panel = QWidget()
        panel.setFixedWidth(330)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        open_button = QPushButton("Open PLY…")
        open_button.setMinimumHeight(34)
        open_button.clicked.connect(self.open_file_dialog)
        layout.addWidget(open_button)

        # --- view ---
        view_group = QGroupBox("View")
        view_layout = QVBoxLayout(view_group)

        buttons = QGridLayout()
        for index, name in enumerate(VIEWS):
            button = QPushButton(name)
            button.clicked.connect(lambda _checked, n=name: self.set_view(n))
            buttons.addWidget(button, index // 2, index % 2)
        view_layout.addLayout(buttons)

        zoom_row = QHBoxLayout()
        zoom_row.addWidget(QLabel("Zoom"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(4, 200)          # zoom * 100
        self.zoom_slider.setValue(int(DEFAULT_ZOOM * 100))
        self.zoom_slider.setInvertedAppearance(True)   # right = closer
        self.zoom_slider.valueChanged.connect(self._on_zoom)
        zoom_row.addWidget(self.zoom_slider)
        view_layout.addLayout(zoom_row)

        layout.addWidget(view_group)

        # --- appearance ---
        appearance = QGroupBox("Appearance")
        appearance_layout = QVBoxLayout(appearance)

        appearance_layout.addWidget(QLabel("Colour by"))
        self.colour_combo = QComboBox()
        self.colour_combo.addItems([label for label, _ in COLOUR_MODES])
        self.colour_combo.currentIndexChanged.connect(self.refresh)
        appearance_layout.addWidget(self.colour_combo)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Point size"))
        self.size_slider = QSlider(Qt.Orientation.Horizontal)
        self.size_slider.setRange(1, 12)
        self.size_slider.setValue(2)
        self.size_slider.valueChanged.connect(self._on_point_size)
        size_row.addWidget(self.size_slider)
        appearance_layout.addLayout(size_row)

        layout.addWidget(appearance)

        # --- filters ---
        filters = QGroupBox("Filters")
        filters_layout = QVBoxLayout(filters)

        near_row = QHBoxLayout()
        near_row.addWidget(QLabel("Min range"))
        self.near_spin = QDoubleSpinBox()
        self.near_spin.setRange(0.0, 1e7)
        self.near_spin.setDecimals(1)
        self.near_spin.valueChanged.connect(self.refresh)
        near_row.addWidget(self.near_spin)
        filters_layout.addLayout(near_row)

        far_row = QHBoxLayout()
        far_row.addWidget(QLabel("Max range"))
        self.far_spin = QDoubleSpinBox()
        self.far_spin.setRange(0.0, 1e7)
        self.far_spin.setDecimals(1)
        self.far_spin.setValue(1e7)
        self.far_spin.valueChanged.connect(self.refresh)
        far_row.addWidget(self.far_spin)
        filters_layout.addLayout(far_row)

        filters_layout.addWidget(QLabel("Semantic classes"))
        self.label_box = QVBoxLayout()
        self.label_checks: dict[int, QCheckBox] = {}
        filters_layout.addLayout(self.label_box)

        layout.addWidget(filters)

        # --- live reload ---
        live = QGroupBox("Live reload")
        live_layout = QVBoxLayout(live)

        self.live_check = QCheckBox("Watch file for changes")
        self.live_check.setChecked(True)
        self.live_check.setToolTip(
            "Reloads whenever Unreal rewrites the file, keeping your camera,\n"
            "colour mode and filters."
        )
        live_layout.addWidget(self.live_check)

        self.live_status = QLabel("waiting…")
        self.live_status.setStyleSheet("color: #7a8088; font-size: 11px;")
        live_layout.addWidget(self.live_status)

        layout.addWidget(live)

        # --- units ---
        units = QGroupBox("Coordinate frame")
        units_layout = QVBoxLayout(units)
        self.metres_check = QCheckBox("Metres instead of centimetres")
        self.metres_check.toggled.connect(self._on_units)
        units_layout.addWidget(self.metres_check)
        layout.addWidget(units)

        # --- stats ---
        self.stats_label = QLabel("No cloud loaded.")
        self.stats_label.setStyleSheet("font-family: Consolas, monospace; font-size: 11px;")
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.stats_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.stats_label, stretch=1)

        return panel

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (self.colour_combo, self.size_slider, self.near_spin,
                       self.far_spin, self.metres_check, self.zoom_slider):
            widget.setEnabled(enabled)

    # ---- embedding ----

    def _adopt_render_window(self) -> None:
        if self.surface.adopt(WINDOW_TITLE):
            self.statusBar().showMessage("Ready.")
        else:
            self.statusBar().showMessage(
                "3D view is a separate window (embedding unavailable on this platform)."
            )

    def _pump(self) -> None:
        if not self.vis.poll_events():
            self.timer.stop()
            self.close()
            return
        self.vis.update_renderer()

    # ---- loading ----

    def open_file_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open point cloud", "", "PLY files (*.ply);;All files (*)"
        )
        if path:
            self.load(Path(path))

    def _check_for_changes(self) -> None:
        """Reload when the watched file's timestamp moves."""
        if self.source_path is None or not self.live_check.isChecked():
            return

        try:
            mtime = os.path.getmtime(self.source_path)
        except OSError:
            self.live_status.setText("file missing")
            return

        if mtime == self.last_mtime:
            return

        try:
            columns = read_ply(self.source_path)
        except Exception:
            # Unreal may be mid-write. Leave last_mtime alone and try again on
            # the next tick rather than showing a truncated cloud.
            self.reload_failures += 1
            self.live_status.setText(f"waiting for write… ({self.reload_failures})")
            return

        if columns["xyz"].shape[0] == 0:
            return

        if self.columns is None:
            # First time the file has appeared: full load, so the camera frames
            # the cloud and the class filters get built.
            self.load(self.source_path)
            return

        self.last_mtime = mtime
        self.reload_failures = 0

        rescale = 0.01 if self.metres else 1.0
        if rescale != 1.0:
            columns["xyz"] = columns["xyz"] * rescale
            columns["ranges"] = columns["ranges"] * rescale
            columns["origin"] = columns["origin"] * rescale

        previous_labels = set(self.label_checks)
        self.columns = columns

        # Only rebuild the class checkboxes if the classes themselves changed,
        # so a reload does not silently re-tick boxes you unticked.
        if "label" in columns:
            current_labels = set(np.unique(columns["label"].astype(np.int32)).tolist())
        else:
            current_labels = set()

        if current_labels != previous_labels:
            self._rebuild_label_filters()

        # Camera, colour mode, point size and range filters are all left as-is.
        self.refresh()
        self.live_status.setText(
            f"updated · {columns['xyz'].shape[0]:,} points"
        )

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
        try:
            self.last_mtime = os.path.getmtime(path)
        except OSError:
            self.last_mtime = 0.0
        self.reload_failures = 0
        self.metres_check.blockSignals(True)
        self.metres_check.setChecked(False)
        self.metres_check.blockSignals(False)

        self.setWindowTitle(f"LiDAR Point Cloud Viewer — {path.name}")

        self._rebuild_label_filters()
        self._configure_ranges()
        self._set_controls_enabled(True)

        self.refresh(refit=True)
        self.apply_camera()

    def _on_units(self, metres: bool) -> None:
        if self.columns is None:
            return

        scale = 0.01 if metres else 100.0
        self.columns["xyz"] = self.columns["xyz"] * scale
        self.columns["ranges"] = self.columns["ranges"] * scale
        self.columns["origin"] = self.columns["origin"] * scale
        self.metres = metres

        self._configure_ranges()
        self.refresh(refit=True)
        self.apply_camera()

    def _rebuild_label_filters(self) -> None:
        while self.label_box.count():
            item = self.label_box.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.label_checks.clear()

        if self.columns is None or "label" not in self.columns:
            self.label_box.addWidget(QLabel("  (no label channel)"))
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

        self.geometry.points = o3d.utility.Vector3dVector(self.columns["xyz"][indices])
        if len(indices):
            self.geometry.colors = o3d.utility.Vector3dVector(
                make_colours(self.columns, indices, mode)
            )

        if not self.geometry_added:
            self.vis.add_geometry(self.geometry, reset_bounding_box=True)
            self.geometry_added = True
        elif refit:
            # Refit only for a new file or a unit change. Colour and filter
            # changes update in place so the camera is left alone.
            self.vis.clear_geometries()
            self.vis.add_geometry(self.geometry, reset_bounding_box=True)
        else:
            self.vis.update_geometry(self.geometry)

        self.statusBar().showMessage(
            f"{len(indices):,} of {self.columns['xyz'].shape[0]:,} points shown"
        )
        self._update_stats(len(indices))

    def _on_point_size(self, value: int) -> None:
        self.vis.get_render_option().point_size = float(value)

    def _on_zoom(self, value: int) -> None:
        self.zoom = value / 100.0
        self.vis.get_view_control().set_zoom(self.zoom)

    def set_view(self, name: str) -> None:
        self.view_name = name
        self.apply_camera()

    def apply_camera(self) -> None:
        if self.columns is None:
            return

        front, up = VIEWS[self.view_name]
        control = self.vis.get_view_control()
        control.set_lookat([0.0, 0.0, 0.0])   # the cloud is centred there
        control.set_front(front)
        control.set_up(up)
        control.set_zoom(self.zoom)

    # ---- stats ----

    def _update_stats(self, shown: int) -> None:
        if self.columns is None:
            return

        ranges = self.columns["ranges"]
        units = "m" if self.metres else "cm"

        lines = [
            f"file    {self.source_path.name if self.source_path else '-'}",
            f"units   {units}",
            f"points  {self.columns['xyz'].shape[0]:,}",
            f"shown   {shown:,}",
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
                lines.append(f"  {name:<12} {count:,}")

        self.stats_label.setText("\n".join(lines))

    # ---- plumbing ----

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt naming
        self.timer.stop()
        self.watch_timer.stop()
        try:
            self.vis.destroy_window()
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
    app.setStyleSheet(
        "QWidget { background-color: #000000; color: #d8d8dc; }"
        "QGroupBox { border: 1px solid #2a2a30; border-radius: 4px;"
        "            margin-top: 8px; padding-top: 8px; }"
        "QGroupBox::title { subcontrol-origin: margin; left: 8px;"
        "                   padding: 0 4px; color: #9aa0a8; }"
        "QPushButton { background-color: #16161a; border: 1px solid #2f2f36;"
        "              border-radius: 4px; padding: 5px; }"
        "QPushButton:hover { background-color: #22222a; }"
        "QComboBox, QDoubleSpinBox { background-color: #16161a;"
        "              border: 1px solid #2f2f36; border-radius: 3px; padding: 3px; }"
        "QStatusBar { color: #9aa0a8; }"
    )

    viewer = Viewer()
    viewer.show()

    target = Path(sys.argv[1]) if len(sys.argv) > 1 else resolve_watch_path()
    viewer.source_path = target
    viewer.live_status.setText(f"watching {target.name}")

    if target.exists():
        viewer.load(target)
    else:
        # Not an error: the watcher will pick the file up as soon as it appears.
        viewer.statusBar().showMessage(f"Waiting for {target}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()