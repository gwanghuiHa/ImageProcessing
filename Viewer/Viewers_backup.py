"""
Image viewers

@author: Gwanghui

1. launch_image_viewer(images)
    images: numpy array with the dimension of (N-shots, X, Y)
2. launch_ict_viewer(ict_dict, sec_per_div=200e-9)
    Example:
        ict_dict = {
            'Ch1_wfm': dat[0]['Ch1_wfm'],
            'Ch2_wfm': dat[0]['Ch2_wfm'],
        }
        launch_ict_viewer(ict_dict, sec_per_div=200e-9)
"""
#%%
import sys
import numpy as np

from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QSpinBox, QLabel, QSlider, QCheckBox
)

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
#%%
class ImageViewer(QMainWindow):
    def __init__(self, images, parent=None):
        super().__init__(parent)

        self.images = np.asarray(images)
        if self.images.ndim != 3:
            raise ValueError("images must have shape (n_shot, ny, nx)")

        self.n_shot, self.ny, self.nx = self.images.shape

        # basic stats for color range
        self.data_min = float(np.nanmin(self.images))
        self.data_max = float(np.nanmax(self.images))
        if self.data_max == self.data_min:
            self.data_max = self.data_min + 1.0

        self.setWindowTitle("Shot Image Viewer")
        self._init_ui()

    # --------------------- UI SETUP ---------------------
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        # --- top controls: mode selection ---
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Single shot", "Grid (all shots)", "Overlay first N"])
        mode_layout.addWidget(mode_label)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        main_layout.addLayout(mode_layout)

        # --- single shot controls ---
        self.single_ctrl_layout = QHBoxLayout()

        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")

        self.shot_spin = QSpinBox()
        self.shot_spin.setRange(0, self.n_shot - 1)
        self.shot_spin.setValue(0)
        self.shot_spin.setPrefix("Shot ")

        self.single_ctrl_layout.addWidget(self.btn_prev)
        self.single_ctrl_layout.addWidget(self.btn_next)
        self.single_ctrl_layout.addWidget(self.shot_spin)
        self.single_ctrl_layout.addStretch()

        main_layout.addLayout(self.single_ctrl_layout)

        # --- overlay controls ---
        self.overlay_ctrl_layout = QHBoxLayout()
        self.overlay_label = QLabel("N overlay:")
        self.overlay_spin = QSpinBox()
        self.overlay_spin.setRange(1, self.n_shot)
        self.overlay_spin.setValue(min(5, self.n_shot))

        self.overlay_ctrl_layout.addWidget(self.overlay_label)
        self.overlay_ctrl_layout.addWidget(self.overlay_spin)
        self.overlay_ctrl_layout.addStretch()

        main_layout.addLayout(self.overlay_ctrl_layout)

        # --- color range sliders ---
        color_layout = QHBoxLayout()

        self.label_vmin = QLabel("vmin")
        self.slider_vmin = QSlider(QtCore.Qt.Horizontal)
        self.slider_vmin.setRange(0, 1000)
        self.slider_vmin.setValue(0)

        self.label_vmax = QLabel("vmax")
        self.slider_vmax = QSlider(QtCore.Qt.Horizontal)
        self.slider_vmax.setRange(0, 1000)
        self.slider_vmax.setValue(1000)

        color_layout.addWidget(self.label_vmin)
        color_layout.addWidget(self.slider_vmin)
        color_layout.addWidget(self.label_vmax)
        color_layout.addWidget(self.slider_vmax)

        main_layout.addLayout(color_layout)

        # --- matplotlib canvas ---
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        # connect signals
        self.mode_combo.currentIndexChanged.connect(self.update_mode_visibility)
        self.btn_prev.clicked.connect(self.on_prev)
        self.btn_next.clicked.connect(self.on_next)
        self.shot_spin.valueChanged.connect(self.redraw)
        self.overlay_spin.valueChanged.connect(self.redraw)
        self.slider_vmin.valueChanged.connect(self.redraw)
        self.slider_vmax.valueChanged.connect(self.redraw)

        # set initial visibility and draw
        self.update_mode_visibility()
        self.redraw()

    # --------------------- HELPERS ---------------------
    def get_vmin_vmax(self):
        # map slider [0, 1000] -> [data_min, data_max]
        smin = self.slider_vmin.value()
        smax = self.slider_vmax.value()

        # ensure smin <= smax
        if smin > smax:
            smin, smax = smax, smin

        frac_min = smin / 1000.0
        frac_max = smax / 1000.0

        vmin = self.data_min + frac_min * (self.data_max - self.data_min)
        vmax = self.data_min + frac_max * (self.data_max - self.data_min)

        if vmin == vmax:
            vmax = vmin + 1e-12

        return vmin, vmax

    def update_mode_visibility(self):
        mode = self.mode_combo.currentText()

        # show single shot controls only in that mode
        single_visible = (mode == "Single shot")
        for i in range(self.single_ctrl_layout.count()):
            item = self.single_ctrl_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setVisible(single_visible)

        # show overlay controls only in overlay mode
        overlay_visible = (mode == "Overlay first N")
        for i in range(self.overlay_ctrl_layout.count()):
            item = self.overlay_ctrl_layout.itemAt(i)
            widget = item.widget()
            if widget is not None:
                widget.setVisible(overlay_visible)

        self.redraw()

    # --------------------- CALLBACKS ---------------------
    def on_prev(self):
        val = self.shot_spin.value()
        if val > 0:
            self.shot_spin.setValue(val - 1)

    def on_next(self):
        val = self.shot_spin.value()
        if val < self.n_shot - 1:
            self.shot_spin.setValue(val + 1)

    # --------------------- DRAWING ----------------------
    def redraw(self):
        mode = self.mode_combo.currentText()
        vmin, vmax = self.get_vmin_vmax()

        self.fig.clf()

        if mode == "Single shot":
            ax = self.fig.add_subplot(111)
            idx = self.shot_spin.value()
            img = self.images[idx]
            im = ax.imshow(img, vmin=vmin, vmax=vmax,
                           origin="lower", aspect="auto")
            ax.set_title(f"Shot {idx}")
            self.fig.colorbar(im, ax=ax)

        elif mode == "Grid (all shots)":
            ncols = min(4, self.n_shot)
            nrows = int(np.ceil(self.n_shot / ncols))
            axes = []
            for i in range(self.n_shot):
                ax = self.fig.add_subplot(nrows, ncols, i + 1)
                axes.append(ax)
                img = self.images[i]
                im = ax.imshow(img, vmin=vmin, vmax=vmax,
                               origin="lower", aspect="auto")
                ax.set_title(f"{i}")
                ax.axis("off")

            # share one colorbar
            if axes:
                self.fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)

        elif mode == "Overlay first N":
            ax = self.fig.add_subplot(111)
            N = self.overlay_spin.value()
            N = max(1, min(N, self.n_shot))
            img_overlay = np.mean(self.images[:N], axis=0)  # change to sum/max if you like

            im = ax.imshow(img_overlay, vmin=vmin, vmax=vmax,
                           origin="lower", aspect="auto")
            ax.set_title(f"Overlay of first {N} shots (mean)")
            self.fig.colorbar(im, ax=ax)

        self.fig.tight_layout()
        self.canvas.draw_idle()


def launch_image_viewer(images):
    """
    Convenience function to launch viewer from Spyder.

    Example:
        import image_viewer as iv
        iv.launch_image_viewer(images)
    """
    app = QtWidgets.QApplication.instance()
    app_created = False

    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        app_created = True

    viewer = ImageViewer(images)
    viewer.show()

    if app_created:
        app.exec_()

    return viewer
#%%
class ICTViewer(QMainWindow):
    """
    ICT waveform viewer with scale + pan controls.

    Parameters
    ----------
    ict_dict : dict
        Keys: channel names (e.g. 'Ch1_wfm', 'Ch2_wfm', ...)
        Values: for each key, either
            - list of 1D numpy arrays, one per shot, OR
            - numpy array of shape (n_shot, n_samples)
    sec_per_div : float
        Horizontal scale in seconds/div (e.g. 200e-9 for 200 ns/div).
        Total time window = 10 * sec_per_div.
    """
    def __init__(self, ict_dict, sec_per_div=200e-9, parent=None):
        super().__init__(parent)

        if not ict_dict:
            raise ValueError("ict_dict is empty")

        # normalize channels -> arrays of shape (n_shot, n_samples)
        self.channels = {}
        first_key = list(ict_dict.keys())[0]

        def normalize_ch(data):
            if isinstance(data, list):
                return np.stack(data, axis=0)
            arr = np.asarray(data)
            if arr.ndim == 1:
                return arr[None, :]
            elif arr.ndim == 2:
                return arr
            else:
                raise ValueError("Channel data must be list of 1D arrays or 2D array")

        ref = normalize_ch(ict_dict[first_key])
        self.n_shot, self.n_samples = ref.shape
        self.channels[first_key] = ref

        for k, v in ict_dict.items():
            if k == first_key:
                continue
            arr = normalize_ch(v)
            if arr.shape != ref.shape:
                raise ValueError(f"Channel {k} shape {arr.shape} "
                                 f"does not match reference {ref.shape}")
            self.channels[k] = arr

        self.ch_names = list(self.channels.keys())

        # base horizontal scale (like your old ictBase)
        self.sec_per_div_base = sec_per_div             # e.g. 200 ns/div
        self.total_time = 10.0 * self.sec_per_div_base  # 10 divisions total
        dt = self.total_time / self.n_samples
        self.t = np.arange(self.n_samples) * dt         # seconds
        self.t_us = self.t * 1e6                        # for plotting in µs
        self.total_time_us = self.total_time * 1e6

        # global amplitude range for ylim scaling
        all_data = np.concatenate(
            [ch.reshape(-1) for ch in self.channels.values()]
        )
        self.y_abs_max = float(np.nanmax(np.abs(all_data)))
        if self.y_abs_max == 0:
            self.y_abs_max = 1.0

        self.setWindowTitle("ICT Waveform Viewer")

        self._init_ui()

    # --------------------- UI SETUP ---------------------
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout()
        central.setLayout(main_layout)

        # --- shot selection + overlay toggle ---
        shot_layout = QHBoxLayout()
        shot_label = QLabel("Shot:")
        self.shot_spin = QSpinBox()
        self.shot_spin.setRange(0, self.n_shot - 1)
        self.shot_spin.setValue(0)
        self.shot_spin.setPrefix(" ")

        self.overlay_shots_cb = QCheckBox("Overlay all shots")
        self.overlay_shots_cb.setChecked(False)

        shot_layout.addWidget(shot_label)
        shot_layout.addWidget(self.shot_spin)
        shot_layout.addWidget(self.overlay_shots_cb)
        shot_layout.addStretch()
        main_layout.addLayout(shot_layout)

        # --- channel checkboxes ---
        ch_layout = QHBoxLayout()
        ch_layout.addWidget(QLabel("Channels:"))
        self.ch_checkboxes = []
        for name in self.ch_names:
            cb = QCheckBox(name)
            cb.setChecked(True)
            cb.stateChanged.connect(self.redraw)
            self.ch_checkboxes.append(cb)
            ch_layout.addWidget(cb)
        ch_layout.addStretch()
        main_layout.addLayout(ch_layout)

        # --- X scale (width) ---
        xw_layout = QHBoxLayout()
        self.label_xw = QLabel("X width (divisions):")
        self.slider_x_width = QSlider(QtCore.Qt.Horizontal)
        self.slider_x_width.setRange(1, 10)      # 1 to 10 divisions visible
        self.slider_x_width.setValue(10)         # start with full window

        xw_layout.addWidget(self.label_xw)
        xw_layout.addWidget(self.slider_x_width)
        main_layout.addLayout(xw_layout)

        # --- X pan ---
        xp_layout = QHBoxLayout()
        self.label_xp = QLabel("X pan:")
        self.slider_x_pan = QSlider(QtCore.Qt.Horizontal)
        self.slider_x_pan.setRange(0, 1000)      # 0..1 fraction over allowed pan range
        self.slider_x_pan.setValue(0)            # start at left

        xp_layout.addWidget(self.label_xp)
        xp_layout.addWidget(self.slider_x_pan)
        main_layout.addLayout(xp_layout)

        # --- Y scale (half-height) ---
        ys_layout = QHBoxLayout()
        self.label_ys = QLabel("Y scale (×global max):")
        self.slider_y_scale = QSlider(QtCore.Qt.Horizontal)
        self.slider_y_scale.setRange(10, 500)    # 0.1x to 5x
        self.slider_y_scale.setValue(100)        # 1x

        ys_layout.addWidget(self.label_ys)
        ys_layout.addWidget(self.slider_y_scale)
        main_layout.addLayout(ys_layout)

        # --- Y pan (center) ---
        yp_layout = QHBoxLayout()
        self.label_yp = QLabel("Y pan:")
        self.slider_y_pan = QSlider(QtCore.Qt.Horizontal)
        self.slider_y_pan.setRange(0, 1000)      # maps to -y_abs_max..+y_abs_max
        self.slider_y_pan.setValue(500)          # center at 0

        yp_layout.addWidget(self.label_yp)
        yp_layout.addWidget(self.slider_y_pan)
        main_layout.addLayout(yp_layout)

        # --- matplotlib canvas ---
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        main_layout.addWidget(self.canvas)

        # connect signals
        self.shot_spin.valueChanged.connect(self.redraw)
        self.slider_x_width.valueChanged.connect(self.redraw)
        self.slider_x_pan.valueChanged.connect(self.redraw)
        self.slider_y_scale.valueChanged.connect(self.redraw)
        self.slider_y_pan.valueChanged.connect(self.redraw)
        self.overlay_shots_cb.stateChanged.connect(self.redraw)

        # initial draw
        self.redraw()

    # --------------------- DRAWING ----------------------
    def redraw(self):
        visible_div = self.slider_x_width.value()          # number of divisions visible (1..10)
        time_window = visible_div * self.sec_per_div_base  # seconds
        win_us = time_window * 1e6                         # µs

        # X pan:
        # allow window center to move from win_us/2 .. total_time_us - win_us/2
        # if window >= total range, clamp to full range
        if win_us >= self.total_time_us:
            x_min_us = 0.0
            x_max_us = self.total_time_us
        else:
            frac = self.slider_x_pan.value() / 1000.0  # 0..1
            x_center_min = win_us / 2.0
            x_center_max = self.total_time_us - win_us / 2.0
            x_center = x_center_min + frac * (x_center_max - x_center_min)
            x_min_us = x_center - win_us / 2.0
            x_max_us = x_center + win_us / 2.0

        # Y scale & pan:
        amp_scale = self.slider_y_scale.value() / 100.0  # 0.1..5.0
        y_half = amp_scale * self.y_abs_max

        frac_y = self.slider_y_pan.value() / 1000.0      # 0..1
        # map 0..1 → -y_abs_max..+y_abs_max
        y_center = (2.0 * frac_y - 1.0) * self.y_abs_max
        y_min = y_center - y_half
        y_max = y_center + y_half

        overlay_all = self.overlay_shots_cb.isChecked()
        shot_idx = self.shot_spin.value()

        self.fig.clf()
        ax = self.fig.add_subplot(111)

        for cb, name in zip(self.ch_checkboxes, self.ch_names):
            if not cb.isChecked():
                continue

            ch_data = self.channels[name]

            if overlay_all:
                for i in range(self.n_shot):
                    wf = ch_data[i]
                    ax.plot(self.t_us, wf, alpha=0.3)
            else:
                wf = ch_data[shot_idx]
                ax.plot(self.t_us, wf, label=name)

        ax.set_xlabel("Time (µs)")
        ax.set_ylabel("Signal")
        title = "ICT Waveforms"
        if overlay_all:
            title += " - All shots"
        else:
            title += f" - Shot {shot_idx}"
        ax.set_title(title)

        ax.grid(True, alpha=0.3)

        ax.set_xlim(x_min_us, x_max_us)
        ax.set_ylim(y_min, y_max)

        if not overlay_all:
            ax.legend(loc="best")

        self.fig.tight_layout()
        self.canvas.draw_idle()


def launch_ict_viewer(ict_dict, sec_per_div=200e-9):
    """
    Convenience function to launch the ICT viewer.

    Example:
        ict_dict = {
            'Ch1_wfm': dat[0]['Ch1_wfm'],
            'Ch2_wfm': dat[0]['Ch2_wfm'],
        }
        launch_ict_viewer(ict_dict, sec_per_div=200e-9)
    """
    app = QtWidgets.QApplication.instance()
    app_created = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
        app_created = True

    viewer = ICTViewer(ict_dict, sec_per_div=sec_per_div)
    viewer.show()

    if app_created:
        app.exec_()

    return viewer
