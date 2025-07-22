import contextlib
import logging
import os
import numpy as np
import vispy
import sys

from PyQt5 import QtGui, QtCore, QtWidgets
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from util import units, config, qt_theme, colormaps, io_ops, snd
from util.config import save_json, load_json
from util.markers import Cursor
from util.qt_threads import CursorUpdater
from util.undo import UndoStack, AddAction
from util.units import pitch

ICON_CACHE = {"no_icon": QtGui.QIcon()}


class ConfigStorer:
    def to_cfg(self, cfg):
        if self.isVisible():
            for varname in self.vars_for_saving:
                cfg[varname] = getattr(self, varname)

    def from_cfg(self, cfg):
        if self.isVisible():
            # logging.info(f"Loading {self.__class__.__name__}")
            for varname in self.vars_for_saving:
                try:
                    setattr(self, varname, cfg[varname])
                except:
                    logging.warning(f"Could not set {varname}")


def get_icon(name):
    if name in ICON_CACHE:
        return ICON_CACHE[name]
    for ext in (".png", ".svg"):
        root_dir = os.path.dirname(os.path.dirname(__file__))
        fp = os.path.join(root_dir, f'icons/{name}{ext}')
        if os.path.isfile(fp):
            ICON_CACHE[name] = QtGui.QIcon(fp)
            return ICON_CACHE[name]
    return ICON_CACHE["no_icon"]


def print_version_info():
    print("Running...")
    print(f"Python: {sys.version}")
    print(f"Numpy: {np.__version__}")
    print(f"Vispy: {vispy.__version__}")


def startup(cls):
    print_version_info()
    appQt = QtWidgets.QApplication([])

    # style
    appQt.setStyle(QtWidgets.QStyleFactory.create('Fusion'))
    appQt.setPalette(qt_theme.dark_palette)
    appQt.setStyleSheet("QToolTip { color: #ffffff; background-color: #353535; border: 1px solid white; }")

    win = cls()
    # only support audio playback if a valid cursor is available
    if isinstance(win.canvas.cursor, Cursor):
        cursor_thread = QtCore.QThread(parent=win)
        cursor_updater = CursorUpdater()
        cursor_updater.moveToThread(cursor_thread)
        cursor_updater.new_pos.connect(win.canvas.cursor.set_time)
        cursor_updater.new_pos.connect(win.canvas.scroll_view)
        cursor_thread.started.connect(cursor_updater.update_cursor)
        # if the cursor updater finishes before the window is closed, kill the thread
        cursor_updater.finished.connect(cursor_thread.quit, QtCore.Qt.DirectConnection)
        win.props.audio_widget.cursor_set.connect(cursor_updater.update_time, QtCore.Qt.DirectConnection)
        win.props.audio_widget.is_playing.connect(cursor_updater.update_playing, QtCore.Qt.DirectConnection)
        # if the window is closed, tell the cursor updater to stop
        win.closing.connect(cursor_updater.stop_data, QtCore.Qt.DirectConnection)
        # when the thread has ended, delete the cursor updater from memory
        cursor_thread.finished.connect(cursor_updater.deleteLater)
        cursor_thread.start()
    
    win.show()
    appQt.exec_()
    config.save_config(win.cfg)


def showdialog(msg_txt):
    msg = QtWidgets.QMessageBox()
    msg.setIcon(QtWidgets.QMessageBox.Information)
    msg.setText(msg_txt)
    # msg.setInformativeText("This is additional information")
    msg.setWindowTitle("Error")
    # msg.setDetailedText("The details are as follows:")
    msg.setStandardButtons(QtWidgets.QMessageBox.Ok)
    msg.exec_()


def grid(buttons):
    qgrid = QtWidgets.QGridLayout()
    qgrid.setHorizontalSpacing(3)
    qgrid.setVerticalSpacing(0)
    for i, line in enumerate(buttons):
        for j, element in enumerate(line):
            # we want to stretch that one
            if 1 == len(line):
                qgrid.addWidget(line[j], i, j, 1, 2)
            else:
                qgrid.addWidget(line[j], i, j)
    for i in range(2):
        qgrid.setColumnStretch(i, 1)
    return qgrid


def vbox(parent, grid_layout):
    box = QtWidgets.QVBoxLayout(parent)
    box.addLayout(grid_layout)
    box.addStretch(1)
    box.setSpacing(3)


# box.setContentsMargins(0, 0, 0, 0)


def vbox2(parent, buttons):
    box = QtWidgets.QVBoxLayout(parent)
    for w in buttons:
        box.addWidget(w)
    box.addStretch(1)


class ChannelWidget(QtWidgets.QScrollArea):
    selected_channels = QtCore.pyqtSignal(bool)

    def __init__(self, ):
        super().__init__()
        self.setToolTip("Select channels for processing.")
        self.setFixedHeight(50)

        self.channel_holder = QtWidgets.QWidget()
        self.setWidget(self.channel_holder)

        self.channel_layout = QtWidgets.QVBoxLayout()
        self.channel_layout.setSpacing(0)
        # self.channel_layout.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        # self.channel_layout.addStretch(1)
        self.channel_layout.setContentsMargins(2, 2, 2, 2)

        self.channel_holder.setLayout(self.channel_layout)
        self.channel_checkboxes = []
        self.setWidgetResizable(True)

    def refill(self, num_channels):
        for channel in self.channel_checkboxes:
            self.channel_layout.removeWidget(channel)
            channel.deleteLater()
        self.channel_checkboxes = []
        # fill the channel UI
        channel_names = ("Front Left", "Front Right", "Center", "LFE", "Back Left", "Back Right")
        for i in range(0, num_channels):
            name = channel_names[i] if i < 6 else str(i)
            checkbox = QtWidgets.QCheckBox(name)
            checkbox.clicked.connect(self.channel_checked)
            self.channel_checkboxes.append(checkbox)
            # set the startup option to just resample channel 0
            # self.channel_checkboxes[-1].setChecked(True if i == 0 else False)
            self.channel_checkboxes[-1].setChecked(True)
            self.channel_layout.addWidget(self.channel_checkboxes[-1])

    @property
    def channels(self, ):
        return [i for i, channel in enumerate(self.channel_checkboxes) if channel.isChecked()]

    def channel_checked(self, v):
        """Emits the indices of all selected channels"""
        if self.channels:
            self.selected_channels.emit(True)


class FileWidget(QtWidgets.QWidget):
    """An entry widget that starts a file selector when clicked and also accepts drag & drop.
    Displays the current file's basename.
    """

    file_changed = QtCore.pyqtSignal(str)
    spectrum_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent, cfg, description="", ask_user=True, cfg_key="dir_in"):
        super(FileWidget, self).__init__(parent)
        self.parent = parent
        self.cfg = cfg
        self.cfg_key = cfg_key
        if not self.cfg:
            self.cfg[self.cfg_key] = "C://"

        self.filepath = ""
        self.description = description
        self.setToolTip(self.description)
        self.ask_user = ask_user

        self.entry = QtWidgets.QLineEdit()
        self.icon = QtWidgets.QPushButton()
        self.sr_label = QtWidgets.QLabel()
        self.sr_label.setAlignment(QtCore.Qt.AlignHCenter)
        self.channel_widget = ChannelWidget()
        self.channel_widget.selected_channels.connect(self.update_selected_channel)

        self.icon.setIcon(get_icon("dir"))
        self.icon.setFlat(True)

        self.setAcceptDrops(True)
        self.entry.setReadOnly(True)

        self.icon.clicked.connect(self.ask_open)

        self.filename = ""

        self.qgrid = QtWidgets.QGridLayout()
        self.qgrid.setContentsMargins(0, 0, 0, 0)
        self.qgrid.addWidget(self.icon, 0, 0)
        self.qgrid.addWidget(self.entry, 0, 1)
        self.qgrid.addWidget(self.sr_label, 1, 0)
        self.qgrid.addWidget(self.channel_widget, 1, 1)

        self.setLayout(self.qgrid)

        self.signal = self.sr = self.num_channels = None
        self.spectra = []

    def set_sr(self, sr):
        self.sr_label.setText(f"{sr // 1000}k")

    def abort_open_new_file(self, new_filepath):
        # only return True if we should abort
        if not self.ask_user:
            return False
        if new_filepath == self.filepath:
            return True
        if self.filepath:
            qm = QtWidgets.QMessageBox
            return qm.No == qm.question(
                self, '', f"Do you really want to load {os.path.basename(new_filepath)}? "
                          f"You will lose unsaved work on {os.path.basename(self.filepath)}!", qm.Yes | qm.No)

    def ignoreEvent(self, event):
        event.ignore()

    def accept_file(self, filepath):
        if os.path.isfile(filepath):
            if os.path.splitext(filepath)[1].lower() in (".flac", ".wav"):
                if not self.abort_open_new_file(filepath):
                    self.filepath = filepath
                    self.cfg[self.cfg_key], self.filename = os.path.split(filepath)
                    self.entry.setText(self.filename)
                    self.signal, self.sr, self.num_channels = io_ops.read_file(filepath)
                    self.channel_widget.refill(self.num_channels)
                    for spectrum in self.spectra:
                        spectrum.signal = self.signal
                        spectrum.sr = self.sr
                        spectrum.change_file(filepath)
                    self.copy_channels()
                    self.set_sr(self.sr)
                    self.file_changed.emit(filepath)
                    # todo
                    try:
                        self.parent.parent.parent.canvas.reset_view()
                    except:
                        logging.warning(f"refactor self.parent.parent.parent.canvas.reset_view() to signal")
                    self.parent.poll()
            else:
                showdialog("Unsupported File Format")

    def get_files(self, event):
        data = event.mimeData()
        urls = data.urls()
        if urls and urls[0].scheme() == 'file':
            return urls

    def dragEnterEvent(self, event):
        if self.get_files(event):
            event.acceptProposedAction()
            self.setFocus(True)

    def dragMoveEvent(self, event):
        if self.get_files(event):
            event.acceptProposedAction()
            self.setFocus(True)

    def dropEvent(self, event):
        urls = self.get_files(event)
        if urls:
            filepath = str(urls[0].path())[1:]
            self.accept_file(filepath)

    def ask_open(self):
        filepath = QtWidgets.QFileDialog.getOpenFileName(self, 'Open ' + self.description, self.cfg.get(self.cfg_key, "C:/"),
                                                         "Audio files (*.flac *.wav)")[0]
        self.accept_file(filepath)
        # update channels & recalculate spectrum
        self.update_selected_channel()

    def copy_channels(self):
        for spectrum, channel in zip(self.spectra, self.channel_widget.channels):
            assert spectrum
            # get the first selected channel
            spectrum.selected_channel = channel

    def update_selected_channel(self, stuff=False):
        self.copy_channels()
        # recalculate spectrum
        self.spectrum_changed.emit("")
        # todo
        try:
            self.parent.parent.display_widget.update_fft_settings()
        except:
            logging.warning(f"refactor self.parent.parent.display_widget.update_fft_settings() to signal")


class SpectrumSettingsWidget(QtWidgets.QGroupBox, ConfigStorer):
    vars_for_saving = ("fft_size", "fft_overlap", "fft_zeropad")

    def __init__(self, with_canvas=True):
        super().__init__("Spectrum")

        fft_l = QtWidgets.QLabel("FFT Size")
        self.fft_c = QtWidgets.QComboBox(self)
        self.fft_c.addItems(
            ("64", "128", "256", "512", "1024", "2048", "4096", "8192", "16384", "32768", "65536", "131072", "262144",
             "524288", "1048576"))
        self.fft_c.setToolTip("This determines the frequency resolution.")
        self.fft_c.setCurrentIndex(5)

        self.clear_storage = QtWidgets.QPushButton("Clear Storage")

        overlap_l = QtWidgets.QLabel("FFT Overlap")
        self.overlap_c = QtWidgets.QComboBox(self)
        self.overlap_c.addItems(("1", "2", "4", "8", "16", "32"))
        self.overlap_c.setToolTip("Increase to improve temporal resolution.")
        self.overlap_c.setCurrentIndex(2)

        zeropad_l = QtWidgets.QLabel("FFT Zero Padding")
        self.zeropad_c = QtWidgets.QComboBox(self)
        self.zeropad_c.addItems(("1", "2", "4", "8", "16"))
        self.zeropad_c.setToolTip("Increase to improve frequency resolution without temporal smearing.")
        self.zeropad_c.setCurrentIndex(2)

        cmap_l = QtWidgets.QLabel("Colors")
        self.cmap_c = QtWidgets.QComboBox(self)
        self.cmap_c.addItems(sorted(colormaps.cmaps.keys()))
        self.cmap_c.setCurrentText("izo")
        cmap_l.setVisible(with_canvas)
        self.cmap_c.setVisible(with_canvas)

        buttons = [(fft_l, self.fft_c), (overlap_l, self.overlap_c), (zeropad_l, self.zeropad_c), (cmap_l, self.cmap_c), (self.clear_storage,)]
        vbox(self, grid(buttons))

        if with_canvas:
            # these should only be updated by the user; programmatic updates should call update_fft_settings
            self.fft_c.activated.connect(self.update_fft_settings)
            self.overlap_c.activated.connect(self.update_fft_settings)
            self.zeropad_c.activated.connect(self.update_fft_settings)
            # these can be updated each time
            self.cmap_c.currentIndexChanged.connect(self.update_cmap)
            self.clear_storage.clicked.connect(self.force_clear_storage)

    @property
    def fft_size(self):
        return int(self.fft_c.currentText())
    
    @fft_size.setter
    def fft_size(self, v):
        self.fft_c.setCurrentText(str(v))

    @property
    def fft_overlap(self):
        return int(self.overlap_c.currentText())

    @fft_overlap.setter
    def fft_overlap(self, v):
        self.overlap_c.setCurrentText(str(v))

    @property
    def fft_zeropad(self):
        return int(self.zeropad_c.currentText())

    @fft_zeropad.setter
    def fft_zeropad(self, v):
        self.zeropad_c.setCurrentText(str(v))

    def update_fft_settings(self):
        self.canvas.fft_size = self.fft_size
        self.canvas.hop = self.fft_size // self.fft_overlap
        self.canvas.zeropad = self.fft_zeropad
        self.canvas.compute_spectra()

    def update_cmap(self):
        self.canvas.set_colormap(self.cmap_c.currentText())

    def force_clear_storage(self):
        self.canvas.clear_fft_storage()


class TracingWidget(QtWidgets.QGroupBox, ConfigStorer):
    vars_for_saving = ("mode", "tolerance")

    def __init__(self, ):
        super().__init__("Tracing")
        trace_l = QtWidgets.QLabel("Mode")
        self.trace_c = QtWidgets.QComboBox(self)
        self.trace_c.addItems(
            (
                "Center of Gravity",
                "Peak",
                "Peak Track",
                "Zero-Crossing",
                "Partials",
                "Correlation",
                "Freehand Draw",
                "Sine Regression"
            ))
        self.trace_c.currentIndexChanged.connect(self.toggle_trace_mode)

        self.rpm_l = QtWidgets.QLabel("Source RPM")
        self.rpm_c = QtWidgets.QComboBox(self)
        self.rpm_c.setEditable(True)
        self.rpm_c.addItems(("Unknown", "33.333", "45", "78"))
        self.rpm_c.setToolTip(
            "This helps avoid bad values in the sine regression. \n"
            "If you don't know the source, measure the duration of one wow cycle. \nRPM = 60/cycle length")

        self.phase_l = QtWidgets.QLabel("Phase Offset")
        self.phase_s = QtWidgets.QSpinBox()
        self.phase_s.setRange(-20, 20)
        self.phase_s.setSingleStep(1)
        self.phase_s.setValue(0)
        self.phase_s.valueChanged.connect(self.update_phase_offset)
        self.phase_s.setToolTip("Adjust the phase of the selected sine regression to match the surrounding regions.")

        tolerance_l = QtWidgets.QLabel("Tolerance")
        self.tolerance_s = QtWidgets.QDoubleSpinBox()
        self.tolerance_s.setRange(.01, 5)
        self.tolerance_s.setSingleStep(.05)
        self.tolerance_s.setValue(.5)
        self.tolerance_s.setSuffix(" st")
        self.tolerance_s.setToolTip("Intervall to consider in the trace, in semitones.")

        adapt_l = QtWidgets.QLabel("Adaptation")
        self.adapt_c = QtWidgets.QComboBox(self)
        self.adapt_c.addItems(("Average", "Linear", "Constant", "None"))
        self.adapt_c.setToolTip("Used to predict the next frequencies when tracing.")
        # might as well hide it until it is re-implemented
        adapt_l.setVisible(False)
        self.adapt_c.setVisible(False)

        target_l = QtWidgets.QLabel("Target Frequency")
        self.target_s = QtWidgets.QDoubleSpinBox()
        self.target_s.setRange(0, 30000)
        self.target_s.setSingleStep(.1)
        self.target_s.setValue(0)
        self.target_s.setSuffix(" Hz")
        self.target_s.setToolTip("The selected traces' mean frequency.")

        self.target_b = QtWidgets.QPushButton("Set Freq")
        self.target_b.clicked.connect(self.update_target)
        self.target_b.setToolTip("Set mean frequency to selected traces.")

        self.autoalign_b = QtWidgets.QCheckBox("Auto-Align")
        self.autoalign_b.setChecked(True)
        self.autoalign_b.setToolTip("Should new traces be aligned with existing ones?")

        show_l = QtWidgets.QLabel("Show")
        self.show_c = QtWidgets.QComboBox(self)
        self.show_c.addItems(("Both", "Traces", "Regressions"))
        self.show_c.currentIndexChanged.connect(self.update_show_settings)

        buttons = (
            (show_l, self.show_c), (trace_l, self.trace_c), (adapt_l, self.adapt_c), (self.rpm_l, self.rpm_c),
            (self.phase_l, self.phase_s), (tolerance_l, self.tolerance_s), (target_l, self.target_s),
            (self.target_b,), (self.autoalign_b,))
        vbox(self, grid(buttons))

        self.toggle_trace_mode()

    @property
    def mode(self):
        return self.trace_c.currentText()

    @mode.setter
    def mode(self, t):
        self.trace_c.setCurrentText(t)

    @property
    def tolerance(self):
        return self.tolerance_s.value()

    @property
    def adapt(self):
        return self.adapt_c.currentText()

    @property
    def auto_align(self):
        return self.autoalign_b.isChecked()

    @property
    def rpm(self):
        return self.rpm_c.currentText()

    def toggle_trace_mode(self):
        b = (self.trace_c.currentText() == "Sine Regression")
        self.rpm_l.setVisible(b)
        self.rpm_c.setVisible(b)
        self.phase_l.setVisible(b)
        self.phase_s.setVisible(b)

    def update_show_settings(self):
        show = self.show_c.currentText()
        if show == "Traces":
            self.canvas.show_regs = False
            self.canvas.show_lines = True
            self.canvas.master_speed.show()
            for trace in self.canvas.lines:
                trace.show()
            self.canvas.master_reg_speed.hide()
            for reg in self.canvas.regs:
                reg.hide()
        elif show == "Regressions":
            self.canvas.show_regs = True
            self.canvas.show_lines = False
            self.canvas.master_speed.hide()
            for trace in self.canvas.lines:
                trace.hide()
            self.canvas.master_reg_speed.show()
            for reg in self.canvas.regs:
                reg.show()
        elif show == "Both":
            self.canvas.show_regs = True
            self.canvas.show_lines = True
            self.canvas.master_speed.show()
            for trace in self.canvas.lines:
                trace.show()
            self.canvas.master_reg_speed.show()
            for reg in self.canvas.regs:
                reg.show()

    def update_phase_offset(self):
        v = self.phase_s.value()
        for reg in self.canvas.regs:
            reg.update_phase(v)
        self.canvas.master_reg_speed.update()

    def update_target(self):
        f = self.target_s.value()
        for reg in self.canvas.lines:
            reg.lock_to(f)
        self.canvas.master_speed.update()


class FiltersWidget(QtWidgets.QGroupBox, ConfigStorer):
    vars_for_saving = ()

    bands_changed = QtCore.pyqtSignal(tuple)

    def __init__(self, ):
        super().__init__("Filters")
        band0_l = QtWidgets.QLabel("Highpass")
        self.band0_s = QtWidgets.QDoubleSpinBox()
        self.band0_s.setRange(0, 10000)
        self.band0_s.setSingleStep(.1)
        self.band0_s.setValue(0)
        self.band0_s.setToolTip("Cull wow below this frequency from the final speed curve.")
        self.band0_s.valueChanged.connect(self.update_bands)

        band1_l = QtWidgets.QLabel("Lowpass")
        self.band1_s = QtWidgets.QDoubleSpinBox()
        self.band1_s.setRange(.01, 10000)
        self.band1_s.setSingleStep(.1)
        self.band1_s.setValue(20)
        self.band1_s.setToolTip("Cull flutter above this frequency from the final speed curve.")
        self.band1_s.valueChanged.connect(self.update_bands)
        buttons = (
            (band0_l, self.band0_s),
            (band1_l, self.band1_s),
        )
        vbox(self, grid(buttons))

    def update_bands(self):
        self.bands_changed.emit((self.band0_s.value(), self.band1_s.value()))


class AlignmentWidget(QtWidgets.QGroupBox, ConfigStorer):
    vars_for_saving = ("smoothing", "ignore_phase",)

    def __init__(self, ):
        super().__init__("Alignment")
        self.ignore_phase_b = QtWidgets.QCheckBox("Ignore phase")
        self.ignore_phase_b.setChecked(False)
        self.ignore_phase_b.setToolTip(
            "Turn on if phase of sources does not match and you want the strongest relationship.\n"
            "Consistent negative values indicate you should invert one source.")

        corr_l = QtWidgets.QLabel("Correlation")
        self.corr_l = QtWidgets.QLabel("None")

        self.smoothing_l = QtWidgets.QLabel("Smoothing")
        self.smoothing_s = QtWidgets.QSpinBox()
        self.smoothing_s.setRange(1, 5)
        self.smoothing_s.setSingleStep(1)
        self.smoothing_s.setValue(3)
        self.smoothing_s.setToolTip("Degree of the smoothing spline")

        self.win_l = QtWidgets.QLabel("Window")
        self.win_s = QtWidgets.QDoubleSpinBox()
        self.win_s.setRange(0.001, 10)
        self.win_s.setSingleStep(.05)
        self.win_s.setValue(0.2)
        self.win_s.setSuffix(" s")
        self.win_s.setToolTip("Length of window for azimuth comparison")

        self.overlap_l = QtWidgets.QLabel("Overlap")
        self.overlap_s = QtWidgets.QSpinBox()
        self.overlap_s.setRange(1, 100)
        self.overlap_s.setSingleStep(1)
        self.overlap_s.setValue(32)
        self.overlap_s.setToolTip("Amount of overlap between azimuth detection windows")

        self.reject_l = QtWidgets.QLabel("Reject")
        self.reject_s = QtWidgets.QDoubleSpinBox()
        self.reject_s.setRange(0.0, 1.0)
        self.reject_s.setSingleStep(.05)
        self.reject_s.setValue(0.1)
        self.reject_s.setToolTip("Reject alignment value if correlation goes lower than this value")

        buttons = (
            (self.ignore_phase_b,), (corr_l, self.corr_l), (self.smoothing_l, self.smoothing_s),
            (self.win_l, self.win_s), (self.overlap_l, self.overlap_s), (self.reject_l, self.reject_s)
        )
        vbox(self, grid(buttons))

    @property
    def smoothing(self):
        return self.smoothing_s.value()

    @smoothing.setter
    def smoothing(self, v):
        self.smoothing_s.setValue(v)

    @property
    def ignore_phase(self): return self.ignore_phase_b.isChecked()

    @ignore_phase.setter
    def ignore_phase(self, is_checked):
        self.ignore_phase_b.setChecked(is_checked)


class DropsWidget(QtWidgets.QGroupBox, ConfigStorer):
    vars_for_saving = ("before_after", "surrounding",)

    def __init__(self, ):
        super().__init__("Dropouts")
        self.before_after_l = QtWidgets.QLabel("Before / After")
        self.before_after_s = QtWidgets.QDoubleSpinBox()
        self.before_after_s.setRange(-1.0, 1.0)
        self.before_after_s.setSingleStep(0.1)
        self.before_after_s.setValue(0.0)
        self.before_after_s.setToolTip("Where to balance the surrounding window from")

        self.surrounding_l = QtWidgets.QLabel("Surrounding Region")
        self.surrounding_s = QtWidgets.QDoubleSpinBox()
        self.surrounding_s.setRange(0.001, 1)
        self.surrounding_s.setSingleStep(.05)
        self.surrounding_s.setValue(0.5)
        self.surrounding_s.setSuffix(" %")
        self.surrounding_s.setToolTip("Length of window outside of marker used to detect intended signal levels")

        self.gain_l = QtWidgets.QLabel("Gain")
        self.gain_s = QtWidgets.QSpinBox()
        self.gain_s.setRange(-15, 15)
        self.gain_s.setSingleStep(1)
        self.gain_s.setValue(0)
        self.gain_s.setSuffix(" dB")
        self.gain_s.setToolTip("Additional gain applied to selected dropouts")

        self.width_l = QtWidgets.QLabel("Width")
        self.width_s = QtWidgets.QSpinBox()
        self.width_s.setRange(1, 200)
        self.width_s.setSingleStep(1)
        self.width_s.setValue(20)
        self.width_s.setSuffix(" ms")
        self.width_s.setToolTip("Baseline width for automatically detected dropouts")
        
        self.sensitivity_l = QtWidgets.QLabel("Sensitivity")
        self.sensitivity_s = QtWidgets.QSpinBox()
        self.sensitivity_s.setRange(0, 10)
        self.sensitivity_s.setSingleStep(1)
        self.sensitivity_s.setValue(5)
        self.sensitivity_s.setToolTip("Raise to detect more dropouts at the cost of also selecting dips in the signal that are not dropouts")

        buttons = (
            (self.before_after_l, self.before_after_s),
            (self.surrounding_l, self.surrounding_s), (self.gain_l, self.gain_s), (self.width_l, self.width_s), (self.sensitivity_l, self.sensitivity_s)
        )
        vbox(self, grid(buttons))

    @property
    def before_after(self):
        return self.before_after_s.value()

    @before_after.setter
    def before_after(self, v):
        self.before_after_s.setValue(v)

    @property
    def surrounding(self):
        return self.surrounding_s.value()

    @surrounding.setter
    def surrounding(self, v):
        self.surrounding_s.setValue(v)

    @property
    def width(self):
        return self.width_s.value()

    @width.setter
    def width(self, v):
        self.width_s.setValue(v)

    @property
    def sensitivity(self):
        return self.sensitivity_s.value()

    @sensitivity.setter
    def sensitivity(self, v):
        self.sensitivity_s.setValue(v)


class DropoutWidget(QtWidgets.QGroupBox):
    def __init__(self, ):
        super().__init__("Alignment")
        mode_l = QtWidgets.QLabel("Mode")
        self.mode_c = QtWidgets.QComboBox(self)
        self.mode_c.addItems(("Heuristic", "Heuristic New", "MaxMono"))
        self.mode_c.currentIndexChanged.connect(self.toggle_mode)

        self.num_bands_l = QtWidgets.QLabel("Bands")
        self.num_bands_s = QtWidgets.QSpinBox()
        self.num_bands_s.setRange(1, 6)
        self.num_bands_s.setSingleStep(1)
        self.num_bands_s.setValue(3)
        self.num_bands_s.setToolTip("Number of bands across which the intensity of a dropout is evaluated")

        self.f_lower_l = QtWidgets.QLabel("Lower")
        self.f_lower_s = QtWidgets.QSpinBox()
        self.f_lower_s.setRange(1, 20000)
        self.f_lower_s.setSingleStep(500)
        self.f_lower_s.setValue(3000)
        self.f_lower_s.setSuffix(" Hz")
        self.f_lower_s.setToolTip("Lower boundary frequency for dropout detection.")

        self.f_upper_l = QtWidgets.QLabel("Upper")
        self.f_upper_s = QtWidgets.QSpinBox()
        self.f_upper_s.setRange(1, 20000)
        self.f_upper_s.setSingleStep(500)
        self.f_upper_s.setValue(12000)
        self.f_upper_s.setSuffix(" Hz")
        self.f_upper_s.setToolTip("Upper boundary frequency for dropout detection.")

        self.max_slope_l = QtWidgets.QLabel("Max Slope")
        self.max_slope_s = QtWidgets.QDoubleSpinBox()
        self.max_slope_s.setRange(0.0, 10)
        self.max_slope_s.setSingleStep(.1)
        self.max_slope_s.setValue(0.5)
        self.max_slope_s.setSuffix(" dB")
        self.max_slope_s.setToolTip("Absolute slope between dB to the left and right of a dropout candidate.")

        self.max_width_l = QtWidgets.QLabel("Max Width")
        self.max_width_s = QtWidgets.QDoubleSpinBox()
        self.max_width_s.setRange(0.000001, 2)
        self.max_width_s.setSingleStep(.01)
        self.max_width_s.setValue(.02)
        self.max_width_s.setSuffix(" s")
        self.max_width_s.setToolTip("Maximum length of a dropout - increase to capture wider dropouts")

        self.bottom_freedom_l = QtWidgets.QLabel("Bottom Freedom")
        self.bottom_freedom_s = QtWidgets.QDoubleSpinBox()
        self.bottom_freedom_s.setRange(0.0000001, 5)
        self.bottom_freedom_s.setSingleStep(.1)
        self.bottom_freedom_s.setValue(2)
        self.bottom_freedom_s.setToolTip("Clips the band's factor to x*gain of the band above")

        buttons = (
            (mode_l, self.mode_c,), (self.num_bands_l, self.num_bands_s), (self.f_upper_l, self.f_upper_s),
            (self.f_lower_l, self.f_lower_s), (self.max_slope_l, self.max_slope_s),
            (self.max_width_l, self.max_width_s),
            (self.bottom_freedom_l, self.bottom_freedom_s))
        vbox(self, grid(buttons))

    def toggle_mode(self):
        b = ("Heuristic" in self.mode_c.currentText())
        self.num_bands_l.setVisible(b)
        self.num_bands_s.setVisible(b)
        self.f_lower_l.setVisible(b)
        self.f_lower_s.setVisible(b)
        self.f_upper_l.setVisible(b)
        self.f_upper_s.setVisible(b)
        self.max_slope_l.setVisible(b)
        self.max_slope_s.setVisible(b)
        self.max_width_l.setVisible(b)
        self.max_width_s.setVisible(b)

    @property
    def mode(self, ): return self.mode_c.currentText()

    @property
    def f_lower(self): return self.f_lower_s.value()

    @property
    def f_upper(self): return self.f_upper_s.value()

    @property
    def num_bands(self): return self.num_bands_s.value()

    @property
    def max_slope(self): return self.max_slope_s.value()

    @property
    def max_width(self): return self.max_width_s.value()

    @property
    def bottom_freedom(self): return self.bottom_freedom_s.value()


class HPSSWidget(QtWidgets.QGroupBox):
    def __init__(self, ):
        super().__init__("HPSS")
        self.h_kernel_l = QtWidgets.QLabel("Harmonic Kernel")
        self.h_kernel_s = QtWidgets.QSpinBox()
        self.h_kernel_s.setRange(1, 99)
        self.h_kernel_s.setSingleStep(1)
        self.h_kernel_s.setValue(31)
        self.h_kernel_s.setToolTip("Kernel size for the harmonic median filter.")

        self.p_kernel_l = QtWidgets.QLabel("Percussive Kernel")
        self.p_kernel_s = QtWidgets.QSpinBox()
        self.p_kernel_s.setRange(1, 99)
        self.p_kernel_s.setSingleStep(1)
        self.p_kernel_s.setValue(31)
        self.p_kernel_s.setToolTip("Kernel size for the percussive median filter.")

        self.power_l = QtWidgets.QLabel("Power")
        self.power_s = QtWidgets.QDoubleSpinBox()
        self.power_s.setRange(0.0, 10)
        self.power_s.setSingleStep(.1)
        self.power_s.setValue(2.0)
        self.power_s.setToolTip("Exponent for the Wiener filter when constructing soft mask matrices.")

        self.margin_l = QtWidgets.QLabel("Margin")
        self.margin_s = QtWidgets.QDoubleSpinBox()
        self.margin_s.setRange(0.000001, 99)
        self.margin_s.setSingleStep(.01)
        self.margin_s.setValue(1.0)
        self.margin_s.setToolTip("margin size(s) for the masks")

        buttons = (
            (self.h_kernel_l, self.h_kernel_s), (self.p_kernel_l, self.p_kernel_s),
            (self.power_l, self.power_s), (self.margin_l, self.margin_s))
        vbox(self, grid(buttons))

    @property
    def h_kernel(self): return self.h_kernel_s.value()

    @property
    def p_kernel(self): return self.p_kernel_s.value()

    @property
    def power(self): return self.power_s.value()

    @property
    def margin(self): return self.margin_s.value()


class StackWidget(QtWidgets.QGroupBox):
    def __init__(self, stack):
        super().__init__("History")
        self.view = QtWidgets.QUndoView(stack)
        self.view.setCleanIcon(get_icon("save"))
        buttons = ((self.view,),)
        vbox(self, grid(buttons))


class OutputWidget(QtWidgets.QGroupBox, ConfigStorer):
    vars_for_saving = ("suffix", "sinc_quality", "resampling_mode")

    def __init__(self, ):
        super().__init__("Output")
        self.mode_l = QtWidgets.QLabel("Mode")
        self.mode_c = QtWidgets.QComboBox(self)
        self.mode_c.addItems(("Linear", "Sinc"))
        self.mode_c.currentIndexChanged.connect(self.toggle_resampling_quality)
        self.mode_c.setToolTip(
            "Linear is fast, but low quality. Always use Sinc for production!")
        self.sinc_quality_l = QtWidgets.QLabel("Quality")
        self.sinc_quality_s = QtWidgets.QSpinBox()
        self.sinc_quality_s.setRange(1, 100)
        self.sinc_quality_s.setSingleStep(1)
        self.sinc_quality_s.setValue(50)
        self.sinc_quality_s.setToolTip(
            "Number of input samples that contribute to each output sample.\n"
            "More samples = more quality, but slower. Only for Sinc mode.")
        self.toggle_resampling_quality()

        self.incremental_b = QtWidgets.QCheckBox("Keep takes")
        self.incremental_b.setChecked(False)
        self.incremental_b.setToolTip("If checked, adds an incrementing suffix (_0, _1, ...) for each time you export the audio.")
        self.suffix_index = 0

        buttons = ((self.mode_l, self.mode_c,), (self.sinc_quality_l, self.sinc_quality_s), (self.incremental_b,))
        vbox(self, grid(buttons))

    def toggle_resampling_quality(self):
        b = (self.mode_c.currentText() == "Sinc")
        self.sinc_quality_l.setVisible(b)
        self.sinc_quality_s.setVisible(b)

    def bump_index(self):
        if self.incremental_b.isChecked():
            self.suffix_index += 1

    @property
    def suffix(self):
        if self.incremental_b.isChecked():
            return f"_{self.suffix_index}"
        return ""

    @suffix.setter
    def suffix(self, suffix_str):
        if suffix_str:
            self.incremental_b.setChecked(True)
            self.suffix_index = int(suffix_str.replace("_", ""))

    @property
    def sinc_quality(self, ):
        return self.sinc_quality_s.value()

    @property
    def resampling_mode(self, ):
        return self.mode_c.currentText()

    @resampling_mode.setter
    def resampling_mode(self, t):
        self.mode_c.setCurrentText(t)


class InspectorWidget(QtWidgets.QLabel):
    def __init__(self, ):
        QtWidgets.QLabel.__init__(self, )
        font = QtGui.QFont("Monospace")
        font.setStyleHint(QtGui.QFont.TypeWriter)
        self.setFont(font)
        self.update_text(None, None)

    def update_text(self, click, f_max):
        t, f = self.get_t_f(click, f_max)
        self.setText(f"{pitch(f):>11} Note\n"
                     f"   {f:8.1f} Hz\n"
                     f"{units.sec_to_timestamp(t)}")

    def get_t_f(self, click, f_max):
        if click is not None:
            t, f = click[0:2]
            if t >= 0 and 0 <= f <= f_max:
                return t, f
        return 0, 0


class MainWindow(QtWidgets.QMainWindow):
    closing = QtCore.pyqtSignal()

    EXT = None
    STORE = {}

    def __init__(self, name, props_widget_cls, canvas_widget_cls, count):
        QtWidgets.QMainWindow.__init__(self)

        self.name = name
        self.resize(1200, 600)
        self.setWindowTitle(name)
        self.setWindowIcon(get_icon(name))

        self.cfg = config.load_config()

        self.props = props_widget_cls(parent=self, count=count)
        self.canvas = canvas_widget_cls(parent=self)
        self.canvas.props = self.props
        self.props.undo_stack.canvas = self.canvas

        for i, spectrum in enumerate(self.canvas.spectra):
            # special case for pan tool, which maps one file to two spectra
            if i == len(self.props.files_widget.files):
                i = 0
            w = self.props.files_widget.files[i]
            w.spectra.append(spectrum)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self.canvas.native)
        splitter.addWidget(self.props)
        self.setCentralWidget(splitter)

    def closeEvent(self, event):
        print("Closing main window!")
        self.closing.emit()
        return super().closeEvent(event)

    def update_title(self, file_name):
        self.setWindowTitle(f"{self.name} {file_name}")

    def add_to_menu(self, button_data):
        for btn in button_data:
            self._add_to_menu(*btn)

    def _add_to_menu(self, submenu, action_name, func, shortcut, icon_name=None):
        action = QtWidgets.QAction(action_name, self)
        if icon_name:
            icon = get_icon(icon_name)
            action.setIcon(icon)
        action.triggered.connect(func)
        if shortcut:
            action.setShortcut(shortcut)
        # self.actions[action_name.lower()] = action
        submenu.addAction(action)


class FilesWidget(QtWidgets.QGroupBox, ConfigStorer):
    """
    Holds several file widgets
    controls what happens when they are loaded
    """

    file_changed = QtCore.pyqtSignal(str)
    spectrum_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent, count, cfg={}, ask_user=True):
        super().__init__("Files")
        self.parent = parent
        # note: count must be 1 or 2
        # idiosyncratic order here so the complicated stuff can remain as is
        labels = ("Reference", "Source")
        self.files = [FileWidget(self, cfg, label, ask_user, cfg_key=f"dir_in_{label.lower()}") for label in labels[-count:]]
        for file in self.files:
            file.spectrum_changed.connect(self.spectrum_changed.emit)
            file.file_changed.connect(self.file_changed.emit)
        vbox2(self, self.files)

    def ask_open(self):
        # propagates the open action onto child widgets which then call a file selector
        for w in self.files:
            w.ask_open()

    @property
    def filepaths(self):
        return [w.filepath for w in self.files]

    def poll(self):
        # called by the child widgets after they have received a file
        # only continue if all slots are filled with files
        if all(self.filepaths):
            self.load()

    def on_load_file(self, filepaths):
        # logging.warning("No loading function defined!")
        pass

    def load(self):
        self.on_load_file(self.filepaths)

    def to_cfg(self, cfg):
        for file_widget in self.files:
            cfg[file_widget.description.lower()] = file_widget.filepath

    def from_cfg(self, cfg):
        # aliases for old style keys
        lut = {"reference": "ref", "source": "src"}
        for file_widget in self.files:
            fk = file_widget.description.lower()
            fp = cfg.get(fk, cfg.get(lut[fk]))
            if not os.path.isfile(fp):
                logging.warning(f"Could not find {fp}")
                return
            file_widget.accept_file(fp)


class ParamWidget(QtWidgets.QWidget):
    """
    Widget for editing parameters
    """

    def __init__(self, parent=None, count=1):
        super(ParamWidget, self).__init__(parent)
        self.parent = parent
        self.files_widget = FilesWidget(self, count, self.parent.cfg)
        self.display_widget = SpectrumSettingsWidget()
        self.tracing_widget = TracingWidget()
        self.filters_widget = FiltersWidget()
        self.output_widget = OutputWidget()
        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setAlignment(QtCore.Qt.AlignCenter)
        self.audio_widget = snd.AudioWidget()
        # only connect 1st file to playback
        for file in self.files_widget.files[:1]:
            file.file_changed.connect(self.audio_widget.load_audio)
        self.inspector_widget = InspectorWidget()
        self.alignment_widget = AlignmentWidget()
        self.dropout_widget = DropsWidget()
        self.undo_stack = UndoStack(self)
        self.stack_widget = StackWidget(self.undo_stack)
        self.buttons = [self.files_widget, self.display_widget, self.tracing_widget, self.alignment_widget, self.dropout_widget,
                        self.filters_widget, self.output_widget, self.stack_widget, self.progress_bar, self.audio_widget,
                        self.inspector_widget]
        vbox2(self, self.buttons)

    def save(self):
        """Save project with all required settings"""
        sync = {}
        for w in self.buttons:
            if isinstance(w, ConfigStorer):
                w.to_cfg(sync)
        for marker_name in self.parent.STORE:
            sync[marker_name] = list(marker.to_cfg() for marker in getattr(self.parent.canvas, marker_name))
        cfg_path = os.path.splitext(self.files_widget.filepaths[0])[0] + self.parent.EXT
        save_json(cfg_path, sync)
        self.undo_stack.setClean()

    @staticmethod
    def fmt_exts(exts):
        return ' '.join([f"*{e}" for e in exts])

    @property
    def sel_str(self):
        ftypes = {"Audio": (".flac", ".wav"), "Project": (self.parent.EXT,)}
        ftypes["All"] = sorted(v for vs in ftypes.values() for v in vs)
        allowed = [f"{k} files ({self.fmt_exts(exts)})" for k, exts in sorted(ftypes.items())]
        return ";;".join(allowed)

    def load(self):
        """Load project with all required settings"""
        print("load")
        file_path = \
        QtWidgets.QFileDialog.getOpenFileName(self, 'Open Project', self.parent.cfg.get("dir_in", "C:/"), self.sel_str)[0]
        if not os.path.isfile(file_path):
            return
        self.parent.cfg["dir_in"], filename = os.path.split(file_path)
        # new style project file
        if file_path.endswith(self.parent.EXT):
            sync = load_json(file_path)
            for w in self.buttons:
                if isinstance(w, ConfigStorer):
                    w.from_cfg(sync)
            _markers = []
            for marker_name, marker_class in self.parent.STORE.items():
                if marker_name in sync:
                    _markers.extend([marker_class.from_cfg(self.parent.canvas, *item) for item in sync[marker_name]])
        else:
            # old style project file or audio file
            self.files_widget.files[0].accept_file(file_path)
            _markers = list(self.parent.canvas.load_visuals_legacy())
        # Cleanup of old data
        self.parent.canvas.delete_traces(delete_all=True)
        self.undo_stack.push(AddAction(_markers))
        self.parent.update_title(filename)
        self.display_widget.update_fft_settings()



class PlotMainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_plot()

    def setup_plot(self):
        # a figure instance to plot on
        self.fig, self.ax = plt.subplots(nrows=1, ncols=1)
        self.ax.set_xlabel('Frequency (Hz)')
        self.ax.set_ylabel('Volume (dB)')
        # the range is not automatically fixed
        self.fig.patch.set_facecolor((53 / 255, 53 / 255, 53 / 255))
        self.ax.set_facecolor((35 / 255, 35 / 255, 35 / 255))
        # this is the Canvas Widget that displays the `figure`
        # it takes the `fig` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.fig)
        self.canvas.mpl_connect('button_press_event', self.onclick)
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)

    @contextlib.contextmanager
    def update_plot(self, x_label=None, y_label=None):
        # discards the old graph
        self.ax.clear()
        yield
        self.ax.set_xlabel(x_label)
        self.ax.set_ylabel(y_label)
        # refresh canvas
        self.canvas.draw()

    def onclick(self, event):
        pass
