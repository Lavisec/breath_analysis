"""
Analysis GUI
------------
PyQt5-based GUI for breath pressure analysis.
Allows the user to load a CSV file or a directory of CSV files,
runs the baseline analysis from lavi_baseline_analysis, and displays
the resulting plots in an interactive embedded matplotlib canvas.
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QListWidget, QListWidgetItem, QLabel, QSizePolicy,
    QFileDialog, QMessageBox, QTabWidget, QToolButton, QMenu, QAction, QComboBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'breath_analysis'))

import pipeline


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.csv_files = []  # tracks all loaded CSV paths
        self.analysis_results = []  # latest results object shown in tabs
        self._export_all_windows = []
        self.init_ui()

    def init_ui(self):
        """Set up the main window layout, widgets, and toolbar."""
        self.setWindowTitle('Breath Analysis')
        self.resize(900, 600)

        # Central widget with a horizontal split:
        # left panel = file controls, right panel = plot area (placeholder for now)
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Left panel
        file_controls = self.create_file_controls()
        main_layout.addWidget(file_controls)

        # Right panel: tabbed plot area
        self.plot_area = self.create_plot_area()
        main_layout.addWidget(self.plot_area, stretch=3)

    def create_toolbar(self):
        """Create the matplotlib navigation toolbar (zoom, pan, home, save)."""
        # Toolbar is created per-tab in display_results, alongside each canvas.
        pass

    def create_file_controls(self):
        """Create the file input area: upload file / upload directory buttons and a file list display."""
        panel = QWidget()
        panel.setFixedWidth(260)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Button
        btn_load = QPushButton('Load CSV File / Directory...')
        btn_load.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_load.clicked.connect(self._on_load)
        layout.addWidget(btn_load)

        # File list label
        layout.addWidget(QLabel('Loaded files:'))

        # List of loaded files
        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        # Run button
        btn_run = QPushButton('Run Analysis')
        btn_run.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        btn_run.clicked.connect(self._on_run)
        layout.addWidget(btn_run)

        self.btn_export_all = QPushButton('Export All Results')
        self.btn_export_all.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.btn_export_all.setVisible(False)
        self.btn_export_all.clicked.connect(self._on_export_all)
        layout.addWidget(self.btn_export_all)

        return panel

    def create_plot_area(self):
        """Create the embedded matplotlib canvas where plots will be rendered."""
        self.tab_widget = QTabWidget()
        return self.tab_widget

    def _on_load(self):
        """Slot for the Load CSV File / Directory button."""
        path = browse_file_or_directory(self)
        if not path:
            return
        if os.path.isdir(path):
            new_files = collect_csv_files(path)
            if not new_files:
                QMessageBox.warning(self, 'No CSV files', 'No CSV files found in the selected directory.')
                return
            self._add_files(new_files, parent_dir=path)
        else:
            self._add_files([path])

    def _add_files(self, paths, parent_dir=None):
        """Add new CSV paths to the list, ignoring duplicates.

        If parent_dir is given, a non-selectable directory header is added
        followed by indented entries for each file inside it.
        """
        new_paths = [p for p in paths if p not in self.csv_files]
        if not new_paths:
            QMessageBox.information(self, 'No new files', 'All selected files are already loaded.')
            return

        if parent_dir is not None:
            # Directory header item (not selectable)
            header = QListWidgetItem(os.path.basename(parent_dir) + '/')
            header.setFlags(Qt.ItemIsEnabled)  # not selectable
            self.file_list.addItem(header)
            for p in new_paths:
                child = QListWidgetItem('    ' + os.path.basename(p))
                self.file_list.addItem(child)
                self.csv_files.append(p)
        else:
            for p in new_paths:
                self.file_list.addItem(os.path.basename(p))
                self.csv_files.append(p)

    def _on_run(self):
        """Slot for the Run Analysis button."""
        if not self.csv_files:
            QMessageBox.warning(self, 'No files', 'Please load at least one CSV file first.')
            return
        try:
            results = run_analysis_on_files(self.csv_files)
        except Exception as e:
            QMessageBox.critical(self, 'Analysis error', str(e))
            return
        self.analysis_results = results
        display_results(self.tab_widget, results)
        self.btn_export_all.setVisible(bool(results))

    def _on_export_all(self):
        """Open bulk export window for all currently analyzed results."""
        if not self.analysis_results:
            QMessageBox.warning(self, 'No results', 'Run analysis first to enable export for all results.')
            return
        # Open as a standalone top-level window (not embedded in the main UI).
        win = ExportAllResultsWindow(self.analysis_results)
        win.show()
        self._export_all_windows.append(win)


def browse_file_or_directory(parent):
    """
    Ask the user whether to load a single CSV file or a directory, then open
    the appropriate dialog.

    Parameters
    ----------
    parent : QWidget
        Parent widget for the dialog.

    Returns
    -------
    str or None
        Absolute path to the selected file or directory, or None if cancelled.
    """
    msg = QMessageBox(parent)
    msg.setWindowTitle('Load CSV File / Directory')
    msg.setText('What would you like to load?')
    btn_file = msg.addButton('CSV File', QMessageBox.AcceptRole)
    btn_dir = msg.addButton('Directory', QMessageBox.AcceptRole)
    msg.addButton('Cancel', QMessageBox.RejectRole)
    msg.exec_()

    clicked = msg.clickedButton()
    if clicked is btn_file:
        path, _ = QFileDialog.getOpenFileName(
            parent, 'Select CSV File', '', 'CSV Files (*.csv)'
        )
        return path if path else None
    elif clicked is btn_dir:
        directory = QFileDialog.getExistingDirectory(
            parent, 'Select Directory', ''
        )
        return directory if directory else None
    return None


def collect_csv_files(path):
    """
    Given a file path or directory path, return all relevant CSV file paths.

    Parameters
    ----------
    path : str
        Path to a single CSV file or a directory.

    Returns
    -------
    list[str]
        List of absolute paths to CSV files.
    """
    if os.path.isfile(path):
        return [path] if path.endswith('.csv') else []
    elif os.path.isdir(path):
        return sorted(
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith('.csv')
        )
    return []


def run_analysis_on_files(csv_files):
    """
    Run pipeline preprocessing up to baseline calculation for a list of CSV files.
    Returns intermediate results ready for display and optional baseline editing.

    Parameters
    ----------
    csv_files : list[str]
        List of paths to CSV files to process.

    Returns
    -------
    list[list[dict]]
        One list per file; each inner list has one result dict per pressure column.
    """
    results = []
    for file_path in csv_files:
        results.append(pipeline.analyze_file_to_baseline(file_path))
    return results


def display_results(tab_widget, results):
    """
    Render the analysis results onto the embedded matplotlib canvas.

    Parameters
    ----------
    tab_widget : QTabWidget
        The tab widget to populate with one tab per result.
    results : list[dict]
        Output of run_analysis_on_files — one entry per CSV file.
    """
    tab_widget.clear()
    for file_results in results:
        if not file_results:
            continue
        filename = os.path.basename(file_results[0]['filename'])
        n_cols = len(file_results)
        pressure_names = [f'pressure{i + 1}' for i in range(n_cols)]

        # --- shared figure with one axes (redrawn when pressure changes) ---
        fig = Figure(figsize=(6, 4), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(1, 1, 1)

        # State: which pressure index is currently shown, and which view is active
        # 'modes' tracks stage independently per pressure index
        state = {'index': 0, 'modes': ['baseline'] * n_cols}
        editors = [None]          # one editor slot, replaced on switch
        baseline_lines = [None]   # same
        view_limits = [None]      # saved xlim/ylim: (xlim, ylim) tuple or None

        def draw_pressure(idx, fig=fig, ax=ax, canvas=canvas, file_results=file_results,
                          filename=filename, editors=editors, baseline_lines=baseline_lines,
                          toolbar_ref=None):
            result = file_results[idx]
            ax.cla()
            ax.plot(result['time'], result['pressure'],
                    label='Filtered Pressure', color='orange', alpha=0.8)
            bl, = ax.plot(result['time'], result['baseline'],
                          label='Baseline', color='green', alpha=0.8, linewidth=2)
            ax.set_title(filename + (f' — pressure{idx + 1}' if n_cols > 1 else ''))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Pressure (arbitrary units)')
            ax.legend()
            ax.grid(True)
            baseline_lines[0] = bl
            editors[0] = BaselineEditor(ax, bl, canvas)
            canvas.draw_idle()

        def draw_breath_analysis(idx, fig=fig, ax=ax, canvas=canvas,
                                 file_results=file_results, filename=filename, n_cols=n_cols, view_limits=view_limits, preserve_view=True):
            result = file_results[idx]
            # Run post-baseline analysis if it hasn't been done yet.
            # We also ensure zelano_parameters exist for export.
            if 'event_list' not in result or 'zelano_parameters' not in result:
                file_results[idx] = pipeline.continue_analysis(result)
                result = file_results[idx]

            # Save current view limits before clearing (if preserve_view and any data exists)
            saved_limits = None
            if preserve_view:
                try:
                    if ax.has_data():
                        saved_limits = (ax.get_xlim(), ax.get_ylim())
                except Exception:
                    pass

            ax.cla()
            press = result['pressure_upsampled']
            time = result['time_upsampled']
            ax.plot(time, press, color='blue', alpha=0.4)
            # ax.plot(time, result['pressure_upsampled_lp'], color='cyan', alpha=0.6)

            seen_labels = set()
            for event in result['event_list']:
                if event['type'] == 'inhale':
                    label = 'Inhale' if 'Inhale' not in seen_labels else ''
                    seen_labels.add('Inhale')
                    ax.plot(time[event['start']:event['end']], press[event['start']:event['end']],
                            color='red', label=label)
                elif event['type'] == 'exhale':
                    label = 'Exhale' if 'Exhale' not in seen_labels else ''
                    seen_labels.add('Exhale')
                    ax.plot(time[event['start']:event['end']], press[event['start']:event['end']],
                            color='green', label=label)
                else:
                    label = 'Pause' if 'Pause' not in seen_labels else ''
                    seen_labels.add('Pause')
                    ax.plot(time[event['start']:event['end']], press[event['start']:event['end']],
                            color='black', label=label)

            inh_th = result.get('inh_amp_th')
            exh_th = result.get('exh_amp_th')
            if inh_th is not None and exh_th is not None:
                inh_th_u = np.interp(time, result['time'], inh_th)
                exh_th_u = np.interp(time, result['time'], exh_th)
                ax.plot(time, inh_th_u, linestyle='--', color='cyan', label='Thresholds')
                ax.plot(time, exh_th_u, linestyle='--', color='cyan')

            ax.set_title(filename + (f' — pressure{idx + 1}' if n_cols > 1 else '') +
                         ' — Breath Analysis')
            ax.set_xlabel('Sample (resampled)')
            ax.set_ylabel('Pressure (baseline corrected)')
            ax.legend()
            ax.grid(True)

            # Restore saved view limits if available
            if saved_limits is not None:
                try:
                    ax.set_xlim(saved_limits[0])
                    ax.set_ylim(saved_limits[1])
                except Exception:
                    pass

            canvas.draw_idle()

        draw_pressure(0)

        toolbar = NavigationToolbar(canvas, tab_widget)

        # --- baseline edit buttons (defined before pressure menu so make_switch can capture them) ---
        btn_edit = QPushButton('Edit Baseline')
        btn_stop = QPushButton('Stop Editing Baseline')
        btn_undo = QPushButton('Undo')
        btn_continue = QPushButton('Continue to Breath Analysis')
        btn_back = QPushButton('Back to Pre-Processing')
        btn_choose_event = QPushButton('Choose Event')
        btn_export = QPushButton('Export')

        btn_stop.setVisible(False)
        btn_undo.setVisible(False)
        btn_back.setVisible(False)
        btn_choose_event.setVisible(False)
        btn_export.setVisible(False)

        # One-shot click handler id for "Choose Event" mode.
        choose_event_cid = [None]

        def clear_choose_event_mode(canvas=canvas, choose_event_cid=choose_event_cid):
            if choose_event_cid[0] is not None:
                canvas.mpl_disconnect(choose_event_cid[0])
                choose_event_cid[0] = None
            canvas.setCursor(Qt.ArrowCursor)

        # --- pressure selector button (top-right) ---
        btn_pressure = QToolButton()
        btn_pressure.setText(pressure_names[0])
        btn_pressure.setPopupMode(QToolButton.InstantPopup)

        pressure_menu = QMenu(btn_pressure)
        for i, name in enumerate(pressure_names):
            action = QAction(name, pressure_menu)

            def make_switch(idx, name=name, btn=btn_pressure, state=state,
                            editors=editors, draw_baseline=draw_pressure,
                            draw_breath=draw_breath_analysis,
                            btn_edit=btn_edit, btn_stop=btn_stop, btn_undo=btn_undo,
                            btn_continue=btn_continue, btn_back=btn_back,
                            btn_choose_event=btn_choose_event, btn_export=btn_export,
                            clear_choose_event_mode=clear_choose_event_mode):
                def switch():
                    if state['index'] == idx:
                        return
                    # stop any active editing before switching
                    if editors[0] is not None:
                        editors[0].set_active(False)
                    clear_choose_event_mode()
                    state['index'] = idx
                    btn.setText(name)
                    # Always hide edit-in-progress buttons when switching
                    btn_stop.setVisible(False)
                    btn_undo.setVisible(False)
                    if state['modes'][idx] == 'baseline':
                        btn_edit.setVisible(True)
                        btn_continue.setVisible(True)
                        btn_back.setVisible(False)
                        btn_choose_event.setVisible(False)
                        btn_export.setVisible(False)
                        draw_baseline(idx)
                    else:
                        btn_edit.setVisible(False)
                        btn_continue.setVisible(False)
                        btn_back.setVisible(True)
                        btn_choose_event.setVisible(True)
                        btn_export.setVisible(True)
                        draw_breath(idx)
                return switch

            action.triggered.connect(make_switch(i))
            pressure_menu.addAction(action)

        btn_pressure.setMenu(pressure_menu)

        def make_start(editors=editors, tb=toolbar,
                       btn_edit=btn_edit, btn_stop=btn_stop, btn_undo=btn_undo):
            def start():
                # Deactivate any active toolbar mode (zoom/pan) before editing
                try:
                    if tb.mode.name in ('ZOOM', 'PAN'):
                        tb.zoom() if tb.mode.name == 'ZOOM' else tb.pan()
                except Exception:
                    pass
                if editors[0]:
                    editors[0].set_active(True)
                btn_edit.setVisible(False)
                btn_stop.setVisible(True)
                btn_undo.setVisible(True)
            return start

        def make_stop(editors=editors, state=state, file_results=file_results,
                       draw=draw_pressure,
                       btn_edit=btn_edit, btn_stop=btn_stop, btn_undo=btn_undo):
            def stop():
                ed = editors[0]
                if ed:
                    ed.set_active(False)
                    idx = state['index']
                    file_results[idx] = pipeline.update_baseline(file_results[idx], ed._baseline)
                    draw(idx)
                btn_stop.setVisible(False)
                btn_undo.setVisible(False)
                btn_edit.setVisible(True)
            return stop

        def make_undo(editors=editors):
            def do_undo():
                if editors[0]:
                    editors[0].undo()
            return do_undo

        btn_edit.clicked.connect(make_start())
        btn_stop.clicked.connect(make_stop())
        btn_undo.clicked.connect(make_undo())

        def make_continue(state=state, editors=editors, draw_breath=draw_breath_analysis,
                          file_results=file_results,
                          btn_edit=btn_edit, btn_continue=btn_continue,
                          btn_stop=btn_stop, btn_undo=btn_undo,
                          btn_back=btn_back, btn_choose_event=btn_choose_event,
                          btn_export=btn_export):
            def on_continue():
                ed = editors[0]
                if ed is not None and ed.active:
                    ed.set_active(False)
                    idx = state['index']
                    file_results[idx] = pipeline.update_baseline(file_results[idx], ed._baseline)
                    btn_stop.setVisible(False)
                    btn_undo.setVisible(False)
                elif ed is not None:
                    ed.set_active(False)
                state['modes'][state['index']] = 'breath_analysis'
                btn_edit.setVisible(False)
                btn_continue.setVisible(False)
                btn_back.setVisible(True)
                btn_choose_event.setVisible(True)
                btn_export.setVisible(True)
                draw_breath(state['index'], preserve_view=False)
            return on_continue

        btn_continue.clicked.connect(make_continue())

        def make_back(state=state, editors=editors, draw_baseline=draw_pressure,
                      btn_edit=btn_edit, btn_continue=btn_continue,
                      btn_back=btn_back, btn_choose_event=btn_choose_event,
                      btn_export=btn_export, clear_choose_event_mode=clear_choose_event_mode):
            def on_back():
                clear_choose_event_mode()
                state['modes'][state['index']] = 'baseline'
                btn_back.setVisible(False)
                btn_choose_event.setVisible(False)
                btn_export.setVisible(False)
                btn_edit.setVisible(True)
                btn_continue.setVisible(True)
                draw_baseline(state['index'])
            return on_back

        btn_back.clicked.connect(make_back())

        def make_choose_event(state=state, file_results=file_results, ax=ax, canvas=canvas,
                              choose_event_cid=choose_event_cid,
                              clear_choose_event_mode=clear_choose_event_mode,
                              draw_breath=draw_breath_analysis):
            def choose_event():
                idx = state['index']
                result = file_results[idx]
                if 'event_list' not in result:
                    file_results[idx] = pipeline.continue_analysis(result)
                    result = file_results[idx]

                event_list = result.get('event_list') or []
                time_axis = result.get('time_upsampled')
                if not event_list or time_axis is None or len(time_axis) == 0:
                    QMessageBox.information(tab_widget, 'No events',
                                            'No events available for this pressure.')
                    return

                clear_choose_event_mode()
                canvas.setCursor(Qt.CrossCursor)

                def on_click(event):
                    if event.inaxes is not ax or event.xdata is None:
                        return
                    clear_choose_event_mode()

                    pos = np.searchsorted(time_axis, event.xdata)
                    sample_idx = int(np.clip(pos, 0, len(time_axis) - 1))
                    if sample_idx > 0 and abs(time_axis[sample_idx] - event.xdata) > abs(time_axis[sample_idx - 1] - event.xdata):
                        sample_idx -= 1

                    chosen = None
                    chosen_idx = None
                    for ev_idx, ev in enumerate(event_list):
                        if int(ev['start']) <= sample_idx <= int(ev['end']):
                            chosen = ev
                            chosen_idx = ev_idx
                            break

                    if chosen is None:
                        QMessageBox.information(
                            tab_widget,
                            'No matching event',
                            f'No event contains index {sample_idx}.'
                        )
                        return

                    def on_event_type_changed():
                        # Redraw the main breath-analysis plot to reflect updated event colors/types.
                        if state['modes'][idx] == 'breath_analysis' and state['index'] == idx:
                            draw_breath(idx)

                    win = ChosenEventWindow(result, event_list, chosen_idx, sample_idx,
                                            on_event_type_changed=on_event_type_changed)
                    win.show()
                    tab_content._chosen_event_windows = getattr(tab_content, '_chosen_event_windows', [])
                    tab_content._chosen_event_windows.append(win)

                choose_event_cid[0] = canvas.mpl_connect('button_press_event', on_click)

            return choose_event

        btn_choose_event.clicked.connect(make_choose_event())

        def make_export(state=state, file_results=file_results):
            def on_export():
                idx = state['index']
                result = file_results[idx]
                if 'zelano_parameters' not in result:
                    file_results[idx] = pipeline.continue_analysis(result)
                    result = file_results[idx]
                win = ExportWindow(result)
                win.show()
                tab_content._export_windows = getattr(tab_content, '_export_windows', [])
                tab_content._export_windows.append(win)
            return on_export

        btn_export.clicked.connect(make_export())

        # Top bar: toolbar on the left, pressure selector + export on the right
        top_bar = QWidget()
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(0, 0, 0, 0)
        top_bar_layout.addWidget(toolbar)
        top_bar_layout.addStretch()
        if n_cols > 1:
            top_bar_layout.addWidget(QLabel('Pressure:'))
            top_bar_layout.addWidget(btn_pressure)

        export_bar = QWidget()
        export_bar_layout = QHBoxLayout(export_bar)
        export_bar_layout.setContentsMargins(0, 0, 0, 0)
        export_bar_layout.addStretch()
        export_bar_layout.addWidget(btn_export)

        btn_row = QWidget()
        btn_row_layout = QHBoxLayout(btn_row)
        btn_row_layout.setContentsMargins(0, 0, 0, 0)
        btn_row_layout.addWidget(btn_edit)
        btn_row_layout.addWidget(btn_stop)
        btn_row_layout.addWidget(btn_undo)

        btn_row2 = QWidget()
        btn_row2_layout = QHBoxLayout(btn_row2)
        btn_row2_layout.setContentsMargins(0, 0, 0, 0)
        btn_row2_layout.addWidget(btn_continue)
        btn_row2_layout.addWidget(btn_back)

        btn_row3 = QWidget()
        btn_row3_layout = QHBoxLayout(btn_row3)
        btn_row3_layout.setContentsMargins(0, 0, 0, 0)
        btn_row3_layout.addWidget(btn_choose_event)

        tab_content = QWidget()
        layout = QVBoxLayout(tab_content)
        layout.addWidget(top_bar)
        layout.addWidget(export_bar)
        layout.addWidget(canvas)
        layout.addWidget(btn_row)
        layout.addWidget(btn_row2)
        layout.addWidget(btn_row3)

        tab_widget.addTab(tab_content, filename)


class ExportWindow(QWidget):
    """Window for selecting and exporting analysis data to CSV."""

    # Time-series fields: (button label, result-dict key)
    TIMESERIES_FIELDS = [
        ('Time', 'time_upsampled'),
        ('Pressure', 'pressure_upsampled'),
    ]

    # Event fields: (button label, internal key used in _build_event_columns)
    EVENT_FIELDS = [
        ('Event Types', 'event_type'),
        ('Event Start Times', 'event_start_time'),
        ('Event End Times', 'event_end_time'),
        ('Extremum Times', 'extremum_time'),
        ('Extremum Values', 'extremum_value'),
    ]

    # Zelano scalar parameters: (column header, zelano_parameters dict key)
    ZELANO_SCALAR_FIELDS = [
        ('Breathing Rate', 'breathing_rate'),
        ('Inter-Breath Interval', 'inter_breath_interval'),
        ('Tidal Volume', 'tidal_volume'),
        ('Minute Ventilation', 'minute_ventilation'),
        ('Duty Cycle', 'duty_cycle'),
        ('CV Duty Cycle', 'cv_duty_cycle'),
        ('CV Breathing Rate', 'cv_breathing_rate'),
        ('CV Breath Volumes', 'cv_breath_volumes'),
    ]

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle('Export Data')
        self.resize(520, 160)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Select columns to export:'))

        self._toggle_btns = {}
        checked_style = 'QPushButton:checked { background-color: #4a90d9; color: white; }'

        fields_row = QWidget()
        fields_layout = QHBoxLayout(fields_row)
        fields_layout.setContentsMargins(0, 0, 0, 0)
        for label, key in self.TIMESERIES_FIELDS + self.EVENT_FIELDS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setStyleSheet(checked_style)
            fields_layout.addWidget(btn)
            self._toggle_btns[key] = btn
        fields_layout.addStretch()
        layout.addWidget(fields_row)

        zelano_row = QWidget()
        zelano_layout = QHBoxLayout(zelano_row)
        zelano_layout.setContentsMargins(0, 0, 0, 0)
        btn_zelano = QPushButton('Zelano Parameters')
        btn_zelano.setCheckable(True)
        btn_zelano.setChecked(True)
        btn_zelano.setStyleSheet(checked_style)
        zelano_layout.addWidget(btn_zelano)
        zelano_layout.addStretch()
        self._toggle_btns['zelano'] = btn_zelano
        layout.addWidget(zelano_row)

        layout.addStretch()

        btn_save = QPushButton('Save')
        btn_save.clicked.connect(self._on_save)
        layout.addWidget(btn_save)

    def _build_event_columns(self):
        """Return a dict of event-derived arrays, keyed by internal event key."""
        event_list = self.result.get('event_list') or []
        time_axis = self.result.get('time_upsampled')
        if not event_list or time_axis is None:
            return {}

        event_types = []
        start_times = []
        end_times = []
        extremum_times = []
        extremum_values = []

        for ev in event_list:
            event_types.append(ev.get('type', ''))
            start_times.append(float(time_axis[int(ev['start'])]))
            end_times.append(float(time_axis[int(ev['end'])]))
            if 'extrimum' in ev:
                extremum_times.append(float(time_axis[int(ev['extrimum'][0])]))
                extremum_values.append(float(ev['extrimum'][1]))
            else:
                extremum_times.append(None)
                extremum_values.append(None)

        return {
            'event_type': event_types,
            'event_start_time': start_times,
            'event_end_time': end_times,
            'extremum_time': extremum_times,
            'extremum_value': extremum_values,
        }

    def _on_save(self):
        # Build key->label mapping for column naming
        key_to_label = {key: label for label, key in self.TIMESERIES_FIELDS + self.EVENT_FIELDS}

        ts_keys = {key for _, key in self.TIMESERIES_FIELDS}
        ev_keys = {key for _, key in self.EVENT_FIELDS}

        selected_ts = {key: self.result[key]
                       for key, btn in self._toggle_btns.items()
                       if key in ts_keys and btn.isChecked() and key in self.result}

        event_cols = self._build_event_columns()
        selected_ev = {key: event_cols[key]
                       for key, btn in self._toggle_btns.items()
                       if key in ev_keys and btn.isChecked() and key in event_cols}

        include_zelano = self._toggle_btns['zelano'].isChecked()
        zelano_cols = self._build_zelano_columns() if include_zelano else {}

        if not selected_ts and not selected_ev and not zelano_cols:
            QMessageBox.warning(self, 'Nothing selected', 'Please select at least one column to export.')
            return

        path, _ = QFileDialog.getSaveFileName(
            self, 'Save CSV', '', 'CSV Files (*.csv)'
        )
        if not path:
            return
        if not path.endswith('.csv'):
            path += '.csv'

        import pandas as pd
        import csv

        main_combined = {}
        for k, v in selected_ts.items():
            main_combined[key_to_label[k]] = pd.Series(v)
        for k, v in selected_ev.items():
            main_combined[key_to_label[k]] = pd.Series(v)

        main_df = pd.DataFrame(main_combined)

        if zelano_cols:
            zelano_df = pd.DataFrame({col: pd.Series(v) for col, v in zelano_cols.items()})
            main_col_names = list(main_df.columns)
            zelano_col_names = list(zelano_df.columns)
            n_main = len(main_col_names)
            n_zelano = len(zelano_col_names)
            max_rows = max(len(main_df), len(zelano_df))

            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Row 1: main headers | empty separator | 'Zelano Parameters' | empty for remaining zelano cols
                writer.writerow(main_col_names + [''] + ['Zelano Parameters'] + [''] * (n_zelano - 1))
                # Row 2: empty for main cols | empty separator | zelano column names
                writer.writerow([''] * n_main + [''] + zelano_col_names)
                # Data rows
                for i in range(max_rows):
                    row = []
                    for col in main_col_names:
                        row.append(main_df[col].iloc[i] if i < len(main_df) else '')
                    row.append('')  # separator
                    for col in zelano_col_names:
                        row.append(zelano_df[col].iloc[i] if i < len(zelano_df) else '')
                    writer.writerow(row)
        else:
            main_df.to_csv(path, index=False)

        QMessageBox.information(self, 'Saved', f'Data saved to {path}')
        self.close()

    def _build_zelano_columns(self):
        """Return a dict of Zelano parameter columns, keyed by human-readable label."""
        zp = self.result.get('zelano_parameters')
        if not zp:
            return {}
        cols = {}
        for label, key in self.ZELANO_SCALAR_FIELDS:
            cols[label] = [zp.get(key)]
        # Per-breath volumes as a column
        cols['Inhale-Exhale Volumes'] = zp.get('inhale_exhale_volumes', [])
        return cols


class ExportAllResultsWindow(QWidget):
    """Window for exporting all analyzed results into one new directory."""

    TIMESERIES_FIELDS = ExportWindow.TIMESERIES_FIELDS
    EVENT_FIELDS = ExportWindow.EVENT_FIELDS
    ZELANO_SCALAR_FIELDS = ExportWindow.ZELANO_SCALAR_FIELDS

    def __init__(self, results, parent=None):
        super().__init__(parent)
        self.setWindowFlag(Qt.Window, True)
        self.results = results  # list[list[result_dict]] shared with UI; includes live edits
        self.setWindowTitle('Export All Results')
        self.resize(560, 180)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Select columns to export for all results:'))

        self._toggle_btns = {}
        checked_style = 'QPushButton:checked { background-color: #4a90d9; color: white; }'

        fields_row = QWidget()
        fields_layout = QHBoxLayout(fields_row)
        fields_layout.setContentsMargins(0, 0, 0, 0)
        for label, key in self.TIMESERIES_FIELDS + self.EVENT_FIELDS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setStyleSheet(checked_style)
            fields_layout.addWidget(btn)
            self._toggle_btns[key] = btn
        fields_layout.addStretch()
        layout.addWidget(fields_row)

        zelano_row = QWidget()
        zelano_layout = QHBoxLayout(zelano_row)
        zelano_layout.setContentsMargins(0, 0, 0, 0)
        btn_zelano = QPushButton('Zelano Parameters')
        btn_zelano.setCheckable(True)
        btn_zelano.setChecked(True)
        btn_zelano.setStyleSheet(checked_style)
        zelano_layout.addWidget(btn_zelano)
        zelano_layout.addStretch()
        self._toggle_btns['zelano'] = btn_zelano
        layout.addWidget(zelano_row)

        layout.addStretch()

        btn_save = QPushButton('Save')
        btn_save.clicked.connect(self._on_save)
        layout.addWidget(btn_save)

    def _build_event_columns(self, result):
        """Return a dict of event-derived arrays, keyed by internal event key."""
        event_list = result.get('event_list') or []
        time_axis = result.get('time_upsampled')
        if not event_list or time_axis is None:
            return {}

        event_types = []
        start_times = []
        end_times = []
        extremum_times = []
        extremum_values = []

        for ev in event_list:
            event_types.append(ev.get('type', ''))
            start_times.append(float(time_axis[int(ev['start'])]))
            end_times.append(float(time_axis[int(ev['end'])]))
            if 'extrimum' in ev:
                extremum_times.append(float(time_axis[int(ev['extrimum'][0])]))
                extremum_values.append(float(ev['extrimum'][1]))
            else:
                extremum_times.append(None)
                extremum_values.append(None)

        return {
            'event_type': event_types,
            'event_start_time': start_times,
            'event_end_time': end_times,
            'extremum_time': extremum_times,
            'extremum_value': extremum_values,
        }

    def _build_zelano_columns(self, result):
        """Return a dict of Zelano parameter columns, keyed by human-readable label."""
        zp = result.get('zelano_parameters')
        if not zp:
            return {}
        cols = {}
        for label, key in self.ZELANO_SCALAR_FIELDS:
            cols[label] = [zp.get(key)]
        cols['Inhale-Exhale Volumes'] = zp.get('inhale_exhale_volumes', [])
        return cols

    def _write_single_result_csv(self, path, result, ts_keys, ev_keys, include_zelano):
        """Write one result dict to CSV using the same structure as ExportWindow."""
        key_to_label = {key: label for label, key in self.TIMESERIES_FIELDS + self.EVENT_FIELDS}

        selected_ts = {key: result[key] for key in ts_keys if key in result}
        event_cols = self._build_event_columns(result)
        selected_ev = {key: event_cols[key] for key in ev_keys if key in event_cols}
        zelano_cols = self._build_zelano_columns(result) if include_zelano else {}

        if not selected_ts and not selected_ev and not zelano_cols:
            raise ValueError('No selected columns are available for this result.')

        import pandas as pd
        import csv

        main_combined = {}
        for k, v in selected_ts.items():
            main_combined[key_to_label[k]] = pd.Series(v)
        for k, v in selected_ev.items():
            main_combined[key_to_label[k]] = pd.Series(v)

        main_df = pd.DataFrame(main_combined)

        if zelano_cols:
            zelano_df = pd.DataFrame({col: pd.Series(v) for col, v in zelano_cols.items()})
            main_col_names = list(main_df.columns)
            zelano_col_names = list(zelano_df.columns)
            n_main = len(main_col_names)
            n_zelano = len(zelano_col_names)
            max_rows = max(len(main_df), len(zelano_df))

            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(main_col_names + [''] + ['Zelano Parameters'] + [''] * (n_zelano - 1))
                writer.writerow([''] * n_main + [''] + zelano_col_names)
                for i in range(max_rows):
                    row = []
                    for col in main_col_names:
                        row.append(main_df[col].iloc[i] if i < len(main_df) else '')
                    row.append('')
                    for col in zelano_col_names:
                        row.append(zelano_df[col].iloc[i] if i < len(zelano_df) else '')
                    writer.writerow(row)
        else:
            main_df.to_csv(path, index=False)

    def _on_save(self):
        ts_keys = {key for _, key in self.TIMESERIES_FIELDS if self._toggle_btns[key].isChecked()}
        ev_keys = {key for _, key in self.EVENT_FIELDS if self._toggle_btns[key].isChecked()}
        include_zelano = self._toggle_btns['zelano'].isChecked()

        if not ts_keys and not ev_keys and not include_zelano:
            QMessageBox.warning(self, 'Nothing selected', 'Please select at least one column group to export.')
            return

        export_dir = QFileDialog.getExistingDirectory(self, 'Choose Directory to Save All Results', '')
        if not export_dir:
            return

        written = 0
        skipped = []
        used_names = set()

        for file_idx, file_results in enumerate(self.results):
            if not file_results:
                continue

            n_cols = len(file_results)
            for pressure_idx, result in enumerate(file_results):
                # Ensure analysis-derived fields exist for bulk export.
                if 'event_list' not in result or 'zelano_parameters' not in result:
                    file_results[pressure_idx] = pipeline.continue_analysis(result)
                    result = file_results[pressure_idx]

                base_name = os.path.splitext(os.path.basename(result.get('filename', f'dataset_{file_idx + 1}.csv')))[0]
                if n_cols > 1:
                    base_name = f'{base_name}_pressure{pressure_idx + 1}'

                candidate = f'{base_name}.csv'
                suffix = 2
                while candidate in used_names:
                    candidate = f'{base_name}_{suffix}.csv'
                    suffix += 1
                used_names.add(candidate)

                out_path = os.path.join(export_dir, candidate)
                try:
                    self._write_single_result_csv(out_path, result, ts_keys, ev_keys, include_zelano)
                    written += 1
                except Exception as e:
                    skipped.append(f'{candidate}: {e}')

        if written == 0:
            msg = 'No files were exported.'
            if skipped:
                msg += '\n\nDetails:\n' + '\n'.join(skipped[:8])
            QMessageBox.warning(self, 'Export completed with no files', msg)
            return

        if skipped:
            msg = (f'Exported {written} files to:\n{export_dir}\n\n'
                   f'Skipped {len(skipped)} files.\n\nDetails:\n' + '\n'.join(skipped[:8]))
            QMessageBox.warning(self, 'Export completed with warnings', msg)
        else:
            QMessageBox.information(self, 'Saved', f'Exported {written} files to:\n{export_dir}')
        self.close()


class IndependentEventsWindow(QWidget):
    """Window for analyzing individual breath events."""

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.result = result
        self.setWindowTitle('Analyze Independent Events')
        self.resize(800, 600)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel('Independent event analysis — coming soon.'))


class ChosenEventWindow(QWidget):
    """Window for displaying one selected event from the breath-analysis plot.

    The highlighted span edges can be grabbed with the mouse to change the
    start and end of the event.  Adjacent events are updated accordingly via
    pipeline.update_event_bounds.
    """

    def __init__(self, result, event_list, chosen_idx, sample_idx, on_event_type_changed=None, parent=None):
        super().__init__(parent)
        self.result = result
        self.event_list = event_list
        self.chosen_idx = chosen_idx
        self.sample_idx = sample_idx
        self._on_changed_cb = on_event_type_changed  # callback to refresh main window

        # Drag state
        self._dragging = None          # None, 'start', or 'end'
        self._drag_current_start = None
        self._drag_current_end = None

        # Plot references (set by _draw_plot)
        self.fig = None
        self.canvas = None
        self.ax = None
        self._span = None
        self._window_start = 0
        self._window_end = 0
        self._window_initialized = False
        self._current_start = 0
        self._current_end = 0

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        event = self.event_list[self.chosen_idx]
        event_type = event.get('type', 'event').capitalize()
        self.setWindowTitle(f'Chosen Event - {event_type}')
        self.resize(760, 460)

        layout = QVBoxLayout(self)

        self.summary_label = QLabel(self._summary_text())
        layout.addWidget(self.summary_label)

        self.fig = Figure(figsize=(7, 4), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(1, 1, 1)

        self._draw_plot()

        # Mouse event connections for edge dragging
        self.canvas.mpl_connect('button_press_event', self._on_press)
        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('button_release_event', self._on_release)

        layout.addWidget(self.canvas)

        type_row = QWidget()
        type_row_layout = QHBoxLayout(type_row)
        type_row_layout.setContentsMargins(0, 0, 0, 0)
        type_row_layout.addStretch()
        type_row_layout.addWidget(QLabel('Event Type:'))
        self.type_combo = QComboBox()
        self.type_combo.addItems(['Inhale', 'Exhale', 'Pause'])

        current_type = str(event.get('type', 'pause')).capitalize()
        if current_type not in ('Inhale', 'Exhale', 'Pause'):
            current_type = 'Pause'
        self.type_combo.setCurrentText(current_type)
        self.type_combo.currentTextChanged.connect(self._on_type_combo_changed)

        type_row_layout.addWidget(self.type_combo)
        type_row_layout.addStretch()
        layout.addWidget(type_row)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _summary_text(self):
        event = self.event_list[self.chosen_idx]
        return (
            f"Type: {event.get('type', '')} | "
            f"Start: {int(event.get('start', 0))} | "
            f"End: {int(event.get('end', 0))} | "
            f"Clicked Index: {int(self.sample_idx)}"
        )

    def _compute_window_bounds(self):
        """Return (window_start, window_end) indices for a 13x-event-width context."""
        event = self.event_list[self.chosen_idx]
        time = self.result.get('time_upsampled')
        if time is None or len(time) == 0:
            return 0, 0

        n = len(time)
        chosen_start = max(0, min(n - 1, int(event.get('start', 0))))
        chosen_end = max(chosen_start, min(n - 1, int(event.get('end', 0))))

        event_width = chosen_end - chosen_start + 1
        window_width = min(n, max(13 * event_width, 1))

        center = (chosen_start + chosen_end) / 2.0
        window_start = int(round(center - (window_width - 1) / 2.0))
        window_end = window_start + window_width - 1

        if window_start < 0:
            window_end += -window_start
            window_start = 0
        if window_end > n - 1:
            window_start = max(0, window_start - (window_end - (n - 1)))
            window_end = n - 1

        return window_start, window_end

    def _draw_plot(self):
        """Draw (or redraw) the event plot from scratch."""
        event = self.event_list[self.chosen_idx]
        pressure = self.result.get('pressure_upsampled')
        time = self.result.get('time_upsampled')

        self.ax.cla()
        self._span = None

        if pressure is None or time is None or len(time) == 0:
            self.canvas.draw_idle()
            return

        n = len(time)
        chosen_start = max(0, min(n - 1, int(event.get('start', 0))))
        chosen_end = max(chosen_start, min(n - 1, int(event.get('end', 0))))

        self._current_start = chosen_start
        self._current_end = chosen_end

        if not self._window_initialized:
            self._window_start, self._window_end = self._compute_window_bounds()
            self._window_initialized = True

        window_start = self._window_start
        window_end = self._window_end

        ctx_time = time[window_start:window_end + 1]
        ctx_pressure = pressure[window_start:window_end + 1]
        self.ax.plot(ctx_time, ctx_pressure, color='gray', linewidth=1.4, alpha=0.85)

        chosen_time = time[chosen_start:chosen_end + 1]
        chosen_pressure = pressure[chosen_start:chosen_end + 1]
        self.ax.plot(chosen_time, chosen_pressure, color='blue', linewidth=2.6, label='Chosen Event')

        self._span = self.ax.axvspan(time[chosen_start], time[chosen_end],
                                     color='gold', alpha=0.18, zorder=0)

        if 'extrimum' in event:
            ext_i = int(event['extrimum'][0])
            ext_v = float(event['extrimum'][1])
            if 0 <= ext_i < len(time):
                self.ax.scatter(time[ext_i], ext_v, color='magenta', zorder=4, label='Chosen Extremum')

        self.ax.set_xlim(time[window_start], time[window_end])
        event_type = event.get('type', 'event').capitalize()
        self.ax.set_title(f"{event_type} Event")
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Pressure')
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw_idle()

    def _get_edge_tolerance(self):
        """Return the x-data tolerance equivalent to 8 screen pixels."""
        try:
            bbox = self.ax.get_window_extent()
            xlim = self.ax.get_xlim()
            x_range = xlim[1] - xlim[0]
            if bbox.width > 0 and x_range > 0:
                return 8.0 * x_range / bbox.width
        except Exception:
            pass
        return 0.0

    def _update_span_preview(self, left_x, right_x):
        """Replace the axvspan to reflect the current drag position."""
        if self._span is not None:
            self._span.remove()
        self._span = self.ax.axvspan(left_x, right_x, color='gold', alpha=0.18, zorder=0)
        self.canvas.draw_idle()

    def _nearest_sample_idx(self, xdata, time):
        """Return the nearest sample index on the upsampled time axis."""
        pos = int(np.clip(np.searchsorted(time, xdata), 0, len(time) - 1))
        if pos > 0 and abs(time[pos] - xdata) > abs(time[pos - 1] - xdata):
            pos -= 1
        return pos

    def _find_event_idx_at_sample(self, sample_idx):
        """Return the event index that contains sample_idx, or None."""
        for idx, ev in enumerate(self.event_list):
            if int(ev['start']) <= sample_idx <= int(ev['end']):
                return idx
        return None

    def _center_window_on_chosen_event_keep_size(self):
        """Recenter x-limits on the chosen event while keeping window width fixed."""
        time = self.result.get('time_upsampled')
        if time is None or len(time) == 0:
            return

        n = len(time)
        window_width = self._window_end - self._window_start + 1
        if window_width <= 0:
            self._window_start, self._window_end = self._compute_window_bounds()
            window_width = self._window_end - self._window_start + 1

        ev = self.event_list[self.chosen_idx]
        ev_start = max(0, min(n - 1, int(ev.get('start', 0))))
        ev_end = max(ev_start, min(n - 1, int(ev.get('end', 0))))
        center = (ev_start + ev_end) / 2.0

        window_start = int(round(center - (window_width - 1) / 2.0))
        window_end = window_start + window_width - 1

        if window_start < 0:
            window_end += -window_start
            window_start = 0
        if window_end > n - 1:
            window_start = max(0, window_start - (window_end - (n - 1)))
            window_end = n - 1

        self._window_start = window_start
        self._window_end = window_end

    def _select_chosen_event(self, new_idx, clicked_sample_idx):
        """Switch highlighted event and redraw with same context width."""
        if new_idx == self.chosen_idx:
            return

        self.chosen_idx = new_idx
        self.sample_idx = clicked_sample_idx
        self._dragging = None
        self._drag_current_start = None
        self._drag_current_end = None

        self._center_window_on_chosen_event_keep_size()

        event_type = str(self.event_list[self.chosen_idx].get('type', 'pause')).capitalize()
        if event_type not in ('Inhale', 'Exhale', 'Pause'):
            event_type = 'Pause'
        self.type_combo.blockSignals(True)
        self.type_combo.setCurrentText(event_type)
        self.type_combo.blockSignals(False)

        self._draw_plot()
        self.summary_label.setText(self._summary_text())
        self.setWindowTitle(f"Chosen Event - {event_type}")

        if self._on_changed_cb is not None:
            self._on_changed_cb()

    # ------------------------------------------------------------------
    # Mouse event handlers for edge dragging
    # ------------------------------------------------------------------

    def _on_press(self, event):
        if event.button != 1 or event.inaxes is not self.ax or event.xdata is None:
            return
        time = self.result.get('time_upsampled')
        if time is None:
            return

        # Double-click selects a different event and recenters using fixed window size.
        if event.dblclick:
            sample_idx = self._nearest_sample_idx(event.xdata, time)
            clicked_idx = self._find_event_idx_at_sample(sample_idx)
            if clicked_idx is not None and clicked_idx != self.chosen_idx:
                self._select_chosen_event(clicked_idx, sample_idx)
            return

        tol = self._get_edge_tolerance()
        left_edge = time[self._current_start]
        right_edge = time[self._current_end]

        if abs(event.xdata - left_edge) <= tol:
            self._dragging = 'start'
            self._drag_current_start = self._current_start
            self._drag_current_end = self._current_end
        elif abs(event.xdata - right_edge) <= tol:
            self._dragging = 'end'
            self._drag_current_start = self._current_start
            self._drag_current_end = self._current_end
        else:
            self._dragging = None

    def _on_motion(self, event):
        time = self.result.get('time_upsampled')
        if time is None:
            return

        if self._dragging is None:
            # Update cursor to hint at draggable edges
            if event.inaxes is self.ax and event.xdata is not None:
                tol = self._get_edge_tolerance()
                if (abs(event.xdata - time[self._current_start]) <= tol or
                        abs(event.xdata - time[self._current_end]) <= tol):
                    self.canvas.setCursor(Qt.SizeHorCursor)
                else:
                    self.canvas.setCursor(Qt.ArrowCursor)
            else:
                self.canvas.setCursor(Qt.ArrowCursor)
            return

        if event.xdata is None:
            return

        # Snap to nearest sample
        pos = int(np.clip(np.searchsorted(time, event.xdata), 0, len(time) - 1))

        if self._dragging == 'start':
            # Allow extending into previous event; previous must have >= 1 sample
            if self.chosen_idx > 0:
                min_idx = int(self.event_list[self.chosen_idx - 1]['start']) + 1
            else:
                min_idx = 0
            new_start = max(min_idx, min(pos, self._drag_current_end - 1))
            self._drag_current_start = new_start
        else:  # 'end'
            if self.chosen_idx < len(self.event_list) - 1:
                # Allow extending into next event; next must have >= 1 sample
                max_idx = int(self.event_list[self.chosen_idx + 1]['end']) - 1
            else:
                max_idx = len(time) - 1
            new_end = max(self._drag_current_start + 1, min(pos, max_idx))
            self._drag_current_end = new_end

        self._update_span_preview(time[self._drag_current_start], time[self._drag_current_end])

    def _on_release(self, event):
        if event.button != 1 or self._dragging is None:
            return

        time = self.result.get('time_upsampled')
        if time is None:
            self._dragging = None
            return

        new_start = self._drag_current_start if self._dragging == 'start' else None
        new_end = self._drag_current_end if self._dragging == 'end' else None

        pipeline.update_event_bounds(self.result, self.chosen_idx, new_start, new_end)

        self._dragging = None
        self._drag_current_start = None
        self._drag_current_end = None

        self._draw_plot()
        self.summary_label.setText(self._summary_text())

        if self._on_changed_cb is not None:
            self._on_changed_cb()

    # ------------------------------------------------------------------
    # Type combo handler
    # ------------------------------------------------------------------

    def _on_type_combo_changed(self, selected_text):
        pipeline.change_event_type(self.result, self.chosen_idx, selected_text.lower())

        self.summary_label.setText(self._summary_text())
        self.setWindowTitle(f"Chosen Event - {selected_text}")

        if self._on_changed_cb is not None:
            self._on_changed_cb()


class BaselineEditor:
    """
    Handles interactive vertical dragging of a baseline line on a matplotlib axes.

    Must be activated via set_active(True) before dragging works.
    Click anywhere in the axes to grab and drag the baseline up/down.
    Undo restores the previous baseline position.
    """

    def __init__(self, ax, line, canvas):
        self._ax = ax
        self._line = line
        self._canvas = canvas
        self._baseline = np.array(line.get_ydata(), dtype=float)
        self._undo_stack = []
        self._dragging = False
        self._drag_start_y = None
        self.active = False

        canvas.mpl_connect('button_press_event', self._on_press)
        canvas.mpl_connect('motion_notify_event', self._on_motion)
        canvas.mpl_connect('button_release_event', self._on_release)

    def set_active(self, state):
        """Enable or disable edit mode."""
        self.active = state
        # Change cursor to indicate edit mode
        if state:
            self._canvas.setCursor(Qt.SizeVerCursor)
        else:
            self._canvas.setCursor(Qt.ArrowCursor)
            self._dragging = False

    def _on_press(self, event):
        if not self.active or event.button != 1 or event.inaxes is not self._ax:
            return
        self._undo_stack.append(self._baseline.copy())
        self._dragging = True
        self._drag_start_y = event.ydata

    def _on_motion(self, event):
        if not self._dragging:
            return
        # Use ydata if inside axes, otherwise get it from the figure coords
        if event.ydata is not None and event.inaxes is self._ax:
            y = event.ydata
        else:
            # Convert figure-space coords to data coords so dragging
            # continues even when cursor leaves the axes boundary
            try:
                inv = self._ax.transData.inverted()
                _, y = inv.transform((event.x, event.y))
            except Exception:
                return
        delta = y - self._drag_start_y
        self._baseline = self._undo_stack[-1] + delta
        self._line.set_ydata(self._baseline)
        self._canvas.draw_idle()

    def _on_release(self, event):
        if event.button != 1 or not self._dragging:
            return
        self._dragging = False
        self._drag_start_y = None

    def undo(self):
        """Restore the baseline to the state before the last drag."""
        if not self._undo_stack:
            return
        self._baseline = self._undo_stack.pop()
        self._line.set_ydata(self._baseline)
        self._canvas.draw_idle()


def main():
    """Entry point: create the QApplication and show the main window."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())



if __name__ == "__main__":
    main()
