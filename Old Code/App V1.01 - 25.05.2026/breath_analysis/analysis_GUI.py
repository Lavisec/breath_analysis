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
    QFileDialog, QMessageBox, QInputDialog, QTabWidget, QToolButton, QMenu, QAction
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
        display_results(self.tab_widget, results)


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
    Run the pipeline up to baseline calculation for a list of CSV files.
    Returns intermediate results ready for display and optional baseline editing.
    Call pipeline.continue_analysis(result) on each result to finish the analysis.

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

        def draw_pressure(idx, fig=fig, ax=ax, canvas=canvas, file_results=file_results,
                          filename=filename, editors=editors, baseline_lines=baseline_lines,
                          toolbar_ref=None):
            result = file_results[idx]
            ax.cla()
            ax.plot(result['time'], result['pressure_filtered'],
                    label='Filtered Pressure', color='orange', alpha=0.8)
            bl, = ax.plot(result['time'], result['baseline'],
                          label='Baseline', color='green', alpha=0.8, linewidth=2)
            ax.set_title(filename + (f' — pressure{idx + 1}' if n_cols > 1 else ''))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Pressure (arbitrary units)')
            ax.legend()
            baseline_lines[0] = bl
            editors[0] = BaselineEditor(ax, bl, canvas)
            canvas.draw_idle()

        def draw_breath_analysis(idx, fig=fig, ax=ax, canvas=canvas,
                                 file_results=file_results, filename=filename, n_cols=n_cols):
            result = file_results[idx]
            # Run post-baseline analysis if it hasn't been done yet
            if 'event_list' not in result:
                file_results[idx] = pipeline.continue_analysis(result)
                result = file_results[idx]

            ax.cla()
            press = result['pressure_resampled']
            time = result['time_resampled']
            ax.plot(time, press, color='blue', alpha=0.4)
            # ax.plot(time, result['pressure_resampled_lp'], color='cyan', alpha=0.6)

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

            ax.axhline(result['inhale_parameters']['threshold_dict']['amplitude_threshold'],
                       linestyle='--', color='cyan', label='Thresholds')
            ax.axhline(result['exhale_parameters']['threshold_dict']['amplitude_threshold'],
                       linestyle='--', color='cyan')

            ax.set_title(filename + (f' — pressure{idx + 1}' if n_cols > 1 else '') +
                         ' — Breath Analysis')
            ax.set_xlabel('Sample (resampled)')
            ax.set_ylabel('Pressure (baseline corrected)')
            ax.legend()
            canvas.draw_idle()

        draw_pressure(0)

        toolbar = NavigationToolbar(canvas, tab_widget)

        # --- baseline edit buttons (defined before pressure menu so make_switch can capture them) ---
        btn_edit = QPushButton('Edit Baseline')
        btn_stop = QPushButton('Stop Editing Baseline')
        btn_undo = QPushButton('Undo')
        btn_continue = QPushButton('Continue to Breath Analysis')
        btn_back = QPushButton('Back to Pre-Processing')
        btn_independent = QPushButton('Analyze Independent Events')
        btn_export = QPushButton('Export')

        btn_stop.setVisible(False)
        btn_undo.setVisible(False)
        btn_back.setVisible(False)
        btn_independent.setVisible(False)
        btn_export.setVisible(False)

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
                            btn_independent=btn_independent, btn_export=btn_export):
                def switch():
                    if state['index'] == idx:
                        return
                    # stop any active editing before switching
                    if editors[0] is not None:
                        editors[0].set_active(False)
                    state['index'] = idx
                    btn.setText(name)
                    # Always hide edit-in-progress buttons when switching
                    btn_stop.setVisible(False)
                    btn_undo.setVisible(False)
                    if state['modes'][idx] == 'baseline':
                        btn_edit.setVisible(True)
                        btn_continue.setVisible(True)
                        btn_back.setVisible(False)
                        btn_independent.setVisible(False)
                        btn_export.setVisible(False)
                        draw_baseline(idx)
                    else:
                        btn_edit.setVisible(False)
                        btn_continue.setVisible(False)
                        btn_back.setVisible(True)
                        btn_independent.setVisible(True)
                        btn_export.setVisible(True)
                        draw_breath(idx)
                return switch

            action.triggered.connect(make_switch(i))
            pressure_menu.addAction(action)

        btn_pressure.setMenu(pressure_menu)

        def make_start(editors=editors, tb=toolbar):
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
                       draw=draw_pressure):
            def stop():
                ed = editors[0]
                if ed:
                    ed.set_active(False)
                    # Write the edited baseline back into the result and re-run
                    # the post-baseline steps of the pipeline.
                    idx = state['index']
                    result = file_results[idx]
                    result['baseline'] = ed._baseline.copy()
                    file_results[idx] = pipeline.continue_analysis(result)
                    # Redraw so the plot reflects any downstream changes
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
                          btn_back=btn_back, btn_independent=btn_independent,
                          btn_export=btn_export):
            def on_continue():
                ed = editors[0]
                if ed is not None and ed.active:
                    # Save the edited baseline and re-run the pipeline, same as Stop Editing
                    ed.set_active(False)
                    idx = state['index']
                    result = file_results[idx]
                    result['baseline'] = ed._baseline.copy()
                    file_results[idx] = pipeline.continue_analysis(result)
                    btn_stop.setVisible(False)
                    btn_undo.setVisible(False)
                elif ed is not None:
                    ed.set_active(False)
                state['modes'][state['index']] = 'breath_analysis'
                btn_edit.setVisible(False)
                btn_continue.setVisible(False)
                btn_back.setVisible(True)
                btn_independent.setVisible(True)
                btn_export.setVisible(True)
                draw_breath(state['index'])
            return on_continue

        btn_continue.clicked.connect(make_continue())

        def make_back(state=state, editors=editors, draw_baseline=draw_pressure,
                      btn_edit=btn_edit, btn_continue=btn_continue,
                      btn_back=btn_back, btn_independent=btn_independent,
                      btn_export=btn_export):
            def on_back():
                state['modes'][state['index']] = 'baseline'
                btn_back.setVisible(False)
                btn_independent.setVisible(False)
                btn_export.setVisible(False)
                btn_edit.setVisible(True)
                btn_continue.setVisible(True)
                draw_baseline(state['index'])
            return on_back

        btn_back.clicked.connect(make_back())

        def make_open_independent(state=state, file_results=file_results):
            def open_independent():
                idx = state['index']
                result = file_results[idx]
                win = IndependentEventsWindow(result)
                win.show()
                # Keep a reference so the window isn't garbage-collected
                tab_content._independent_windows = getattr(tab_content, '_independent_windows', [])
                tab_content._independent_windows.append(win)
            return open_independent

        btn_independent.clicked.connect(make_open_independent())

        def make_export(state=state, file_results=file_results):
            def on_export():
                idx = state['index']
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
        btn_row3_layout.addWidget(btn_independent)

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
        ('Time', 'time_resampled'),
        ('Pressure', 'pressure_resampled'),
    ]

    # Event fields: (button label, internal key used in _build_event_columns)
    EVENT_FIELDS = [
        ('Event Types', 'event_type'),
        ('Event Start Times', 'event_start_time'),
        ('Event End Times', 'event_end_time'),
        ('Extremum Times', 'extremum_time'),
        ('Extremum Values', 'extremum_value'),
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

        layout.addStretch()

        btn_save = QPushButton('Save')
        btn_save.clicked.connect(self._on_save)
        layout.addWidget(btn_save)

    def _build_event_columns(self):
        """Return a dict of event-derived arrays, keyed by internal event key."""
        event_list = self.result.get('event_list') or []
        time_resampled = self.result.get('time_resampled')
        if not event_list or time_resampled is None:
            return {}

        event_types = []
        start_times = []
        end_times = []
        extremum_times = []
        extremum_values = []

        for ev in event_list:
            event_types.append(ev.get('type', ''))
            start_times.append(float(time_resampled[int(ev['start'])]))
            end_times.append(float(time_resampled[int(ev['end'])]))
            if 'extrimum' in ev:
                extremum_times.append(float(time_resampled[int(ev['extrimum'][0])]))
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

        if not selected_ts and not selected_ev:
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

        # Build DataFrame using Series so columns of different lengths are NaN-padded
        # Use button labels as column headers
        combined = {}
        for k, v in selected_ts.items():
            combined[key_to_label[k]] = pd.Series(v)
        for k, v in selected_ev.items():
            combined[key_to_label[k]] = pd.Series(v)

        df = pd.DataFrame(combined)
        df.to_csv(path, index=False)
        QMessageBox.information(self, 'Saved', f'Data saved to {path}')
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
