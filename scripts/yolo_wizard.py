#!/usr/bin/env python3
"""
Qt wizard around ls_to_yolo.build_dataset().

Add one or more sources (each = a Label Studio export JSON + its videos folder),
pick an output dataset folder, set the val fraction, and click Build. The
build runs on a worker thread so the window stays responsive, and the log
panel streams progress.

Run:
    venv/bin/python scripts/yolo_wizard.py
"""
import sys
from pathlib import Path

from PySide6 import QtCore, QtWidgets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ls_to_yolo import build_dataset


class BuildWorker(QtCore.QThread):
    line = QtCore.Signal(str)
    done = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, sources, out, val_frac, seed):
        super().__init__()
        self.sources, self.out = sources, out
        self.val_frac, self.seed = val_frac, seed

    def run(self):
        try:
            result = build_dataset(
                self.sources, self.out, self.val_frac, self.seed,
                log=lambda s: self.line.emit(str(s)),
            )
            self.done.emit(result)
        except Exception as e:  # surface any failure in the UI, not the console
            self.failed.emit(f"{type(e).__name__}: {e}")


class Wizard(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Label Studio → YOLO dataset")
        self.resize(820, 560)
        self.worker = None

        root = QtWidgets.QVBoxLayout(self)

        root.addWidget(QtWidgets.QLabel(
            "Add one row per annotation batch. Each row pairs an export JSON "
            "with the videos folder it was annotated from."))

        # --- sources table ---
        self.table = QtWidgets.QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Export JSON", "Videos folder"])
        self.table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        root.addWidget(self.table, 1)

        row_btns = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("+ Add source…")
        rm = QtWidgets.QPushButton("Remove selected")
        add.clicked.connect(self.add_source)
        rm.clicked.connect(self.remove_selected)
        row_btns.addWidget(add)
        row_btns.addWidget(rm)
        row_btns.addStretch(1)
        root.addLayout(row_btns)

        # --- output + options ---
        form = QtWidgets.QFormLayout()
        out_row = QtWidgets.QHBoxLayout()
        self.out_edit = QtWidgets.QLineEdit("data/datasets/plate_v2")
        out_browse = QtWidgets.QPushButton("Browse…")
        out_browse.clicked.connect(self.pick_out)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(out_browse)
        out_w = QtWidgets.QWidget()
        out_w.setLayout(out_row)
        form.addRow("Output dataset:", out_w)

        self.val_spin = QtWidgets.QDoubleSpinBox()
        self.val_spin.setRange(0.0, 0.9)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.2)
        form.addRow("Val fraction:", self.val_spin)

        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 999999)
        form.addRow("Seed:", self.seed_spin)
        root.addLayout(form)

        # --- build button + log ---
        self.build_btn = QtWidgets.QPushButton("Build dataset")
        self.build_btn.clicked.connect(self.build)
        root.addWidget(self.build_btn)

        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("Build log will appear here…")
        root.addWidget(self.log, 1)

    # ---- source rows ----
    def add_source(self):
        export, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Label Studio export JSON", "data/annotations",
            "JSON files (*.json)")
        if not export:
            return
        videos = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select the videos folder for this export", "data/processed")
        if not videos:
            return
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(export))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(videos))

    def remove_selected(self):
        for idx in sorted(self.table.selectionModel().selectedRows(),
                          key=lambda i: i.row(), reverse=True):
            self.table.removeRow(idx.row())

    def pick_out(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output dataset folder", "data/datasets")
        if d:
            self.out_edit.setText(d)

    def _sources(self):
        return [(self.table.item(r, 0).text(), self.table.item(r, 1).text())
                for r in range(self.table.rowCount())]

    # ---- build ----
    def build(self):
        sources = self._sources()
        if not sources:
            QtWidgets.QMessageBox.warning(self, "No sources",
                                          "Add at least one source first.")
            return
        out = self.out_edit.text().strip()
        if not out:
            QtWidgets.QMessageBox.warning(self, "No output", "Pick an output folder.")
            return

        self.log.clear()
        self.build_btn.setEnabled(False)
        self.worker = BuildWorker(sources, out, self.val_spin.value(),
                                  self.seed_spin.value())
        self.worker.line.connect(self.log.appendPlainText)
        self.worker.done.connect(self.on_done)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def on_done(self, result):
        self.build_btn.setEnabled(True)
        QtWidgets.QMessageBox.information(
            self, "Done",
            f"Wrote {result['n_images']} images.\n"
            f"train videos: {result['train_videos']} | "
            f"val videos: {result['val_videos']}")

    def on_failed(self, msg):
        self.build_btn.setEnabled(True)
        QtWidgets.QMessageBox.critical(self, "Build failed", msg)


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Wizard()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
