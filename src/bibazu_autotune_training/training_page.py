"""Select reviewed campaigns, inspect their coverage, and train one model."""

from pathlib import Path

from bibazu_autotune.ml.artifacts import FLIP_CLASSES, model_directory, read_metadata
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .gp_training import read_campaigns, train_and_save


class TrainingWorker(QThread):
    progress = pyqtSignal(str)

    def __init__(self, operation, parent):
        super().__init__(parent)
        self.operation = operation
        self.result = None
        self.error = ""

    def run(self):
        try:
            self.result = self.operation(self.progress.emit)
        except Exception as exc:
            self.error = str(exc) or type(exc).__name__


class TrainingPage(QWidget):
    def __init__(self, current_campaign, campaign_directory, parent=None):
        super().__init__(parent)
        self.current_campaign = current_campaign
        self.campaign_directory = campaign_directory
        self.worker = None
        self.dataset = None

        layout = QVBoxLayout(self)
        title = QLabel("Gaussian Process · Flipklassifikation")
        layout.addWidget(title)
        form = QFormLayout()
        self.name = QLineEdit("GP · Physics v1")
        form.addRow("Modellname", self.name)
        form.addRow("Speicherordner", QLabel(str(model_directory())))
        layout.addLayout(form)
        buttons = QHBoxLayout()
        self.add = QPushButton("Kampagnendateien auswählen …")
        self.add.clicked.connect(self.browse)
        self.add_current = QPushButton("Kampagne aus Datenprüfung hinzufügen")
        self.add_current.clicked.connect(self.add_open_campaign)
        self.remove = QPushButton("Auswahl entfernen")
        self.remove.clicked.connect(self.remove_selected)
        for button in (self.add, self.add_current, self.remove):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.campaigns = QListWidget()
        self.campaigns.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        layout.addWidget(self.campaigns, 1)
        actions = QHBoxLayout()
        self.inspect_button = QPushButton("Daten einlesen")
        self.inspect_button.clicked.connect(self.inspect)
        self.train_button = QPushButton("GP trainieren und speichern")
        self.train_button.clicked.connect(self.train)
        actions.addWidget(self.inspect_button)
        actions.addWidget(self.train_button)
        layout.addLayout(actions)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.report = QTextBrowser()
        layout.addWidget(self.report, 2)
        self.name.textChanged.connect(self.update_controls)
        self.invalidate()

    def running(self):
        return self.worker is not None and self.worker.isRunning()

    def selected_paths(self):
        return tuple(
            self.campaigns.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.campaigns.count())
        )

    def browse(self):
        start = self.selected_paths()
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Kampagnen auswählen",
            str(start[-1] if start else self.campaign_directory()),
            "Kampagnen (campaign.json)",
        )
        self.add_campaigns([Path(path).parent for path in paths])

    def add_open_campaign(self):
        path = self.current_campaign()
        if path is None:
            self.status.setText("Zuerst eine Kampagne unter „Daten prüfen“ öffnen.")
        else:
            self.add_campaigns([path])

    def add_campaigns(self, paths):
        existing = set(self.selected_paths())
        for path in paths:
            directory = Path(path).resolve()
            if directory not in existing:
                item = QListWidgetItem(str(directory))
                item.setData(Qt.ItemDataRole.UserRole, directory)
                self.campaigns.addItem(item)
                existing.add(directory)
        if paths:
            self.invalidate()

    def remove_selected(self):
        for item in self.campaigns.selectedItems():
            self.campaigns.takeItem(self.campaigns.row(item))
        self.invalidate()

    def invalidate(self):
        self.dataset = None
        self.report.clear()
        self.status.setText("Kampagnen auswählen und Daten einlesen.")
        self.update_controls()

    def update_controls(self):
        busy = self.worker is not None
        for widget in (self.add, self.add_current, self.remove, self.campaigns, self.name):
            widget.setEnabled(not busy)
        self.inspect_button.setEnabled(not busy and self.campaigns.count() > 0)
        ready = False
        if self.dataset is not None:
            try:
                self.dataset.validate()
                ready = True
            except ValueError:
                pass
        self.train_button.setEnabled(not busy and ready and bool(self.name.text().strip()))

    def start_worker(self, operation, on_result):
        if self.worker is not None:
            return
        worker = TrainingWorker(operation, self)
        self.worker = worker
        worker.progress.connect(self.status.setText)
        worker.finished.connect(lambda: self.finish_worker(worker, on_result))
        self.update_controls()
        worker.start()

    def finish_worker(self, worker, on_result):
        self.worker = None
        if worker.error:
            self.status.setText("Fehler: " + worker.error)
        else:
            on_result(worker.result)
        self.update_controls()
        worker.deleteLater()

    def inspect(self):
        paths = self.selected_paths()
        self.dataset = None
        self.report.clear()
        self.status.setText("Lese Kampagnen und berechne Merkmale …")
        self.start_worker(lambda _progress: read_campaigns(paths), self.show_dataset)

    def show_dataset(self, data):
        self.dataset = data
        counts = data.counts
        lines = [
            f"{len(data.labels)} verwendbare Versuche · {len(data.by_part)} Bauteile · "
            f"{len(data.campaigns)} Kampagnen",
            " · ".join(f"{label}: {counts[label]}" for label in FLIP_CLASSES),
        ]
        for part, info in data.by_part.items():
            summary = " · ".join(f"{label}: {info['counts'][label]}" for label in FLIP_CLASSES)
            lines.append(f"{info['name']} ({part[:8]}) · {summary}")
        if data.excluded:
            lines.extend(("", "Nicht verwendet:"))
            lines.extend(f"{count} × {reason}" for reason, count in data.excluded.items())
        self.report.setPlainText("\n".join(lines))
        try:
            data.validate()
        except ValueError as exc:
            self.status.setText(str(exc))
        else:
            missing = [label for label in FLIP_CLASSES if not counts[label]]
            if missing:
                self.status.setText(
                    "Zwei-Klassen-Training möglich. Keine Schätzung für " + ", ".join(missing) + "."
                )
            else:
                self.status.setText("Trainingsdaten bereit.")

    def train(self):
        dataset, name = self.dataset, self.name.text().strip()
        self.status.setText("Trainingslauf startet …")

        def fit(progress):
            return train_and_save(dataset, name, progress=progress)

        self.start_worker(fit, self.show_model)

    def show_model(self, path):
        try:
            metadata = read_metadata(path)
            validation = metadata["validation"]
        except (OSError, ValueError, KeyError) as exc:
            self.status.setText(f"Modell gespeichert in {path}. Prüfbericht nicht lesbar: {exc}")
            return
        if validation["status"] == "validated":
            self.status.setText(
                f"Modell gespeichert · {validation['folds']} Gruppenfalten · "
                f"Log Loss {validation['log_loss']:.3f}"
            )
            lines = [
                "Klassenreihenfolge: " + ", ".join(validation["class_order"]),
                "Konfusionsmatrix (Zeilen: beobachtet, Spalten: vorhergesagt):",
                *(str(row) for row in validation["confusion_matrix"]),
            ]
            for part, result in validation["by_part"].items():
                name = self.dataset.by_part[part]["name"]
                lines.append(f"{name} ({part[:8]}) · Log Loss {result['log_loss']:.3f}")
        else:
            self.status.setText("Modell gespeichert · An neuen Bauteilen ungeprüft")
            lines = ["Bauteilprüfung benötigt alle modellierten Klassen in jeder Trainingsfalte."]
        missing = [label for label in FLIP_CLASSES if label not in metadata["classes"]]
        if missing:
            lines.insert(0, "Keine Schätzung für " + ", ".join(missing) + ".")
        self.report.setPlainText(
            self.report.toPlainText() + "\n\n" + str(path) + "\n" + "\n".join(lines)
        )
