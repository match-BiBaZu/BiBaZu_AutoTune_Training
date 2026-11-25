# BiBaZu AutoTune Training

Eigenständige Desktop-Anwendung für die experimentelle Datenerfassung,
Datenprüfung und Training eines Gaussian Process für zwei oder drei Flipklassen.
Der Workflow und seine Tests
liegen in diesem Repository. AutoTune bleibt die gemeinsame Python-Abhängigkeit
für analytisches Modell, Geometrie, Poseauswahl und SPS-Dienst.

## Installation und Start

Python 3.11+ oder uv installieren. Die beiden Repositories liegen nebeneinander:

```text
Niklas_ws/
  BiBaZu_AutoTune/
  BiBaZu_AutoTune_Training/
  USBaZu_Baumer_Kamera_image_processing/
```

In `BiBaZu_AutoTune_Training`:

```powershell
./Setup.ps1
./Start.cmd
# Simulation ohne SPS und Kamera:
./Start.cmd --demo
```

`Start-Training.cmd` ist ein alternativer Name für denselben Start.
`Setup.ps1` erstellt die eigene `.venv` und installiert AutoTune als lokale
editierbare Abhängigkeit. Modelländerungen aus dem benachbarten AutoTune-Repo
werden damit direkt verwendet. Die Kopplung steht ausdrücklich in
`pyproject.toml`; es gibt keine Kopien des SPS- oder analytischen Modellcodes.
Ohne uv installiert Setup beide Pakete mit pip. Für eine Installation außerhalb
dieses Workspaces muss das passende AutoTune-Paket zuvor verfügbar sein.

## SPS, Kamera und bestehende Daten

Die Trainingsanwendung startet standardmäßig selbst den gemeinsamen fensterlosen
SPS-Dienst und verwendet dessen lokale HTTP-API. AutoTune und Training benutzen
dieselben Sitzungen und dieselbe Windows-Prozesssperre `Local\BiBaZuPlcControl`.
AutoTune, Drucktest und Pressure Control vor einer Messkampagne schließen.
Die Kameraverarbeitung bleibt in ihrer eigenen GUI und wird über HTTP verbunden.
Der direkte Videopfad im reservierten Versuchsordner bleibt unverändert.

```powershell
./Start.ps1 --vision-url http://127.0.0.1:8766
./Start.ps1 --pressure-url http://127.0.0.1:8765
./Start.ps1 --output-dir C:/Daten/Versuche
```

Neue Kampagnen werden standardmäßig unter `data/experiments` dieses Repos
gespeichert. Beim Verschieben des Programms werden keine Kampagnen, Videos oder
Prüfentscheidungen verschoben oder konvertiert. Bisherige Kampagnen bleiben unter
`BiBaZu_AutoTune/data/experiments` und lassen sich über **Kampagne öffnen** bzw.
**Kampagne wiederaufnehmen** weiterhin verwenden. Standardordner für Roadmap/STL
und gemeinsame Modellvorgaben bleiben in ihren bisherigen Benutzereinstellungen.

Der vollständige [Workflow](docs/experiments.md) beschreibt Aufzeichnung,
Klassifizierung, Datenprüfung und Modelltraining. Im Reiter **Modelltraining**
lassen sich Kampagnen mehrerer Bauteile zu einem Trainingsdatensatz kombinieren.
Gespeicherte Modelle stehen in AutoTune unter **ML-Empfehlung …** zur Auswahl.

## Tests

```powershell
./.venv/Scripts/python.exe -m pytest -q
./.venv/Scripts/python.exe -m ruff check src tests
```

Tests verwenden Simulationen und steuern keine reale Anlage an. Ein realer
Anlagen-Probelauf nach der Umstellung prüft SPS-Verbindung, Kamera, eine kleine
Versuchsreihe und das Wiederaufnehmen einer bisherigen Kampagne.
