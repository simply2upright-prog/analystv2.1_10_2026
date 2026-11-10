# DivFlow-Signale (GitHub-Seite)

Diese drei Dateien kommen in dein Repo `analystv2.1_10_2026`:

| Datei hier | Ziel im Repo |
|---|---|
| `divflow_signals.py` | Hauptverzeichnis (neben `engine.py`) |
| `divflow_signals.yml` | `.github/workflows/divflow_signals.yml` |
| optional `divflow_extra.txt` | Hauptverzeichnis, ein Yahoo-Ticker pro Zeile für Titel, die nicht in `ticker_meta.json` stehen (z. B. deine Depot-Titel) |

1. Dateien hochladen (Add file → Upload files, die yml mit dem Pfad `.github/workflows/`).
2. Actions → "DivFlow Signals" → Run workflow. Dauer ca. 10 bis 20 Minuten für rund 1000 Ticker.
3. Es entsteht `divflow_signals.json.gz` im Hauptverzeichnis. Danach läuft die Action werktags um 22:30 UTC von selbst.
4. In DivFlow: Aktien-Screener → "Signale aus GitHub laden". Per Cron: `15 23 * * 1-5 php .../cron/tech_import.php`.

Das Skript nutzt `engine.py` (Indikatoren, Kriterien, Klassifikation) und fragt Yahoo auf dem GitHub-Runner ab. Test lokal: `python divflow_signals.py --limit 30`.

## Kerzen-Speicher
`divflow_bars.json.gz` speichert je Ticker die letzten 8 Kerzen. Fehlt Yahoo später die neueste Kerze (Wochenende, Europa), wird die gespeicherte wieder angehängt. Zusätzlicher Lauf werktags 17:55 UTC (nach dem Europa-Mail-Scan).
