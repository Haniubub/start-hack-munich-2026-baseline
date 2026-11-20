# Optimierung auf Branch `noel`: Stand, Messmethode, offene Ideen

Stand: Sa 21.11.2026. Diese Datei ist das Gedächtnis für die nächste Optimierungsrunde.

## Messwerte

Abwechselnd gemessen (alter und neuer Stand im Wechsel, je frischer Prozess), Erstlauf gesamt in ms, Median / Minimum. Alle Läufe `PASS`.

| Umgebung | Original | Philip + Max (Commit `11b696f`) | Dieser Stand |
| --- | ---: | ---: | ---: |
| Apple M4, macOS, Threads auf 1 begrenzt, 12 Läufe | 12'827 | 672,7 / 619,3 | **507,7 / 480,5** |
| Docker auf demselben M4, Linux aarch64, `--cpus 1`, 10 Läufe | nicht gemessen | 734,1 / 725,5 | **560,0 / 553,8** |

Phasen (Median, ms):

| Phase | Mac vorher | Mac jetzt | Docker vorher | Docker jetzt |
| --- | ---: | ---: | ---: | ---: |
| Imports | 450,1 | 336,9 | 395,4 | 319,0 |
| Build (inkl. Init, Laden) | 37,1 | 36,9 | 35,2 | 35,4 |
| Kompilation | 106,1 | 74,5 | 210,9 | 133,6 |
| Forward | 51,5 | 37,3 | 51,9 | 44,1 |
| Gradient (inkl. Sensitivity, Transfer) | 29,4 | 18,7 | 41,3 | 28,9 |
| Warm (Median) | 75,1 | 47,2 | 88,1 | 64,3 |

Eine Messung auf x86-Linux fehlt weiterhin.

## Was in dieser Runde geändert wurde

| Nr. | Änderung | Datei | Gewinn (Mac) |
| --- | --- | --- | --- |
| 1 | SuperLU-Modul direkt laden statt `import scipy.sparse.linalg` | `fast.py`, `_load_superlu` | 633 → 551 ms |
| 2 | `power_detector` und `signal_detector` als Host-Callback mit eigener Rückwärtsregel, abschaltbar über `components.HOST_DETECTORS` | `components.py` | Kompilation 100 → 68 ms |
| 3 | Ableitungen nur für Funktionseingänge, die Designvariablen sind | `fast.py`, `_EntryBlock` | Rückwärtslauf 24 → 15 ms |
| 4 | SuperLU mit `Relax=1`, `PanelSize=1` (keine aufgefüllten Supernodes) | `fast.py`, `_splu` | Zerlegung 20,9 → 12,5 ms |
| 5 | Zwei Löser-Aufrufe pro Frequenz statt drei, Rauschen für alle Frequenzen auf einmal | `fast.py`, `forward` | Teil von 43 → 30 ms |
| 6 | Schnellerer Aufbau des Matrix-Musters (Schlüssel + `searchsorted` statt dichter Masken) | `fast.py`, `SystemPlan` | 3,6 → 1,8 ms je Aufruf |
| 7 | Plotly erst bei Bedarf laden | `__init__.py` | Import `differometor` 107 → 15 ms (zusammen mit Nr. 1) |

## Wo die restliche Zeit steckt (Mac, 508 ms)

| Block | ms | Davon in unserer Hand |
| --- | ---: | --- |
| Imports | 337 | ca. 15 (`differometor`), der Rest ist `import jax` (ca. 205) und `importlib.metadata` in `benchmark.py` (ca. 85) |
| Build | 37 | ca. 10 (`run_build_step`), der Rest sind `jnp.asarray`-Aufrufe in `benchmark.py` |
| Kompilation | 75 | ca. 10 (unsere Callbacks), der Rest ist der Readout in `benchmark.py` |
| Forward | 37 | alles: SuperLU 14, Lösen 6, Einträge 3, Muster 2, Rest Übergabe |
| Gradient | 19 | alles |

Rund 400 ms sind fix, solange `benchmark.py` unverändert bleibt.

## Messmethode

```bash
source .venv/bin/activate
# Wechselvergleich lokal: Vergleichsstand als Kopie oder Git-Worktree in einem zweiten Ordner
python runs/noel/ab.py 12 vorher=<ordner_vorher> neu=.
# Wechselvergleich in Docker (Linux, --cpus 1)
docker build -q -t differometor-bench:neu . && python runs/noel/ab_docker.py 10 vorher=differometor-bench:base neu=differometor-bench:neu
# Profil der Host-Callbacks
source runs/env.sh && python runs/noel/profil.py . forward
```

Einzelne Läufe streuen um bis zu 40 ms. Darum immer abwechselnd messen und Median und Minimum ansehen.

## Korrektheit prüfen (nach jeder Änderung)

```bash
source .venv/bin/activate && source runs/env.sh
python runs/check_fast.py runs/noel/orig.npz --fallback
python runs/check_setups.py
python runs/noel/test_detectors.py .
```

`runs/noel/orig.npz` ist nicht im Repo (`*.npz` steht in `.gitignore`). Neu erzeugen im unveränderten Kit (Branch `main`), braucht rund 14 GiB RAM:

```bash
python benchmark.py --repeats 3 --output orig.json --save-arrays orig.npz
```

Sollwerte gegen das Original-JAX (auf Noels M4 erzeugt):

| Grösse | Abweichung (max, relativ zum Maximum) |
| --- | ---: |
| carrier | 9,7e-12 |
| signal | 1,2e-07 |
| noise | 3,8e-08 |
| sensitivity | 5,8e-09 |
| gradient | 8,0e-08 |
| Fehler gegen Feyers Referenz | 5,3745e-06 (Grenze 1e-5) |

Das entspricht dem Stand von Philip. Andere Setups (aLIGO schnell, aLIGO verknüpft, Voyager) weichen um höchstens 1e-8 ab.

## Verworfen

| Idee | Grund |
| --- | --- |
| Eigene Sparse-LU ohne Zeilentausch | Numerisch instabil: Mit erzwungener Diagonal-Pivotierung lag der Fehler bei 1e75. Die Matrix hat Einträge von 1e-11 bis 1,5e13 |
| Eigene, über die Frequenzen gebündelte LU in NumPy | Etwa 15 ms reiner Speicherverkehr, also nicht schneller als SuperLU, dazu das Stabilitätsproblem |
| Dichte Teilblöcke mit `np.linalg.solve` | Um Grössenordnungen zu teuer |
| XLA-Flags, Kompilier-Cache | Regelfrage, siehe `FRAGEN_AN_FEYER.md` |
| Rückgabe von Stellvertreter-Objekten statt Arrays, um den Readout zu umgehen | Wäre Schummeln |

## Offene Ideen

| Nr. | Idee | Schätzung | Haken |
| --- | --- | --- | --- |
| 1 | Dichte Basismatrizen (2 × 9 MB) nicht mehr durch die Callbacks schicken, sondern ihre Sparse-Form im Build erzeugen | 5–8 ms | Sparse-Form und Matrizen können auseinanderlaufen, wenn jemand die Arrays von Hand ändert |
| 2 | `build.py` straffen (`set_instructions`, `pairs_to_arrays`) | ca. 3 ms | Viel Code für wenig Gewinn |
| 3 | Triviale Unbekannte vor der Zerlegung herausnehmen (95 Zeilen und 22 Spalten haben nur den Diagonaleintrag) | ca. 2 ms | Aufwendig, betrifft alle drei Lösungen |
| 4 | Unter Linux ist die Kompilation fast doppelt so teuer wie auf dem Mac (134 statt 75 ms). Ursache suchen | offen | Vermutlich der Readout selbst |
| 5 | Messung auf x86-Linux | – | Braucht eine passende Maschine |

## Abmachungen

- Noel hat am 21.11. freigegeben: Jede Runde committet ihre behaltenen Änderungen und pusht sie auf `origin/noel`.
- Nur der Branch `noel`. Kein Merge, kein Pull Request, kein Force-Push, keine anderen Branches anfassen.
- Verworfene Versuche werden nicht committet.
- Eine Änderung bleibt nur, wenn sie im Wechselvergleich schneller ist und alle Prüfungen unverändert gut ausfallen.
