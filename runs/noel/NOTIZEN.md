# Optimierung auf Branch `noel`: Stand, Messmethode, offene Ideen

Stand: Sa 21.11.2026, nach der vierten Runde (Commit `ba8f1f2`). Diese Datei ist das Gedächtnis für die nächste Optimierungsrunde.

## Messwerte

Vier Stände im Wechsel gemessen (je frischer Prozess), Erstlauf gesamt in ms, Median / Minimum. Alle Läufe `PASS`. Die absoluten Werte hängen vom Zustand der Maschine ab. Vergleichbar sind nur Werte aus derselben Messreihe.

| Umgebung | Philip + Max (`11b696f`) | Nach Runde 1 (`c261b63`) | Nach Runde 2 (`49d386b`) | Nach Runde 3 (`1442ef3`) |
| --- | ---: | ---: | ---: | ---: |
| Apple M4, macOS, Threads auf 1 begrenzt, 12 Läufe | 606,5 / 602,7 | 460,3 / 457,9 | 454,4 / 450,1 | **445,8 / 441,3** |
| Docker auf demselben M4, Linux aarch64, `--cpus 1`, 10 Läufe | 716,5 / 707,2 | 542,8 / 537,4 | 532,3 / 528,2 | **519,7 / 517,3** |

Das Original-Kit braucht auf diesem Mac 12'827 ms (gemessen in Runde 1). Eine Messung von `noel` auf x86-Linux fehlt weiterhin. Joel hat eine solche Maschine (siehe «Stand der anderen Branches»).

Phasen (Median, ms):

| Phase | Mac: Philip + Max | Mac: Runde 1 | Mac: Runde 2 | Mac: Runde 3 | Docker: Philip + Max | Docker: Runde 1 | Docker: Runde 2 | Docker: Runde 3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Imports | 407,1 | 314,7 | 313,5 | 312,4 | 386,2 | 308,9 | 308,4 | 309,0 |
| Build (inkl. Init, Laden) | 32,1 | 31,7 | 29,9 | 30,3 | 32,6 | 32,8 | 29,2 | 29,0 |
| Kompilation | 94,3 | 65,0 | 65,0 | 65,3 | 205,4 | 129,9 | 130,0 | 130,1 |
| Forward | 47,4 | 33,3 | 31,5 | 26,4 | 50,4 | 43,4 | 42,1 | 35,0 |
| Gradient (inkl. Sensitivity, Transfer) | 26,9 | 16,4 | 13,2 | 11,7 | 39,3 | 27,7 | 21,9 | 17,7 |
| Warm (Median) | 66,9 | 43,6 | 38,3 | 32,8 | 84,0 | 62,0 | 51,9 | 38,6 |

Runde 4 gegen Runde 3, eigene Messreihe (Mac 16 Läufe, Docker 12 Läufe, Median):

| Umgebung | Erstlauf Runde 3 (`6635856`) | Erstlauf Runde 4 (`ba8f1f2`) | Gradient-Phase | Warm |
| --- | ---: | ---: | --- | --- |
| Mac | 437,9 ms | **435,9 ms** | 10,7 → 9,8 ms | 32,4 → 31,3 ms |
| Docker `--cpus 1` | 519,3 ms | **515,0 ms** | 16,4 → 14,8 ms | 40,3 → 38,4 ms |

Die Reports `mac-neu.json`, `mac-neu.png` und `docker-neu.json` zeigen den Stand `ba8f1f2` (Erstlauf 433,5 ms bzw. 505,7 ms, warm 30,8 bzw. 41,3 ms). Der Mac-Report ist der mittlere von drei Läufen (424,1 bis 435,4 ms), der Docker-Report der mittlere von fünf (499,5 bis 514,8 ms). Der Rechner war dabei ruhiger als in den Messreihen oben. Die `*-vorher`-Reports zeigen Philip + Max.

## Was in Runde 4 geändert wurde

| Nr. | Änderung | Datei | Gewinn im Wechselvergleich |
| --- | --- | --- | --- |
| 17 | Der Rückwärtslauf speichert Ableitungen nur noch für Funktionsausgänge, die wirklich von einer Designvariable abhängen, und zwar in der Form, in der die Dual-Zahlen sie liefern. Vorher füllte `_EntryBlock.run` pro Gruppe ein dichtes Array (Zeilen × Aufrufe × 4 Ausgänge × Eingänge), grösstenteils Nullen. Ableitungen, die nicht von der Frequenz abhängen, behalten eine einzige Zeile | `fast.py`, `_EntryBlock.run` und `back` | Rückwärts-Callback einzeln auf dem Mac 8,0 → 7,2 ms. Gradient-Phase: Mac 10,7 → 9,8 ms, Docker 16,4 → 14,8 ms |

In dieser Runde gemessen und nicht übernommen:

| Versuch | Ergebnis |
| --- | --- |
| Die 22 kleinen Arrays für den Rückwärtslauf im Forward-Callback zu zwei Residuen bündeln (36 → 15 Callback-Argumente, 23 → 1 durchgereichtes Residuum) | Erstlauf: Docker 514,6 → 511,4 ms (Gradient-Phase 16,1 → 13,8 ms), Mac 438,0 → 436,6 ms, je 12 Läufe. Aber warm in Docker 38,9 → 44,8 ms. Grund: Ohne die vielen kleinen Arrays stutzt glibc den Heap nach jedem Aufruf, und jeder Aufruf fasst wieder rund 4'400 (Forward) und 3'300 (Rückwärts) frische Seiten an. Auf dem Mac bleibt warm gleich. Eine Variante, die nur die 18 Index-Arrays bündelt, verhält sich warm genauso. Zurückgenommen, Entscheid bei Noel |
| Alle Ergebnis-Arrays des Forward-Laufs in einem gemeinsamen Speicherblock anlegen | 886 → 472 Seitenfehler, aber nur 0,1 bis 0,3 ms. Nicht der Mühe wert |
| SuperLU mit anderen Werten für `PanelSize` (1, 2, 4, 8) und `Relax` (1, 2, 3, 4, 6), mit und ohne `SymmetricMode` | Keine Einstellung ist messbar schneller als `PanelSize=1`, `Relax=1` (12,96 ms im Median für Zerlegung und beide Lösungen auf dem Mac). Die beste andere liegt bei 12,83 ms, also innerhalb der Streuung. Ab `Relax=2` wird es langsamer, bis 17,8 ms |
| Idee 1 als Wegwerf-Versuch: Forward-Callback ohne die zwei dichten Matrizen | Forward-Phase im Erstlauf in Docker 32,7 → 29,2 ms (−3,5 ms), rund 3'000 Seitenfehler weniger, warm −0,8 ms. Damit ist die Schätzung für Idee 1 durch eine Messung ersetzt |

## Was in Runde 3 geändert wurde

| Nr. | Änderung | Datei | Gewinn im Wechselvergleich |
| --- | --- | --- | --- |
| 11 | Max' Commit `f1000e9` zusammengeführt (Merge `7279114`): Beispiel-Setups, `Setup.add`/`Setup.space` und der dichte JAX-Weg liegen in Modulen, die erst bei Bedarf geladen werden. `_EntryBlock.back` füllt nur die Slots der jeweiligen Gruppe, Produkte im Rückwärtslauf in-place | `setups.py`, `simulate.py`, drei neue Module, `fast.py` | `import differometor` ohne `.pyc` in Docker 15,5 → 13,0 ms (einzeln gemessen, je 15 Prozesse). Rückwärts-Callback auf dem Mac 8,9 → 8,5 ms |
| 12 | `_BlockLU`: 16 Frequenzen bilden ein Block-Diagonalsystem, also ein SuperLU-Aufruf pro Block für die Zerlegung und einer pro Lösung. Ergebnis bitgleich | `fast.py` | Zerlegung und beide Lösungen: Mac 13,4 → 12,1 ms, Docker (erster Aufruf) 18,5 → 16,8 ms |
| 13 | Die dichte Signalmatrix (9 MB) ist nicht mehr in den Residuen. Der Forward-Callback gibt dem Rückwärtslauf über `SystemPlan.base_residuals` nur die Werte mit, die er braucht | `fast.py` | Nr. 11 bis 13 und 16 zusammen: Docker Forward 42,0 → 38,1 ms, Gradient 21,3 → 19,2 ms |
| 14 | Forward- und Rückwärtslauf verarbeiten je 16 Frequenzen am Stück statt alle 101 auf einmal | `fast.py` | Docker: Forward 37,5 → 36,0 ms, Gradient 19,3 → 18,8 ms, warm 49,7 → 46,1 ms. Mac (Callbacks einzeln): Forward gleich, Rückwärts 8,5 → 9,0 ms |
| 15 | Die Komponentenfunktionen schreiben ihre Ausgänge direkt in die Matrixeinträge. Eingänge, die nicht von der Frequenz abhängen, gehen als eine Zeile hinein statt als 101 | `fast.py`, `_EntryBlock` | Callbacks einzeln, Mac: Forward 21,6 → 20,6 ms, Rückwärts 9,1 → 7,8 ms. Bitgleich |
| 16 | Suche der Nichtnull-Einträge auf der flachen Float-Sicht. Vorarbeit, die nur der Rückwärtslauf braucht, läuft nicht mehr im Forward | `fast.py` | zusammen ca. 0,5 ms pro Forward-Aufruf |

Zu Nr. 13: Die Residuen haben eine feste Grösse (16 konstante Einträge pro Unbekannte, das Benchmark-Setup hat 2). Hat eine Signalmatrix mehr, rechnet der gewöhnliche Rückwärtslauf trotzdem richtig. Nur der allgemeine Rückwärtslauf (Gewichte ausserhalb der Detektorzeilen) meldet dann einen klaren Fehler, statt falsch zu rechnen. `test_adjoint.py` prüft beides.

Zu Nr. 14: Der Gradient ändert sich in der 14. Stelle, weil die Summen über die Frequenzen anders geklammert sind. Blockgrösse 16 ist gemessen: Mit 8 ist der Gradient langsamer (Docker 19,8 statt 17,5 ms), mit 32 beide Phasen (Docker Forward 35,5 statt 34,8 ms, Gradient 18,4 statt 17,5 ms).

## Was in Runde 2 geändert wurde

| Nr. | Änderung | Datei | Gewinn im Wechselvergleich |
| --- | --- | --- | --- |
| 8 | Rückwärtslauf ohne dichte Matrizen: `SystemPlan(A, signal_base=False, noise_base=False)` baut den Plan nur aus den Index-Arrays. Die Rauschmatrix geht nicht mehr an den Rückwärts-Callback (`BACKWARD_KEYS`) | `fast.py` | Gradient: Docker 26,9 → 21,4 ms, Mac 16,2 → 13,3 ms |
| 9 | Signal-Systeme skaliert (`_balance`, `D⁻¹·S·D` mit Zweierpotenzen), symmetrisch nach `MMD_AT_PLUS_A` angeordnet, Schwellen-Pivotsuche mit `DiagPivotThresh = 0,1`. Faktoren: ca. 6'100 statt 8'330 Einträge | `fast.py`, `SystemPlan._analyze` | Forward: Docker 42,3 → 39,3 ms, Mac 32,8 → 30,6 ms |
| 10 | Dichte Matrizen im Build 64-Byte-ausgerichtet anlegen (`_aligned_zeros`), dann übernimmt `jnp.asarray` sie ohne Kopie. Einheitsdiagonale direkt setzen statt über `np.eye` | `build.py` | Build −3,4 ms, Forward +2,3 ms in Docker (netto −1,1 ms). Mac netto −0,6 ms |

Zu Nr. 9: Die Skalierung macht den Löser auch genauer. Bei zufällig veränderten Designvariablen (`test_perturbed.py`, 6 Versuche) weicht die Sensitivität um höchstens 2,1e-8 (im Logarithmus) vom Original-JAX ab. Der Stand davor lag in Versuch 4 um 8,2e-6 daneben.

## Was in Runde 1 geändert wurde

| Nr. | Änderung | Datei | Gewinn (Mac, Messung vom 20.11.) |
| --- | --- | --- | --- |
| 1 | SuperLU-Modul direkt laden statt `import scipy.sparse.linalg` | `fast.py`, `_load_superlu` | 633 → 551 ms |
| 2 | `power_detector` und `signal_detector` als Host-Callback mit eigener Rückwärtsregel, abschaltbar über `components.HOST_DETECTORS` | `components.py` | Kompilation 100 → 68 ms |
| 3 | Ableitungen nur für Funktionseingänge, die Designvariablen sind | `fast.py`, `_EntryBlock` | Rückwärtslauf 24 → 15 ms |
| 4 | SuperLU mit `Relax=1`, `PanelSize=1` (keine aufgefüllten Supernodes) | `fast.py`, `_splu` | Zerlegung 20,9 → 12,5 ms |
| 5 | Zwei Löser-Aufrufe pro Frequenz statt drei, Rauschen für alle Frequenzen auf einmal | `fast.py`, `forward` | Teil von 43 → 30 ms |
| 6 | Schnellerer Aufbau des Matrix-Musters (Schlüssel + `searchsorted` statt dichter Masken) | `fast.py`, `SystemPlan` | 3,6 → 1,8 ms je Aufruf |
| 7 | Plotly erst bei Bedarf laden (Max' Fassung) | `__init__.py` | Import `differometor` 107 → 15 ms (zusammen mit Nr. 1) |

## Wo die restliche Zeit steckt (Docker, Stand Runde 3, Gradient Runde 4)

| Phase | ms | Unser Anteil |
| --- | ---: | --- |
| Imports | 309 | 13 ms für `import differometor`, fast alles Übersetzen der Quelltexte, weil das Image keine `__pycache__`-Ordner hat: `fast.py` 4,8, `build.py` 4,4, `components.py` 2,0, `setups.py` 0,9, `simulate.py` 0,6. Mit Cache: 1,4 ms |
| Initialisierung, Laden | 3 | nichts |
| Build | 26 | ca. 9 ms (`run_build_step`). Der Rest sind die ersten `jnp.asarray`-Aufrufe in `benchmark.py` (Start des JAX-Backends, ca. 9 ms, dazu je 3,5 ms für die Listen) |
| Forward: Lowering, Kompilation | 7,5 | alles, aber fast nur JAX-Grundaufwand (31 Argumente) |
| Forward | 35 | alles. Callback 27 bis 30 ms: Zerlegung 11, Lösen 4,6, Schleife in `forward` 3,6, Einträge 2,1, Plan 1,4, Matrixwerte 1,2. Ausserhalb des Callbacks ca. 5 ms: JAX kopiert 19 MB Argumente hinein und 6 MB Ergebnisse heraus |
| Sensitivity: Lowering | 11 | ca. 2 ms (zwei Callbacks). Der Rest ist das Tracing des Readouts |
| Sensitivity: Kompilation | 66 | nichts. Das ist der Readout aus `benchmark.py` in XLA |
| Gradient: Lowering | 5 | wenig |
| Gradient: Kompilation | 36 | nichts |
| Gradient | 15 | fast alles. Callback 10,0 bis 10,7 ms: Funktionen mit Ableitung 3,5 bis 4,0, Schleife in `backward` 3,6, Rückrechnung auf die Parameter 1,3. Ausserhalb ca. 3,5 ms Übergabe (36 Argumente) |

Rund 430 von 520 ms sind fix, solange `benchmark.py` unverändert bleibt und keine JAX-Schalter gesetzt werden dürfen.

## Erkenntnisse aus Runde 4 (Messungen, Erklärungen sind als solche bezeichnet)

| Thema | Befund |
| --- | --- |
| Warum warme Aufrufe in Docker mal schnell und mal langsam sind | Gemessen: Mit einer festen Stutz-Schwelle von 19 MB fallen pro warmem Forward-Aufruf 2'900 bis 4'700 Seitenfehler an, mit einer sehr grossen Schwelle 700 bis 1'600. Erklärung: glibc passt zwei Schwellen laufend an. Die mmap-Schwelle wird so gross wie der grösste freigegebene Block (hier die 9-MB-Kopien der dichten Matrizen), die Stutz-Schwelle doppelt so gross (18 MB). Ab dem zweiten Aufruf liegen die zwei 9-MB-Kopien im Heap des Arbeits-Threads. Werden sie und unsere Zwischenarrays freigegeben, ist oben mehr als 18 MB frei, und glibc gibt den Speicher ans System zurück. Der nächste Aufruf fasst ihn wieder frisch an |
| Warum das bisher meist nicht passierte | Vermutung, passend zu den Messungen: Kleine, länger lebende Arrays (die durchgereichten Index-Arrays, der zurückgegebene Gradient) liegen zufällig über den freigegebenen Bereichen und verhindern das Stutzen. Mit weniger kleinen Arrays (Bündelungs-Versuch) trat es in 10 von 12 Läufen auf, vorher in 2 von 12. Das ist ein Zufall des Speicherlayouts, kein Verdienst des Codes |
| Feste Schwellen als Gegenprobe | Mit `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=9500000:glibc.malloc.trim_threshold=2000000000` (nur zur Messung gesetzt, zusammen mit dem Bündelungs-Versuch): Forward-Phase im Erstlauf 33,4 → 31,7 ms, Gradient-Phase 13,4 → 11,6 ms (Mittel aus zwei bis drei Läufen im selben Messskript). Der Rückwärts-Callback hat dann schon im ersten Aufruf keine Seitenfehler mehr. Rund 3,5 ms im Erstlauf, aber eine Einstellung der Umgebung, also Regelfrage B6 |
| Kosten pro Callback-Argument | Rund 11 µs pro Argument und Aufruf, dazu je rund 15 µs beim Lowering und 20 µs bei der Kompilation (Versuch mit 0, 10 und 40 Argumenten). Ein durchgereichtes Residuum kostet rund 10 µs Lowering und 20 µs Kompilation |
| Build | `run_build_step` braucht warm 6,5 bis 7,5 ms: `build` 4,8 ms (davon `set_instructions` geschätzt 2 ms in 519 Aufrufen), `pairs_to_arrays` 1,5 ms, `prepare_arrays` 0,1 ms. Die Übergabe der 38 Arrays an JAX kostet zusammen nur 0,6 ms |

## Erkenntnisse aus Runde 3 (alle gemessen)

| Thema | Befund |
| --- | --- |
| JAX-Schalter | `jax.config.update("jax_disable_most_optimizations", True)` spart in Docker rund 42 ms (Sensitivity-Kompilation 64 → 36,5 ms, Gradient-Kompilation 35,3 → 20,6 ms), auf dem Mac 13 ms (24 → 17 und 17,8 → 12,1 ms). Loss und Gradient stimmen auf 11 Stellen. Nicht eingebaut: Regelfrage B6 in `FRAGEN_AN_FEYER.md`. Joel hat dieselbe Variante vorbereitet (`sx2`, `sx2-env`) |
| Threads | Der Forward-Callback läuft in einem XLA-Arbeits-Thread, der Rückwärts-Callback und die Detektoren im Haupt-Thread. Im Arbeits-Thread fasste jeder Forward-Aufruf wieder frische Speicherseiten an, auch im warmen Zustand (rund 4'200 Seitenfehler pro Aufruf, jetzt 250 bis 1'800) |
| Seitenfehler | In Docker kostet ein Seitenfehler rund 0,6 µs. Erstlauf vorher: 16'000 bis 16'800 in der Forward-Phase und 8'100 bis 9'300 in der Gradient-Phase. Jetzt: 11'000 bis 11'650 und 6'400 bis 6'500. Was bleibt: 19 MB Argument-Kopien von JAX, die Faktoren der Zerlegung (2'900), die Ergebnis-Arrays |
| Allocator-Umgebungsvariablen | `MALLOC_MMAP_THRESHOLD_`, `MALLOC_TRIM_THRESHOLD_` und `MALLOC_TOP_PAD_` ändern im echten Ablauf nichts an den Seitenfehlern (im Minimalbeispiel im Haupt-Thread schon). Auch darum bringt nur weniger Speicher etwas |
| Docker-VM | Seitengrösse 4 kB, Transparent Huge Pages auf `always`, glibc 2.41 |
| SuperLU | Rund 110 ns pro Spalte, dazu rund 10 µs Grundaufwand pro Aufruf (aus den Messungen abgeleitet). Die Zerlegung ist mit 11 ms weiterhin der grösste Einzelposten |
| Speicher zwischen den Blöcken freigeben (`del`) | Kein Gewinn (26,7 bis 27,2 gegen 26,9 bis 27,9 ms) |
| Nachgeladene Module | Auf dem Mac (mit `.pyc`) wandern ca. 0,5 ms vom Import in den Build, weil JAX einige eigene Module jetzt erst dort lädt. Netto dort neutral |
| Streuung | Imports ±3 ms, Kompilation ±3 ms pro Messreihe. Gewinne unter 3 ms sind in der Gesamtzeit nicht sichtbar, nur in den Phasen Forward und Gradient |

## Erkenntnisse aus Runde 2 (alle gemessen)

| Thema | Befund |
| --- | --- |
| Callback-Argumente | JAX kopiert jedes Argument eines `pure_callback` in ein neues NumPy-Array. Zwei 9-MB-Matrizen kosten pro Aufruf ca. 1,4 ms in Docker und beim ersten Aufruf auch auf dem Mac. Darum zählt jede grosse Eingabe |
| `jnp.asarray` auf der CPU | Kopiert ein gewöhnliches NumPy-Array (16-Byte-Ausrichtung). Ein 64-Byte-ausgerichtetes übernimmt es ohne Kopie. Ein Aufruf kostet ca. 15 µs |
| Durchgereichte Residuen | Ein Jit-Programm, das eine Eingabe unverändert zurückgibt, kopiert sie |
| Diagonale als Pivot | Mit symmetrischer Anordnung ist reine Diagonal-Pivotierung hier stabil (Abweichung 7,8e-8). Die Notiz aus Runde 1 («Fehler 1e75») stammte aus einem Versuch, bei dem nur die Spalten umsortiert waren |
| Unsymmetrische Anordnung | Eine Markowitz-Reihenfolge in Python bringt nicht weniger Füllung (6'230 statt 6'060 Einträge) und kostet 3,8 ms |

## Messmethode

```bash
source .venv/bin/activate
# Wechselvergleich lokal: Vergleichsstand als Kopie (runs/noel/vergleich) oder Git-Worktree
python runs/noel/ab.py 12 vorher=runs/noel/vergleich neu=.
# Wechselvergleich in Docker (Linux, --cpus 1), beliebig viele Images
docker build -q -t differometor-bench:neu . && python runs/noel/ab_docker.py 10 vorher=differometor-bench:runde4 neu=differometor-bench:neu
# Profil der Host-Callbacks, lokal oder im Image
source runs/env.sh && python runs/noel/profil.py . forward
docker run --rm --cpuset-cpus 0 --entrypoint python differometor-bench:neu /app/runs/noel/profil.py /app backward
```

Einzelne Läufe streuen um bis zu 40 ms. Darum immer abwechselnd messen und Median und Minimum ansehen. Der erste Lauf eines frisch gebauten Images ist ein Ausreisser (562 statt 520 ms) und zählt nicht.

Vorhandene Docker-Images: `differometor-bench:base` (Philip + Max), `:vorher` (`c261b63`), `:kandidat3` (`49d386b`), `:runde3` (`1442ef3`), `:runde4` (`ba8f1f2`), `:joel` (`76e799f`).

## Korrektheit prüfen (nach jeder Änderung)

```bash
source .venv/bin/activate && source runs/env.sh
python runs/check_fast.py runs/noel/orig.npz --fallback
python runs/check_setups.py
python runs/noel/test_detectors.py .
python runs/noel/test_adjoint.py .      # allgemeiner Rückwärtslauf, Blockgrösse, Residuen-Grenze
python runs/noel/test_perturbed.py .    # veränderte Designvariablen gegen das Original-JAX
```

`runs/noel/orig.npz` ist nicht im Repo (`*.npz` steht in `.gitignore`). Neu erzeugen im unveränderten Kit (Branch `main`), braucht rund 14 GiB RAM:

```bash
python benchmark.py --repeats 3 --output orig.json --save-arrays orig.npz
```

Sollwerte (Stand `ba8f1f2`, auf Noels M4, unverändert seit Runde 2):

| Grösse | Abweichung gegen das Original-JAX (max, relativ zum Maximum) |
| --- | ---: |
| carrier | 9,7e-12 |
| signal | 1,161e-07 |
| noise | 3,787e-08 |
| sensitivity | 5,735e-09 |
| powers | 1,394e-11 |
| gradient | 7,985e-08 |
| Fehler gegen Feyers Referenz | 5,3743e-06 (Grenze 1e-5) |

| Prüfung | Sollwert |
| --- | --- |
| `check_setups.py` | aLIGO (schnell) höchstens 1,1e-8, aLIGO verknüpft und Voyager 0 |
| `test_detectors.py` | WORST 2,9e-14 |
| `test_adjoint.py` | WORST 4,0e-08, Blockgrössen 1 und 2 gegen 16: Abweichung 0, Meldung «fails with a clear message, as intended» |
| `test_perturbed.py` | WORST 2,1e-08 |

## Verworfen

| Idee | Grund |
| --- | --- |
| Eigene, über die Frequenzen gebündelte LU in NumPy | Etwa 15 ms reiner Speicherverkehr, also nicht schneller als SuperLU |
| Dichte Teilblöcke mit `np.linalg.solve` | Um Grössenordnungen zu teuer |
| Reine Diagonal-Pivotierung ohne Absicherung | Geschätzt 1 ms schneller, aber kein Schutz, wenn ein Pivot klein wird |
| Einfache Unbekannte vorab in NumPy herausrechnen (751 → 324 Unbekannte) | In Runde 3 nachgerechnet (Schätzung): Zerlegung und Lösen würden rund 5,6 ms sparen, die Vorab-Rechnung und das Zurückrechnen für beide Löserichtungen kosten rund 5 ms. Dafür rund 150 Zeilen neue Sparse-Algebra |
| Zweiter Rückwärts-Callback hinter `lax.cond` | Durch Nr. 13 erledigt: Die Signalmatrix wird gar nicht mehr übergeben |
| Zwischenergebnisse des Forward-Laufs im Prozess aufheben (Token) | Wäre ein Cache. Abmachung: keine Caches |
| XLA-Flags, Kompilier-Cache, `jax.config`-Schalter, `mallopt`, Allocator mit einer grossen Anforderung «vorwärmen» | Regelfrage, siehe `FRAGEN_AN_FEYER.md` (B6) |
| Rückgabe von Stellvertreter-Objekten statt Arrays, um den Readout zu umgehen | Wäre Schummeln |
| Prüfung der dichten Matrizen in XLA statt im Callback | Jede echte XLA-Operation kostet 5 bis 15 ms Kompilierzeit, mehr als die Kopie |

## Offene Ideen

| Nr. | Idee | Schätzung | Haken |
| --- | --- | --- | --- |
| 1 | Die dichte Signal- und Rauschmatrix nicht mehr durch den Forward-Callback schicken, sondern ihre Sparse-Form aus dem Build mitgeben | gemessen im Wegwerf-Versuch: 3,5 ms im Erstlauf in Docker, 0,8 ms warm | Als zusätzliche Einträge im `arrays`-Dict könnten Sparse-Form und dichte Matrix auseinanderlaufen, wenn jemand die Matrix von Hand ersetzt. Ohne dieses Risiko ginge es nur, wenn die Sparse-Form am Matrix-Objekt selbst hängt (eigener Pytree-Typ statt `jax.Array` für drei Werte im Dict, rund 120 Zeilen). Entscheid fürs Team, Noel wurde am 21.11. gefragt (Empfehlung: lassen) |
| 2 | Kleine Arrays für den Rückwärtslauf bündeln (in Runde 4 gebaut, gemessen, zurückgenommen) | gemessen: 3,2 ms im Erstlauf in Docker, 1,4 ms auf dem Mac | Warm in Docker 6 ms langsamer (siehe Runde 4). Nur sinnvoll, wenn allein der Erstlauf zählt. Entscheid bei Noel |
| 3 | `build.py` straffen (`set_instructions` 519 Aufrufe, `pairs_to_arrays`) | geschätzt 2 bis 3 ms | Das ist Upstream-Code mit vielen kleinen NumPy-Schritten pro Bauteil. Ein Umbau wäre gross und müsste für alle Setups gleich bleiben |
| 4 | Falls Feyer Einstellungen im Code erlaubt (B6): eine Zeile `jax.config.update("jax_disable_most_optimizations", True)` in `differometor/__init__.py`, dazu feste glibc-Schwellen über `mallopt` | gemessen: 42 ms in Docker und 13 ms auf dem Mac für den JAX-Schalter, rund 3,5 ms in Docker für die glibc-Schwellen | Nur nach Zusage von Feyer |
| 5 | `noel` auf Joels x86-Maschine messen (AMD EPYC, `taskset -c 14 python benchmark.py --cpus 1`) | – | Joel fragen |

## Stand der anderen Branches

Vor jeder Runde: `git fetch --all --prune`, dann alle fremden Branches auf neue Commits prüfen.

| Branch | Zuletzt gesehen | Stand in `noel` |
| --- | --- | --- |
| `origin/philip` | `ab82e42` (20.11. 19:58), unverändert am 21.11. 06:35 | vollständig enthalten |
| `origin/max/lu-once` | `79ae580` (20.11. 20:08), unverändert | vollständig enthalten |
| `origin/max/nacht` | `f1000e9` (21.11. 03:25), neu in Runde 3 | zusammengeführt in `7279114`. Übernommen: nachgeladene Module, `_EntryBlock.back`, In-place-Produkte, `signal_values` ohne Zwischenkopie. Ersetzt durch den Stand von `noel`: Spaltenordnung `MMD_ATA` (7'916 Faktoreinträge gegen 6'100) |
| `origin/Joel` | `76e799f` (21.11. 05:19), unverändert am 21.11. 06:35, baut auf Philips `ab82e42` auf | nichts zu übernehmen. Joels drei Code-Änderungen (Host-Detektoren, Plotly bei Bedarf, SuperLU ohne SciPy-Import) hat `noel` seit Runde 1 in eigener Fassung |
| `origin/sx`, `origin/sx2`, `origin/sx2-env` | `2c9e8bf`, `036194a`, `bf6bf60` (21.11. 03:33 bis 04:00), unverändert am 21.11. 06:35 | nicht übernommen. Das sind Joels vorbereitete Varianten mit XLA-Flags bzw. `jax_disable_most_optimizations`, laut seinem README erst nach Zusage von Feyer zu aktivieren. Gleiche Regelfrage wie unser B6 |
| `origin/main` | `5eece1d` | unverändertes Kit |

Joels Stand gegen `noel` auf Noels Rechner (je 10 Läufe im Wechsel, Median):

| Umgebung | `origin/Joel` (`76e799f`) | `noel` (`1442ef3`) |
| --- | ---: | ---: |
| Mac | 484,9 ms (Forward 45,8, Gradient 29,1, warm 67,2) | 442,9 ms (Forward 26,5, Gradient 11,7, warm 32,7) |
| Docker `--cpus 1` | 565,3 ms (Forward 49,2, Gradient 42,9, warm 84,4) | 516,8 ms (Forward 35,2, Gradient 17,8, warm 39,4) |

Joels README nennt für seinen Stand auf einem AMD EPYC (x86-Linux, 1 CPU) 762 ms im Erstlauf, davon Imports 280, Kompilation 230, Forward und Gradient 150, Build 60 ms. Forward und Gradient haben dort also einen viel grösseren Anteil als auf dem Mac. Der Vorsprung von `noel` dürfte dort entsprechend grösser ausfallen. Gemessen ist das nicht.

## Abmachungen

- Zu Beginn jeder Runde die Branches von Max, Philip und Joel holen und sichten. Was messbar hilft, in `noel` übernehmen (Merge oder Cherry-Pick) und wie jede Änderung messen und prüfen. Fremde Branches nie verändern.
- Noel hat am 21.11. freigegeben: Jede Runde committet ihre behaltenen Änderungen und pusht sie auf `origin/noel`.
- Nur der Branch `noel`. Kein Merge von `noel` in andere Branches, kein Pull Request, kein Force-Push, keine anderen Branches anfassen.
- Verworfene Versuche werden nicht committet.
- Eine Änderung bleibt nur, wenn sie im Wechselvergleich schneller ist und alle Prüfungen unverändert gut ausfallen.
- `benchmark.py`, Fixtures und `requirements.txt` bleiben unangetastet. Keine Caches, nichts vorberechnen.
