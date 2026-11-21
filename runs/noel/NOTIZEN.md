# Optimierung auf Branch `noel`: Stand, Messmethode, offene Ideen

Stand: Sa 21.11.2026, nach der zweiten Runde (Commit `49d386b`). Diese Datei ist das Gedächtnis für die nächste Optimierungsrunde.

## Messwerte

Drei Stände im Wechsel gemessen (je frischer Prozess), Erstlauf gesamt in ms, Median / Minimum. Alle Läufe `PASS`. Die absoluten Werte hängen vom Zustand der Maschine ab: Am 21.11. gegen 03:00 liegen sie rund 10 % tiefer als in der ersten Runde kurz nach Mitternacht. Vergleichbar sind nur Werte aus derselben Messreihe.

| Umgebung | Philip + Max (`11b696f`) | Nach Runde 1 (`c261b63`) | Nach Runde 2 (`49d386b`) |
| --- | ---: | ---: | ---: |
| Apple M4, macOS, Threads auf 1 begrenzt, 12 Läufe | 601,5 / 588,2 | 459,2 / 446,5 | **453,2 / 442,3** |
| Docker auf demselben M4, Linux aarch64, `--cpus 1`, 10 Läufe | 710,8 / 699,4 | 539,4 / 534,6 | **528,7 / 524,0** |

Das Original-Kit braucht auf diesem Mac 12'827 ms (gemessen in Runde 1). Eine Messung auf x86-Linux fehlt weiterhin.

Phasen (Median, ms):

| Phase | Mac: Philip + Max | Mac: Runde 1 | Mac: Runde 2 | Docker: Philip + Max | Docker: Runde 1 | Docker: Runde 2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Imports | 404,0 | 313,4 | 314,1 | 384,8 | 309,3 | 307,4 |
| Build (inkl. Init, Laden) | 31,6 | 31,7 | 30,1 | 32,0 | 32,4 | 28,7 |
| Kompilation | 91,1 | 64,2 | 63,4 | 205,7 | 127,7 | 128,5 |
| Forward | 47,4 | 33,1 | 31,6 | 50,1 | 43,0 | 41,7 |
| Gradient (inkl. Sensitivity, Transfer) | 26,6 | 16,2 | 13,2 | 38,5 | 26,9 | 21,1 |
| Warm (Median) | 66,5 | 43,0 | 38,1 | 82,4 | 59,5 | 51,6 |

Die Reports `mac-neu.json`, `mac-neu.png` und `docker-neu.json` zeigen den Stand `49d386b` (Erstlauf 440,5 ms bzw. 519,5 ms). Die `*-vorher`-Reports zeigen Philip + Max.

## Was in Runde 2 geändert wurde

| Nr. | Änderung | Datei | Gewinn im Wechselvergleich |
| --- | --- | --- | --- |
| 8 | Rückwärtslauf ohne dichte Matrizen: `SystemPlan(A, signal_base=False, noise_base=False)` baut den Plan nur aus den Index-Arrays. Die Rauschmatrix geht nicht mehr an den Rückwärts-Callback (`BACKWARD_KEYS`) | `fast.py` | Gradient: Docker 26,9 → 21,4 ms, Mac 16,2 → 13,3 ms |
| 9 | Signal-Systeme skaliert (`_balance`, `D⁻¹·S·D` mit Zweierpotenzen), symmetrisch nach `MMD_AT_PLUS_A` angeordnet, Schwellen-Pivotsuche mit `DiagPivotThresh = 0,1`. Faktoren: ca. 6'100 statt 8'330 Einträge | `fast.py`, `SystemPlan._analyze` | Forward: Docker 42,3 → 39,3 ms, Mac 32,8 → 30,6 ms |
| 10 | Dichte Matrizen im Build 64-Byte-ausgerichtet anlegen (`_aligned_zeros`), dann übernimmt `jnp.asarray` sie ohne Kopie. Einheitsdiagonale direkt setzen statt über `np.eye` | `build.py` | Build −3,4 ms, Forward +2,3 ms in Docker (netto −1,1 ms). Mac netto −0,6 ms |

Zu Nr. 9: Die Skalierung macht den Löser auch genauer. Bei zufällig veränderten Designvariablen (`test_perturbed.py`, 6 Versuche) weicht die Sensitivität jetzt um höchstens 2,1e-8 (im Logarithmus) vom Original-JAX ab. Der Stand davor lag in Versuch 4 um 8,2e-6 daneben.

Zu Nr. 10: Der Forward-Lauf wird langsamer, weil er die bisher unberührten Speicherseiten der Matrizen zum ersten Mal liest. Im Build fallen dafür drei Kopien von je 9 MB weg.

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

## Wo die restliche Zeit steckt (Docker, Einzelphasen)

Gemessen zu Beginn von Runde 2 (5 frische Läufe, Median). Die Spalte «Unser Anteil» ist das, was Code in `differometor/` beeinflusst.

| Phase | ms | Unser Anteil |
| --- | ---: | --- |
| Imports | 297 | 14 ms für `import differometor`, davon 12,5 ms Übersetzen der Quelltexte, weil das Image keine `__pycache__`-Ordner hat. Mit Cache: 1,4 ms |
| Initialisierung, Laden | 3 | nichts |
| Build | 29 | ca. 9 ms (`run_build_step`). Der Rest sind die ersten `jnp.asarray`-Aufrufe in `benchmark.py` |
| Forward: Lowering, Kompilation | 7 | alles, aber fast nur JAX-Grundaufwand (31 Argumente) |
| Forward | 42 → 39 | alles. Jetzt: Zerlegung ca. 10, Lösen 6, Einträge 2,4, Muster 1,6, Übergabe ca. 4 |
| Sensitivity: Lowering | 11 | ca. 2 ms (zwei Callbacks). Der Rest ist das Tracing des Readouts |
| Sensitivity: Kompilation | 65 | nichts. Das ist der Readout aus `benchmark.py` in XLA: erster echter LLVM-Lauf (ca. 40 ms Grundaufwand), `abs` ca. 12 ms, Indexzugriff ca. 6 ms, `log10` ca. 5 ms |
| Gradient: Lowering | 5 | wenig |
| Gradient: Kompilation | 34 | nichts. Davon ca. 17 ms für das Scatter aus dem Indexzugriff im Readout |
| Gradient | 24 → 21 | fast alles: Callback ca. 13 ms, Übergabe ca. 4 ms |

Rund 430 von 529 ms sind fix, solange `benchmark.py` unverändert bleibt. Mit JAX-Detektoren statt Host-Callbacks wären es in Docker 70 ms mehr (Sensitivity-Kompilation 92 statt 62 ms, Gradient-Kompilation 73 statt 34 ms).

## Erkenntnisse aus Runde 2 (alle gemessen)

| Thema | Befund |
| --- | --- |
| Callback-Argumente | JAX kopiert jedes Argument eines `pure_callback` in ein neues NumPy-Array. Zwei 9-MB-Matrizen kosten pro Aufruf ca. 1,4 ms in Docker und beim ersten Aufruf auch auf dem Mac. Darum zählt jede grosse Eingabe |
| Speicherseiten | In Docker entfielen im Erstlauf ca. 13'400 Seitenfehler auf die Forward-Phase und 12'000 auf die Gradient-Phase, bei 0,5 bis 1 µs pro Fehler. Frischer Speicher kostet dort 0,03 bis 0,06 ms pro MB |
| `jnp.asarray` auf der CPU | Kopiert ein gewöhnliches NumPy-Array (16-Byte-Ausrichtung). Ein 64-Byte-ausgerichtetes übernimmt es ohne Kopie |
| Durchgereichte Residuen | Ein Jit-Programm, das eine Eingabe unverändert zurückgibt, kopiert sie (9 MB: 0,13 ms auf dem Mac). Das ist klein gegen die Callback-Kopien |
| Erster Aufruf | Der erste Forward-Callback ist 2 bis 5 ms langsamer als spätere. Vorab Speicher anfassen hilft nicht. Es ist kalter Code, kein Speichereffekt |
| Diagonale als Pivot | Mit symmetrischer Anordnung ist reine Diagonal-Pivotierung hier stabil (Abweichung 7,8e-8 wie zwischen zwei Varianten mit Zeilentausch). Die Notiz aus Runde 1 («Fehler 1e75») stammte aus einem Versuch, bei dem nur die Spalten umsortiert waren. Dann ist die «Diagonale» ein beliebiger Eintrag |
| Unsymmetrische Anordnung | Eine Markowitz-Reihenfolge in Python bringt nicht weniger Füllung (6'230 statt 6'060 Einträge) und kostet 3,8 ms |
| Gegenprobe im Ablauf | Der Rückwärts-Callback läuft im echten Ablauf in Docker rund 40 % langsamer als isoliert. Die CPU-Zeit des Threads entspricht der Wandzeit, es ist also kein Gedränge um die eine CPU |

## Messmethode

```bash
source .venv/bin/activate
# Wechselvergleich lokal: Vergleichsstand als Kopie (runs/noel/vergleich) oder Git-Worktree
python runs/noel/ab.py 12 vorher=runs/noel/vergleich neu=.
# Wechselvergleich in Docker (Linux, --cpus 1), beliebig viele Images
docker build -q -t differometor-bench:neu . && python runs/noel/ab_docker.py 10 vorher=differometor-bench:kandidat3 neu=differometor-bench:neu
# Profil der Host-Callbacks, lokal oder im Image
source runs/env.sh && python runs/noel/profil.py . forward
docker run --rm --cpuset-cpus 0 --entrypoint python differometor-bench:neu /app/runs/noel/profil.py /app backward
```

Einzelne Läufe streuen um bis zu 40 ms. Darum immer abwechselnd messen und Median und Minimum ansehen. Kleine Gewinne zeigen sich in den Phasen Forward und Gradient, die nur um etwa 0,5 ms streuen, nicht in der Gesamtzeit.

Vorhandene Docker-Images: `differometor-bench:base` (Philip + Max), `:vorher` (`c261b63`), `:kandidat` (`6117a37`), `:kandidat2` (`b658976`), `:kandidat3` (`49d386b`).

## Korrektheit prüfen (nach jeder Änderung)

```bash
source .venv/bin/activate && source runs/env.sh
python runs/check_fast.py runs/noel/orig.npz --fallback
python runs/check_setups.py
python runs/noel/test_detectors.py .
python runs/noel/test_adjoint.py .      # allgemeiner Rückwärtslauf, Gewichte auf allen Ausgängen
python runs/noel/test_perturbed.py .    # veränderte Designvariablen gegen das Original-JAX
```

`runs/noel/orig.npz` ist nicht im Repo (`*.npz` steht in `.gitignore`). Neu erzeugen im unveränderten Kit (Branch `main`), braucht rund 14 GiB RAM:

```bash
python benchmark.py --repeats 3 --output orig.json --save-arrays orig.npz
```

Sollwerte (Stand `49d386b`, auf Noels M4):

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
| `test_adjoint.py` | WORST 4,0e-08 |
| `test_perturbed.py` | WORST 2,1e-08 |

## Verworfen

| Idee | Grund |
| --- | --- |
| Eigene, über die Frequenzen gebündelte LU in NumPy | Etwa 15 ms reiner Speicherverkehr, also nicht schneller als SuperLU |
| Dichte Teilblöcke mit `np.linalg.solve` | Um Grössenordnungen zu teuer |
| Reine Diagonal-Pivotierung ohne Absicherung | Geschätzt 1 ms schneller als Nr. 9 (keine Skalierung nötig), aber kein Schutz, wenn ein Pivot klein wird |
| Selten gebrauchten Quelltext erst bei Bedarf laden (Beispiel-Setups, `simulate_jax`) | Geschätzt höchstens 2,5 ms und nur ohne `__pycache__`, dafür ein Umbau der Modulstruktur |
| Zweiter Rückwärts-Callback hinter `lax.cond`, damit die Signalmatrix nur im Sonderfall übergeben wird | Spart ca. 0,5 ms und kostet geschätzt etwa gleich viel beim Tracing. Nicht gebaut |
| Zwischenergebnisse des Forward-Laufs im Prozess aufheben (Token) | Wäre ein Cache. Abmachung: keine Caches |
| XLA-Flags, Kompilier-Cache, `jax.config`-Schalter, `mallopt` | Regelfrage, siehe `FRAGEN_AN_FEYER.md` |
| Rückgabe von Stellvertreter-Objekten statt Arrays, um den Readout zu umgehen | Wäre Schummeln |

## Offene Ideen

| Nr. | Idee | Schätzung | Haken |
| --- | --- | --- | --- |
| 1 | Sparse-Form der Signal- und Rauschmatrix im Build erzeugen und als zusätzliche Argumente an `simulate` geben. Dann gehen die dichten Matrizen nie mehr durch einen Callback | 4–5 ms in Docker, 2–3 ms auf dem Mac | Sparse-Form und dichte Matrix können auseinanderlaufen, wenn jemand die Arrays von Hand ändert. Prüfen lässt sich das nur, indem man die dichte Matrix doch liest. Entscheid fürs Team |
| 2 | Einfache Unbekannte vorab in NumPy herausrechnen (Einheitsdiagonale, ein einziger Nebeneintrag in Zeile oder Spalte). Zwei Runden verkleinern das System von 751 auf 324 Unbekannte | geschätzt 2–3 ms zusätzlich zu Nr. 9 (Versuch mit der alten Zerlegung: 11,9 → 6,6 ms, Lösen 5,0 → 3,1 ms, dazu 3–4 ms für die Vorab-Rechnung) | Rund 150 Zeilen neue Sparse-Algebra für beide Löserichtungen |
| 3 | `build.py` straffen (`set_instructions` 519 Aufrufe, `pairs_to_arrays`) | ca. 3 ms | Viel Code für wenig Gewinn |
| 4 | Dual-Zahlen im Rückwärtslauf beschleunigen (`entries` 4 bis 6 ms, 104 Multiplikationen in Python) | 1–2 ms | Unklar, ob ohne handgeschriebene Ableitungen möglich |
| 5 | Messung auf x86-Linux | – | Braucht eine passende Maschine |

## Stand der anderen Branches

Vor jeder Runde: `git fetch --all --prune`, dann alle fremden Branches auf neue Commits prüfen.

| Branch | Zuletzt gesehen | Stand in `noel` |
| --- | --- | --- |
| `origin/philip` | `ab82e42` (20.11. 19:58), unverändert am 21.11. 03:25 | vollständig enthalten |
| `origin/max/lu-once` | `79ae580` (20.11. 20:08), unverändert am 21.11. 03:25 | vollständig enthalten |
| `origin/Joel` | `5eece1d`, neu seit 21.11. zwischen 02:37 und 03:25. Zeigt auf denselben Commit wie `main`, noch keine eigenen Commits | nichts zu übernehmen |
| `origin/main` | `5eece1d` | unverändertes Kit |

## Abmachungen

- Zu Beginn jeder Runde die Branches von Max, Philip und Joel holen und sichten. Was messbar hilft, in `noel` übernehmen (Merge oder Cherry-Pick) und wie jede Änderung messen und prüfen. Fremde Branches nie verändern.
- Noel hat am 21.11. freigegeben: Jede Runde committet ihre behaltenen Änderungen und pusht sie auf `origin/noel`.
- Nur der Branch `noel`. Kein Merge, kein Pull Request, kein Force-Push, keine anderen Branches anfassen.
- Verworfene Versuche werden nicht committet.
- Eine Änderung bleibt nur, wenn sie im Wechselvergleich schneller ist und alle Prüfungen unverändert gut ausfallen.
- `benchmark.py`, Fixtures und `requirements.txt` bleiben unangetastet. Keine Caches, nichts vorberechnen.
