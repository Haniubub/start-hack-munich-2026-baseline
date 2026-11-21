# Fehlversuche: was nichts gebracht oder geschadet hat

Branch `philip-nacht`, Nachtlauf vom 21.11.2026. Hier steht alles, was ausprobiert und **nicht** behalten wurde, damit es niemand ein zweites Mal versucht. Was behalten wurde, steht in [`NOTIZEN.md`](NOTIZEN.md).

Messumgebung für alle Zahlen: x86-64 Linux, Intel Xeon 2,1 GHz (Emerald Rapids, die Nachfolge-Generation des Xeon Platinum 8468), `python benchmark.py --cpus 1 --repeats 3`, ohne `XLA_FLAGS`. Verglichen wird immer abwechselnd in frischen Prozessen. Die Maschine streut zwischen zwei Messreihen um bis zu 50 ms im Gesamtwert, innerhalb einer Reihe um etwa 10 ms. Einzelne Phasen sind deutlich stabiler als der Gesamtwert.

## Übersicht

| Nr. | Versuch | Ergebnis | Urteil |
| --- | --- | --- | --- |
| 1 | Andere SuperLU-Einstellungen (`Relax`, `PanelSize`, `DiagPivotThresh`, ohne `SymmetricMode`) | 44–62 ms statt 42 ms für 101 Zerlegungen | schlechter |
| 2 | Spaltenordnung `MMD_AT_PLUS_A` statt COLAMD | 102 ms statt 43 ms, Fill 18 724 statt 8 249 | viel schlechter |
| 3 | Vorab-Elimination nur mit NumPy (`np.add.reduceat`) | Verwaltung kostet 36 ms, spart 41 ms | fast nichts gebracht |
| 4 | Dasselbe mit umgedrehtem Speicherlayout `(Unbekannte, F, k)` | Verwaltung noch 30 ms | zu wenig |
| 5 | Speicherverwaltung von glibc umstellen (`mallopt`) | Gesamt 1010,6 → 1002,7 ms (Median), Minimum gleich, Build +7 ms | nichts gebracht |
| 6 | Grössere Supernodes für SuperLU auf dem Restsystem (200 Unbekannte) | 17,7–23,2 ms statt 16,9 ms | schlechter |
| 7 | `np.take(..., out=...)` ohne `mode="clip"` im Rückwärtslauf | Gradient im Erstlauf 34,0 → 36,3 ms | schlechter |
| 8 | Mehr Eliminationsstufen oder ein lockereres Kriterium (Einträge bis Betrag 2, 10, 100) | Restsystem 125–133 statt 200 Unbekannte, aber 44–61 ms statt 37 ms | schlechter |
| 9 | Kompilier-Einstellungen von JAX selbst (`jax_optimization_level`, `jax_exec_time_optimization_effort`, `jax_memory_fitting_effort`) | Readout bleibt bei 180–190 ms | nichts gebracht |

## Die Versuche im Einzelnen

### 1. SuperLU-Einstellungen

Noel hatte `Relax=1`, `PanelSize=1` und `SymmetricMode` auf dem Mac eingestellt. Auf x86 hätte das Optimum woanders liegen können. Es liegt nicht woanders. Gemessen an den echten 101 Signalmatrizen (751 × 751, 2 597 Einträge), bestes von 5–7 Durchgängen:

| Einstellung | 101 Zerlegungen | Einträge in L+U |
| --- | ---: | ---: |
| Stand Noel (`Relax=1`, `PanelSize=1`, `SymmetricMode`) | 42,1 ms | 8 330 |
| Standard-Supernodes von SuperLU | 61,5 ms | 13 120 |
| ohne `SymmetricMode` | 45,8 ms | 8 330 |
| `DiagPivotThresh=0.1` | 44,1 ms | 8 329 |
| `DiagPivotThresh=0.001` | 44,1 ms | 8 326 |
| `PanelSize=2` | 44,2 ms | 8 330 |
| `Relax=2` | 44,8 ms | 8 704 |

Der Fill hängt also nicht an der Pivotwahl. Die Zeit steckt im festen Aufwand pro Spalte (rund 0,5 µs), nicht in der Arithmetik. Das war der Hinweis, der zur Vorab-Elimination geführt hat.

### 2. Spaltenordnung `MMD_AT_PLUS_A`

Die Ordnung für fast symmetrische Muster. Sie verdoppelt hier den Fill (18 724 statt 8 249 Einträge) und macht die Zerlegung 2,4-mal langsamer (101,8 statt 42,7 ms). Auch mit `DiagPivotThresh=0.01` bleibt es dabei. Auf dem verkleinerten System nach der Vorab-Elimination ist sie ebenfalls schlechter (45,8 statt 24,5 ms). Die dritte Ordnung von SuperLU, `MMD_ATA`, ist dagegen leicht besser als COLAMD und wurde übernommen (siehe `NOTIZEN.md`).

### 3. und 4. Vorab-Elimination nur mit NumPy

Die Idee selbst ist behalten worden (siehe `NOTIZEN.md`). Zwei Umsetzungen davon sind gescheitert:

- **Mit `np.add.reduceat` über `(F, Einträge, k)`:** SuperLU brauchte nur noch 20 statt 61 ms, die Verwaltung in NumPy aber 36 ms (rechte Seiten nachführen 18 ms, Schur-Komplement 9 ms, Auswahl 4 ms, Rest Umkopieren). Netto 5 ms Gewinn.
- **Mit Layout `(Unbekannte, F, k)`, damit `reduceat` über zusammenhängende Blöcke läuft:** Verwaltung 30 ms. Jeder Aufruf schiebt mehrfach 1,6 MB grosse Zwischenfelder durch den Speicher.

Gelöst hat es erst der Wechsel auf die kompilierten CSR-Routinen aus SciPy (`csr_matvecs`, `csc_matvecs`), die direkt im Lösungsvektor arbeiten. Damit kostet das Nachführen aller rechten Seiten zusammen 2,6 ms.

### 5. `mallopt`

Im ersten Aufruf fallen im Vorwärtslauf rund 11 000 Seitenfehler an (44 MB frischer Speicher), im ersten Rückwärtslauf 5 700. Die Idee war, glibc mit `mallopt(M_MMAP_THRESHOLD, 32 MB)`, `M_TRIM_THRESHOLD` und `M_TOP_PAD` dazu zu bringen, freigegebenen Speicher zu behalten statt ihn dem Kernel zurückzugeben.

8 Läufe im Wechsel: Gesamt 1010,6 → 1002,7 ms im Median, das Minimum ist identisch (944,4 und 944,7 ms). Der Build wird 5–7 ms langsamer, der Vorwärtslauf bleibt gleich. Kein messbarer Gewinn. Dazu wäre es ein Eingriff in den ganzen Prozess, also eher eine Regelfrage wie B6 in `FRAGEN_AN_FEYER.md`.

### 6. Supernodes auf dem Restsystem

Nach der Vorab-Elimination ist das Restsystem dichter (6,4 statt 3,5 Einträge pro Zeile). Grössere Supernodes hätten sich dort lohnen können. Im Wechsel gemessen, 9 Durchgänge, bestes Ergebnis für Zerlegung plus zwei Lösungen:

| Einstellung | Zeit |
| --- | ---: |
| `PanelSize=1`, `Relax=1` (Stand) | 16,9 ms |
| `PanelSize=1`, `Relax=2` | 17,7 ms |
| `PanelSize=2`, `Relax=1` | 17,8 ms |
| `PanelSize=3`, `Relax=1` | 18,6 ms |
| `PanelSize=2`, `Relax=2` | 18,7 ms |
| Standard von SuperLU | 23,2 ms |

Eine erste Messung, bei der die Varianten nacheinander statt im Wechsel liefen, hatte für `PanelSize=2` 25 % Gewinn gezeigt. Das war nur die Reihenfolge: Die zuerst gemessene Variante lief auf kaltem Speicher. Seither wird alles im Wechsel gemessen.

### 7. `np.take` mit `out`

`np.take(a, index, axis=0, out=ziel)` schreibt im Standardmodus `mode="raise"` erst in einen Puffer und kopiert dann. Im Rückwärtslauf hat das die eingesparten Zwischenfelder wieder aufgefressen (Gradient im Erstlauf 34,0 → 36,3 ms, 10 Läufe). Mit `mode="clip"` schreibt NumPy direkt ins Ziel. So ist es eingebaut.

### 8. Mehr Eliminationsstufen

Nach drei Stufen bleiben 200 Unbekannte für SuperLU. Zwei Wege, noch weiter zu kommen, wurden am echten System gemessen (Aufbau, Zerlegung und zwei Lösungen, bestes von 5):

| Variante | Stufen | Rest für SuperLU | Zeit |
| --- | ---: | ---: | ---: |
| Stand: höchstens 3 Stufen, Einträge mit Betrag ≤ 1 | 3 | 200 | 37,1 ms |
| bis 6 Stufen, sonst gleich | 5 | 133 | 46,2 ms |
| bis 6 Stufen, Einträge bis Betrag 2 | 5 | 133 | 44,0 ms |
| bis 6 Stufen, Einträge bis Betrag 10 | 5 | 133 | 44,1 ms |
| bis 6 Stufen, Einträge bis Betrag 100 | 6 | 125 | 60,8 ms |

Jede weitere Stufe kostet mehr an Verwaltung (Schur-Komplement, zwei CSR-Blöcke, vier Aufrufe beim Lösen), als SuperLU an den wenigen Spalten spart. Das lockerere Kriterium bringt kaum zusätzliche Unbekannte: Was nach drei Stufen übrig ist, hat entweder keine Diagonale 1 mehr oder gehört zur Mechanik.

### 9. Kompilier-Einstellungen von JAX

JAX 0.9 hat eigene Schalter für den Kompilieraufwand (`jax.config`: `jax_optimization_level`, `jax_exec_time_optimization_effort`, `jax_memory_fitting_effort`). Nur gemessen, nicht eingebaut, weil auch das unter Frage B6 fiele. Auf der CPU ändern sie nichts: Der Readout kompiliert mit `O0`, `O1` oder Aufwand −1 weiter in 180–190 ms, der Gradient in 85–100 ms.

## Nicht ausprobiert, weil es an den Regeln hängt

- **XLA-Flags.** Auf x86 kostet das Kompilieren rund 290 ms von 960 ms, davon 186 ms für den Readout und 96 ms für die Gradientenfunktion. Diese Programme bestehen fast nur aus Operationen von `benchmark.py` selbst (42 und 25 Operationen, unsere Anteile sind je zwei Callbacks). Aus der Bibliothek heraus lässt sich daran nichts ändern, ausser über Flags. Das ist Frage B6. Nur zur Einordnung gemessen und nicht eingebaut: `XLA_FLAGS=--xla_backend_optimization_level=0` senkt den Readout von 185 auf 120 ms und den Gradienten von 91 auf 68 ms, zusammen rund 100 ms. `--xla_llvm_disable_expensive_passes=true` und `--xla_cpu_use_thunk_runtime=false` ändern nichts.
- **Code der importierten Bibliotheken ins eigene Repo kopieren.** Bringt nichts: `benchmark.py` importiert `jax` selbst und vor `differometor` (rund 370 ms). `import differometor` kostet danach noch 4–5 ms.
