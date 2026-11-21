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

## Nicht ausprobiert, weil es an den Regeln hängt

- **XLA-Flags.** Auf x86 kostet das Kompilieren rund 290 ms von 960 ms, davon 186 ms für den Readout und 96 ms für die Gradientenfunktion. Diese Programme bestehen fast nur aus Operationen von `benchmark.py` selbst (42 und 25 Operationen, unsere Anteile sind je zwei Callbacks). Aus der Bibliothek heraus lässt sich daran nichts ändern, ausser über Flags wie `--xla_backend_optimization_level`. Das ist Frage B6.
- **Code der importierten Bibliotheken ins eigene Repo kopieren.** Bringt nichts: `benchmark.py` importiert `jax` selbst und vor `differometor` (rund 370 ms). `import differometor` kostet danach noch 4–5 ms.
