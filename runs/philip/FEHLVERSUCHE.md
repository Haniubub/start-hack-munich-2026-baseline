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
| 10 | Noels neue Zerlegung (balanciert, Ordnung auf S + Sᵀ, Schwellen-Pivotisierung 0,1) auf dem Restsystem nach der Vorab-Elimination | Fill 3 369 → 3 143 Einträge pro Frequenz, Zerlegung und zwei Lösungen 17,0 → 15,2 ms im besten Fall | zu wenig für den Aufwand |
| 11 | Detektor-Ausgabe als eigenes Objekt mit `__getitem__` und `__jax_array__` zurückgeben | `jnp.abs(objekt)`, `jnp.sum(objekt)` und `jnp.where` brechen in JAX 0.9 ab | verworfen, anders gelöst |
| 12 | Detektor-Ausgabe mit eigener Unterklasse von `ShapedArray` (eigenes `_getitem`) | bräuchte Einträge in mindestens neun typgebundenen Tabellen von JAX | verworfen, anders gelöst |
| 13 | Noels Frequenzblöcke (je 16 Frequenzen durch Vorwärts- und Rückwärtslauf) | nicht gemessen, nur gesichtet | passt nicht ohne Umbau zur Vorab-Elimination |

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

### 10. Balancierte, symmetrische Ordnung auf dem Restsystem

Noel zerlegt seit `b658976` das volle Signalsystem (751 Unbekannte) balanciert, mit einer Minimum-Degree-Ordnung auf S + Sᵀ und Schwellen-Pivotisierung. Bei ihm sinkt der Fill von 8 330 auf 6 100 Einträge. Hier geht nur noch das Restsystem nach der Vorab-Elimination an SuperLU (200 Unbekannte, 1 273 Einträge). Dort ist wenig zu holen (bestes von 9, Zerlegung aller 101 Frequenzen plus je eine Lösung mit zwei rechten Seiten für S und Sᵀ):

| Variante | Fill pro Frequenz | Zerlegung | Summe |
| --- | ---: | ---: | ---: |
| Stand: `MMD_ATA` auf den Spalten, Teilpivotisierung | 3 369 | 10,5 ms | 17,0 ms |
| `MMD_AT_PLUS_A`, Schwelle 0,1, balanciert (Noels Wahl) | 3 143 | 8,8 ms | 15,2 ms |
| `MMD_AT_PLUS_A`, Schwelle 0,1, ohne Balancieren | 3 637 | 11,2 ms | 17,4 ms |
| `MMD_AT_PLUS_A`, Schwelle 1 (Teilpivotisierung), balanciert | 3 288 | 10,5 ms | 16,6 ms |
| `COLAMD`, Schwelle 0,1, balanciert | 3 627 | 10,3 ms | 17,4 ms |

Die Zeiten streuen zwischen zwei Zeilen um etwa 1 ms. Der beste Fall spart höchstens 1,8 ms, und dazu kämen das Balancieren und das Skalieren jeder rechten Seite und Lösung. Die Genauigkeit an gestörten Parametern, wegen der Noel die Skalierung braucht, hält die Vorab-Elimination hier schon: `runs/noel/test_perturbed.py` gibt mit diesem Stand 2,1e-08 als grösste Abweichung von log(Sensitivität), ohne Balancieren.

### 11. und 12. Wege, die Auswahl der Detektor-Ports abzufangen

Behalten wurde Änderung 17 in `NOTIZEN.md`: Die Zeilenauswahl `signal_detector(...)[ports]` läuft auf dem Host. Zwei andere Wege dorthin sind gescheitert.

- **Eigenes Objekt statt eines JAX-Arrays.** Ein Objekt mit eigenem `__getitem__`, das sich über `__jax_array__` in ein Array verwandelt, hätte nur öffentliche Schnittstellen gebraucht. JAX 0.9 wandelt solche Objekte aber nicht mehr von selbst um: `jnp.abs(objekt)`, `jnp.sum(objekt)` und `jnp.where(..., objekt, ...)` brechen mit «Triggering __jax_array__() during abstractification is no longer supported» ab. Als Pytree registriert hätte es ausserdem die Struktur der Ausgabe geändert, und `jax.vjp(df.signal_detector, ...)` hätte keinen normalen Kotangenten mehr angenommen (so benutzt es `runs/noel/test_detectors.py`).
- **Eigener abstrakter Typ.** `Tracer.__getitem__` fragt den abstrakten Wert des Tracers (`aval._getitem`). Eine Unterklasse von `ShapedArray` mit eigenem `_getitem` würde die Auswahl abfangen und liesse sich über `pure_callback_p.bind(..., result_avals=...)` an die Ausgabe hängen. JAX schlägt aber an mindestens neun Stellen im genauen Typ des abstrakten Werts nach (MLIR-Typen, Ergebnis-Handler, Nullen, Abbildung unter `vmap`, Callback-Formen, Sharding). Jede vergessene Tabelle wäre ein `KeyError` in irgendeinem selten benutzten Fall.

Eingebaut ist der kleinste Eingriff: Der zurückgegebene Tracer bekommt eine Unterklasse seiner eigenen Klasse, die sich nur in `__getitem__` unterscheidet. Der abstrakte Typ, die Pytree-Struktur und alle anderen Operationen bleiben, wie JAX sie liefert.

### 13. Frequenzblöcke

Noel schickt seit `66f83b3` je 16 Frequenzen durch den ganzen Vorwärts- und Rückwärtslauf, damit die Zwischenfelder klein bleiben (Seitenfehler im ersten Vorwärtslauf in Docker 10 300 → 8 100). Hier bestimmt die Vorab-Elimination ihr Muster aus den Beträgen über alle Frequenzen und legt dafür Index-Strukturen an, die einige Millisekunden kosten. Pro Block neu aufgebaut wäre das teurer als der Gewinn. Dafür müsste man den Plan (einmal) von den Zahlen (pro Block) trennen. Das ist ein grösserer Umbau und in dieser Nacht nicht mehr sicher zu prüfen gewesen. Als offene Idee in `NOTIZEN.md` vermerkt.

## Nicht ausprobiert, weil es an den Regeln hängt

- **XLA-Flags.** Auf x86 kostet das Kompilieren rund 290 ms von 960 ms, davon 186 ms für den Readout und 96 ms für die Gradientenfunktion. Diese Programme bestehen fast nur aus Operationen von `benchmark.py` selbst (42 und 25 Operationen, unsere Anteile sind je zwei Callbacks). Aus der Bibliothek heraus lässt sich daran nichts ändern, ausser über Flags. Das ist Frage B6. Nur zur Einordnung gemessen und nicht eingebaut: `XLA_FLAGS=--xla_backend_optimization_level=0` senkt den Readout von 185 auf 120 ms und den Gradienten von 91 auf 68 ms, zusammen rund 100 ms. `--xla_llvm_disable_expensive_passes=true` und `--xla_cpu_use_thunk_runtime=false` ändern nichts.
- **`jax_disable_most_optimizations` und XLA-Flags in der Bibliothek setzen.** Die Branches `origin/sx`, `origin/sx2` und `origin/sx2-env` (Joel, auf Basis `philip`) messen genau das und melden dort 8,5 % weniger im Erstlauf. Das ist dieselbe Regelfrage B6. Hier nicht eingebaut.
- **Code der importierten Bibliotheken ins eigene Repo kopieren.** Bringt nichts: `benchmark.py` importiert `jax` selbst und vor `differometor` (rund 370 ms). `import differometor` kostet danach noch 4–5 ms.
