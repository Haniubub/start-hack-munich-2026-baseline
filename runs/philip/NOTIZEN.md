# Nachtlauf auf Branch `philip-nacht`: was geändert wurde und was es bringt

Stand: Sa 21.11.2026, Basis ist `origin/noel` (`c261b63`, 00:39). Was nicht funktioniert hat, steht in [`FEHLVERSUCHE.md`](FEHLVERSUCHE.md).

## Ergebnis auf x86-Linux

Das ist die erste Messung des Teams auf x86. Maschine: Intel Xeon 2,1 GHz (Emerald Rapids, Nachfolge-Generation des Xeon Platinum 8468 aus dem Brief), `python benchmark.py --cpus 1 --repeats 3`, ohne `XLA_FLAGS`. 12 Läufe pro Stand, abwechselnd, jeder in einem frischen Prozess. Angegeben sind Median / Minimum in ms. Alle Läufe `PASS`.

| Phase | `philip` (`ab82e42`) | `noel` (`c261b63`) | `philip-nacht` (`605b73e`) |
| --- | ---: | ---: | ---: |
| **Erstlauf gesamt** | 1508,5 / 1459,6 | 1044,8 / 985,9 | **914,0 / 864,7** |
| Imports | 509,2 / 486,8 | 373,9 / 359,0 | 368,9 / 350,0 |
| Build | 91,7 / 88,1 | 92,0 / 83,9 | 81,3 / 76,5 |
| Trace / Lower | 73,8 / 69,5 | 52,8 / 51,2 | 54,3 / 49,9 |
| Kompilieren | 525,3 / 500,5 | 289,3 / 280,7 | 287,3 / 278,8 |
| Vorwärtslauf | 176,7 / 166,0 | 131,4 / 125,3 | **78,5 / 73,6** |
| Gradient | 121,0 / 115,3 | 83,9 / 77,2 | **32,2 / 29,5** |
| Warm gesamt | 237,7 / 226,2 | 162,9 / 156,7 | **87,0 / 82,8** |
| Warm Vorwärtslauf | 156,2 / 148,8 | 107,7 / 102,9 | 66,2 / 62,6 |
| Warm Gradient | 80,9 / 76,8 | 55,2 / 50,2 | 19,4 / 18,1 |
| Fehler gegen Feyers Referenz | 5,3742e-06 | 5,3747e-06 | 5,3743e-06 |

Die Spalten `noel` und `philip-nacht` stammen aus derselben Messreihe (02:35), die Spalte `philip` aus einer früheren (02:05, dort lag `noel` bei 1067,5 / 1047,5). Zwischen zwei Messreihen streut die Maschine um 20–50 ms, innerhalb einer Reihe sind die Stände vergleichbar.

Die Zeiten sind anders verteilt als auf dem Mac. Auf x86 kostet das Kompilieren rund 290 ms und damit fast ein Drittel, auf dem M4 sind es 75 ms. SuperLU ist 3,5-mal langsamer als auf dem M4.

## Was geändert wurde

| Nr. | Änderung | Datei | Gewinn (x86, Median, abwechselnd gemessen) |
| --- | --- | --- | --- |
| 1 | Basismatrizen sparse statt dicht (`BaseMatrix`) | `sparse.py`, `build.py`, `fast.py`, `simulate.py` | Erstlauf 1068,9 → 1013,8 ms (8 Läufe). Vorwärts 134,8 → 109,4, Gradient 88,1 → 65,4, Build 95,2 → 86,0 |
| 2 | Alle Frequenzen in einer block-diagonalen SuperLU-Zerlegung (`_BlockLU`) | `fast.py` | Vorwärts 109,0 → 104,2 ms (10 Läufe) |
| 3 | Einfache Unbekannte vor der LU exakt eliminieren (`_CondensedLU`) | `fast.py` | Vorwärts 105,8 → 82,1 ms, warm 93,7 → 67,2 ms (10 Läufe) |
| 4 | Spaltenordnung `MMD_ATA` statt COLAMD für das Restsystem | `fast.py` | Vorwärts 85,3 → 80,7 ms, warm 70,7 → 65,6 ms (10 Läufe) |
| 5 | Rückwärtslauf verwendet den Vorwärtslauf wieder (`FORWARD_CACHE`) | `fast.py` | Gradient 62,7 → 35,3 ms, Vorwärts 81,3 → 85,3 ms, Erstlauf 966,4 → 922,8 ms (10 Läufe) |
| 6 | Werte aller Eliminationsstufen in einem Feld | `fast.py` | Vorwärts 82,6 → 80,0 ms, warm 74,7 → 69,5 ms (10 Läufe) |
| 7 | Layout `(Einträge, F)` statt `(F, Einträge)` | `fast.py` | Vorwärts 79,2 → 75,6 ms, Gradient 35,0 → 33,7 ms, warm gesamt 95,7 → 90,0 ms (10 Läufe) |
| 8 | Rückwärtsformeln an Ort und Stelle rechnen | `fast.py` | Warm Gradient 21,1 → 18,8 ms, Erstlauf Gradient 33,4 → 32,8 ms (14 Läufe) |

### 1. Basismatrizen sparse

Die drei Basismatrizen (Carrier, Signal, Rauschen) sind zu über 99 % leer, lagen aber als 20 MB dichte Felder vor. Sie wurden im Build aufs Gerät kopiert, in jeden Host-Callback hineinkopiert und vom Vorwärtslauf als Residuen des Pullbacks wieder herausgegeben. Noel hatte das als offene Idee 1 mit 5–8 ms geschätzt. Auf x86 sind es 55 ms, weil dort frischer Speicher teurer ist (4-KB-Seiten; selbst nach der Änderung fallen im ersten Vorwärtslauf noch rund 11 000 Seitenfehler an).

`run_build_step` liefert sie jetzt als `BaseMatrix` (`differometor/sparse.py`): Index, Werte und Form der Einträge, registriert als JAX-Pytree. Es gibt weiterhin nur eine Quelle, die Matrizen können also nicht auseinanderlaufen.

- Der schnelle Weg liest die Einträge direkt.
- `simulate_jax` baut die dichte Matrix mit `dense_matrix` wieder auf. Dichte Eingaben von altem Code funktionieren weiter.
- `np.asarray(arrays["signal_matrix"])` gibt die dichte Matrix zurück.
- Alle Ausgaben und Gradienten sind bit-identisch zum Stand davor.

### 2. Block-diagonale Zerlegung

Statt 101 Aufrufen von `gstrf` und 202 von `solve` wird `diag(B_0, …, B_100)` einmal zerlegt. Die Pivotsuche bleibt in ihrem Block, darum sind Faktoren und Lösungen bit-identisch. Gespart wird der feste Aufwand pro Aufruf.

### 3. Vorab-Elimination

SuperLU braucht auf x86 rund 0,5 µs festen Aufwand pro Spalte, unabhängig davon, wie wenig darin steht. Bei 751 Spalten, 101 Frequenzen und nur 3,5 Einträgen pro Zeile ist das fast die ganze Zerlegungszeit.

Die meisten Unbekannten der Signalmatrix sind einfach: Die Diagonale ist exakt 1, und die übrigen Einträge der Zeile haben Betrag ≤ 1 (Quellen, Durchleitungen, Streuung an Spiegeln), oder keine andere Zeile verweist auf die Unbekannte.

- `_eliminable` wählt eine Menge solcher Unbekannten, die untereinander nicht gekoppelt sind. Ihr Block der Matrix ist dann die Einheitsmatrix.
- `_Elimination` entfernt sie mit einem exakten Schur-Komplement, für alle Frequenzen in einem Zug.
- Das wiederholt sich, solange eine Stufe noch mindestens ein Viertel entfernt. Beim Benchmark sind es drei Stufen: 751 → 514 → 318 → 200 Unbekannte. Bei aLIGO bleiben 20 von 82.
- SuperLU zerlegt nur noch den Rest, mit Teilpivotisierung wie bisher.
- Die rechten Seiten werden mit den kompilierten CSR-Routinen aus SciPy nachgeführt (`scipy/sparse/_sparsetools`, direkt geladen wie SuperLU). Fehlt das Modul, entfallen die Stufen und es läuft wie in Änderung 2.

**Warum das stabil ist:** Pivot 1 und Zeileneinträge ≤ 1 bedeuten, dass keine Zeile über ihre eigene Grössenordnung hinaus wächst. Zeilen mit grossen Einträgen (Mechanik, bis 1,5e13) werden nie als Pivotzeile benutzt. Noels Fehlversuch mit erzwungener Diagonal-Pivotierung (Fehler 1e75) pivotierte auf allen Zeilen, auch auf solchen.

**Genauigkeit:** Die Abweichung zum Original-JAX ist gleich gross wie vorher.

| Grösse | vorher (SuperLU allein) | mit Vorab-Elimination und `MMD_ATA` |
| --- | ---: | ---: |
| signal | 1,7e-07 | 1,2e-07 |
| noise | 3,7e-08 | 3,8e-08 |
| sensitivity | 5,7e-09 | 5,7e-09 |
| gradient | 8,0e-08 | 8,0e-08 |
| Fehler gegen Feyers Referenz | 5,3747e-06 | 5,3743e-06 |

Ein Nebenbefund aus `test_condensed.py`: Mit zufälligen rechten Seiten erfüllt SuperLU allein einzelne Gleichungen nur schlecht (zeilenweiser Rückwärtsfehler 1e-03 bis über 1e-01, je nach Spaltenordnung), weil sich die Zeilen um über 20 Grössenordnungen unterscheiden. Mit der Vorab-Elimination liegt er unter 1e-07. Für die echten rechten Seiten des Benchmarks spielt das keine Rolle, beide Wege stimmen dort überein.

### 4. Spaltenordnung

`MMD_ATA` erzeugt auf dem Restsystem etwas weniger Fill als COLAMD. Die Resultate ändern sich im Rahmen der Rundung (Tabelle oben).

### 5. Vorwärtslauf wiederverwenden

Der Rückwärts-Callback hat beide Pläne neu aufgebaut, die Werte der Signalmatrix und die Carrier-Zerlegung neu gerechnet und alle Matrixeinträge ein zweites Mal ausgewertet, diesmal mit Ableitungen. Das hatte der Vorwärts-Callback kurz davor schon getan.

- Folgt auf den Vorwärtslauf ein Pullback (`jax.vjp`, `jax.grad`), wertet er die Einträge gleich mit Ableitungen aus und legt Pläne, Einträge, Tape, Signalwerte und Carrier-LU in einem Zwischenspeicher mit zwei Plätzen ab.
- Der Schlüssel dazu wandert als Residuum durch JAX. Der Rückwärts-Callback benutzt den Eintrag, wenn er da ist und die Parameter übereinstimmen. Sonst rechnet er wie bisher aus den Residuen.
- Reine Vorwärtsaufrufe von `simulate` sind unverändert.
- `fast.FORWARD_CACHE = False` schaltet auf das zustandslose Verhalten zurück.
- Die Resultate sind in beiden Fällen bit-identisch. `test_forward_cache.py` prüft Treffer, verdrängte Einträge, mehrfach benutzte Pullbacks und `jit` gegen den zustandslosen Weg.

Damit sind die Callbacks nicht mehr zustandslos, wie es in Noels Beschreibung von `fast.py` stand. Der Zwischenspeicher ist aber nur eine Abkürzung: Alles, was der Rückwärtslauf braucht, fliesst weiterhin als Array durch JAX.

### 6. bis 8. Weniger Speicherverkehr

- **Ein Feld für alle Stufen:** Jede Eliminationsstufe hat die behaltenen Einträge kopiert und ihre Beträge für alle Frequenzen neu bestimmt. Jetzt ändert eine Stufe nur die betroffenen Zeilen und hängt ihr Fill-in an.
- **Layout `(Einträge, F)`:** Einträge, Signalwerte und ihre Kotangenten lagen als `(F, Einträge)` im Speicher. Jede Auswahl von Einträgen lief damit quer durch den Speicher. Jetzt ist ein Eintrag eine zusammenhängende Zeile. Die Ausgaben bleiben bit-identisch, der Gradient ändert sich um 2,5e-16 relativ (andere Summationsreihenfolge).
- **Rückwärtsformeln an Ort und Stelle:** Der Kotangent der Signaleinträge entstand aus zehn Zwischenfeldern mit je 2 500 × 101 Elementen. Jetzt wird er direkt ins Ziel geschrieben.

## Wo die Zeit auf x86 jetzt steckt

| Block | ms | In unserer Hand |
| --- | ---: | --- |
| Imports | 369 | rund 5 (`import differometor`). Der Rest ist `import jax` und `benchmark.py` |
| Build | 81 | rund 24. Rund 55 sind drei kleine XLA-Programme, die `jnp.asarray(..., dtype=float64)` in `benchmark.py` kompiliert |
| Trace / Lower | 54 | wenig. Jedes Modul kostet 6–9 ms allein für den MLIR-Kontext |
| Kompilieren | 287 | nichts. Readout 186 ms, Gradient 96 ms, fast nur Operationen aus `benchmark.py` |
| Vorwärtslauf | 78 | alles: SuperLU auf dem Restsystem 21, Einträge mit Ableitungen 16, Elimination 15, Signalwerte 5, Rest Übergabe und erster Aufruf |
| Gradient | 32 | alles: Rückwärtsformeln 14, Pullback der Einträge 5, Rest Übergabe an XLA und Readout-Rückweg |

Rund 800 ms sind fix, solange `benchmark.py` unverändert bleibt und keine XLA-Flags erlaubt sind. Von den 1045 ms des Stands `noel` lagen rund 245 ms in unserer Hand, jetzt sind es noch rund 115 ms.

## Prüfen und messen

```bash
source .venv/bin/activate
# Korrektheit (nach jeder Änderung). Referenz-Arrays wie in runs/noel/NOTIZEN.md beschrieben.
python benchmark.py --cpus 1 --repeats 3 --output check.json
python runs/check_fast.py runs/baseline.npz --fallback
python runs/check_setups.py
python runs/noel/test_detectors.py .
python runs/philip/test_condensed.py
python runs/philip/test_forward_cache.py
# Wechselvergleich unter Linux mit --cpus 1 (Vergleichsstand als Git-Worktree in einem zweiten Ordner)
python runs/philip/ab_linux.py 10 vorher=<ordner_vorher> neu=.
```

## Offene Ideen

| Nr. | Idee | Schätzung (x86) | Haken |
| --- | --- | --- | --- |
| 1 | Auswertung der Matrixeinträge mit Ableitungen straffen (16 ms im ersten Aufruf, 11 ms warm) | 3–6 ms | Die Dual-Zahlen-Auswertung ist schon auf die Designvariablen beschränkt |
| 2 | Mehr Unbekannte vor der LU eliminieren. Von den 200 übrigen haben 165 noch Diagonale 1, aber Einträge mit Betrag > 1 | 3–5 ms | Bräuchte Pivots ungleich 1 und eine Prüfung pro Frequenz |
| 3 | Seitenfehler im ersten Vorwärtslauf senken (rund 11 000, also 44 MB frischer Speicher) | 3–8 ms | Kleinteilig, `mallopt` hat nichts gebracht |
| 4 | XLA-Flags für schnelleres Kompilieren | rund 100 ms. Nur zur Einordnung gemessen, nicht eingebaut: `XLA_FLAGS=--xla_backend_optimization_level=0` senkt Readout von 185 auf 120 ms und Gradient von 91 auf 68 ms | Regelfrage B6 |
