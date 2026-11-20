# Optimierungen: Was wir verbessert haben und was es bringt

Differometor ist auf dem Hackathon-Workload (uifo 3×3, 101 Frequenzen, 193 Designvariablen) mit Kompilierung **ca. 27× schneller** und im warmen Lauf **ca. 77× schneller**. Physikalische Outputs und Gradienten bleiben dabei erhalten. Alle Frequenzen werden voll in double precision gerechnet, nichts ist vorberechnet und keine Berechnung wird ausgelassen.

## Gesamtbild

macOS, auf einen Thread begrenzt (`runs/env.sh`), gleiche Maschine:

| Stand | First-Run gesamt | Kompilieren | Warm-Median | Forward | Gradient |
|---|---:|---:|---:|---:|---:|
| Original (unverändertes Differometor), `runs/baseline.json` | 11,04 s | 3,68 s | 5,072 s | 3,509 s | 1,633 s |
| Sparse-Fast-Path, `runs/step2.json` | 0,79 s | 0,09 s | 0,111 s | 0,053 s | 0,057 s |
| + Build in NumPy, `runs/step4.json` | 0,45 s | 0,09 s | 0,108 s | 0,052 s | 0,056 s |
| **+ Solver-/Adjoint-Optimierungen**, `runs/opt/after-*.json` | **0,39–0,44 s** | 0,09 s | **0,066 s** | 0,043 s | **0,022 s** |

Linux in Docker, `--cpus 1`, aarch64. Vorher und nachher liefen im selben Image, abwechselnd gemessen (`runs/opt/docker/`):

| Stand | First-Run gesamt | Warm-Median | Forward | Gradient |
|---|---:|---:|---:|---:|
| Vor der letzten Runde | 1,34–1,57 s | 0,196–0,202 s | 90 ms | 105–111 ms |
| **Nach der letzten Runde** | **1,19–1,28 s** | **0,126–0,131 s** | 76 ms | **49–55 ms** |

Absolute Docker-Zeiten hängen stark vom Zustand der Docker-VM ab. Ein früherer Lauf (`runs/docker/benchmark-docker.json`) maß mit der Vorversion 0,87 s First-Run. Aussagekräftig ist deshalb nur der direkte Vergleich, der im selben Durchgang gemessen wurde.

Zur Einordnung: Die offizielle Baseline im Brief (Xeon Platinum 8468, 1 CPU) liegt bei 44,2 s First-Run und 17,7 s warm.

## Was geändert wurde

### 1. Sparse-Fast-Path statt dichter JAX-Solves (`differometor/fast.py`)

Das Original löst pro Frequenz dichte 751×751-Systeme mit `jnp.linalg.solve`. Für die Rauschpropagation macht es das sogar pro Rauschquelle. Alles läuft durch XLA, und allein das Kompilieren kostet mehrere Sekunden. Die Signalmatrix hat aber nur 2668 Nichtnullen (0,5 %).

- **Sparse LU:** Carrier- und Signalsysteme werden als CSC-Matrizen gebaut und mit SuperLU (`scipy.sparse.linalg.splu`) zerlegt.
- **Rauschen mit einem Solve:** Das Quantenrauschen des Detektors ist `Re(yᴴ N y)` mit `y = S⁻ᴴ w`. Das braucht einen Solve pro Frequenz statt einem pro Rauschquelle.
- **Matrixeinträge über Dual-Zahlen:** Die Einträge entstehen aus den unveränderten Komponentenfunktionen, ausgewertet auf Dual-Zahlen in NumPy. Dadurch muss weder XLA kompilieren, noch muss man die Physikformeln von Hand ableiten.
- **Handgeschriebener Reverse Mode:** Die Ableitung ist ein Adjoint mit JAX-Konvention für komplexe Kotangenten. Eingebunden ist er als `jax.custom_vjp` um zustandslose `jax.pure_callback`s.
- **Fallback auf JAX:** Der Dispatcher nimmt den Fast-Path nur, wenn das Setup passt. Er greift nicht bei Parameter-Linking, Homodyn-Detektoren, mehreren Rauschdetektoren, Carrier-Sweeps oder Ableitungen nach anderen Eingaben. Dann läuft der Original-JAX-Code (`simulate_jax`).

### 2. Build-Schritt in NumPy (`differometor/build.py`)

Der Build hat konstante Blöcke mit eager `jax.numpy` ausgewertet. Das hat beim ersten Lauf ca. 64 kleine XLA-Programme kompiliert (0,45 s statt 0,04 s). Jetzt läuft der Build in NumPy, die Arrays stimmen bis auf 1 ulp mit dem alten Build überein.

### 3. Solver- und Adjoint-Optimierungen (letzte Runde)

Zuerst wurde das Benchmark-Setup analysiert:

- Von den 2668 Nichtnullen der Signalmatrix hängen nur 1052 von der Frequenz ab, 217 der 751 Zeilen sind komplett frequenzunabhängig.
- **Zeilen streichen ist trotzdem nicht möglich.** Jede Unbekannte wirkt auf die Signal-Lösung oder auf den Detektor-Adjoint.
- Der Benchmark liest das Signal nur an einem Port, also an 2 Zeilen (oberes und unteres Seitenband). Der Rauschdetektor liest exakt dieselben 2 Zeilen.

Darauf bauen vier Änderungen auf:

| Änderung | Vorher | Nachher |
|---|---|---|
| **a) Detektor-Basis statt zweiter LU im Backward.** Der Forward löst pro Frequenz einmal `Z = S⁻ᵀ [e_d₁, e_d₂]`, ein Solve mit 2 rechten Seiten. Daraus folgen `Y = conj(Z · conj(w_d))` im Forward und `λ = Z · signal_bar` im Backward. | Backward zerlegt alle 101 Signalmatrizen neu (ca. 24 ms) | Backward braucht für die Signalsysteme **keine** LU mehr |
| **b) Frequenzunabhängige Komponenten nur einmal auswerten.** Komponenten, deren Eingaben keinen Sweep-Parameter enthalten, werden für eine Zeile ausgewertet und dann auf alle Frequenzen kopiert. | 679 Signal- und 66 Rausch-Komponenten, je 101-mal | 179 Signal- und alle 66 Rausch-Komponenten nur 1-mal |
| **c) Forward ohne Dual-Zahlen.** Der Forward braucht nur Werte. Ableitungen werden nur noch im Backward einmal mitgeschrieben. | Dual-Auswertung im Forward **und** im Backward (je ca. 17 ms) | Werte-Auswertung im Forward, Dual-Auswertung nur im Backward |
| **d) Feste Spaltenordnung für die LU.** Alle Frequenzen haben dasselbe Sparsity-Muster. COLAMD läuft deshalb einmal, danach zerlegt SuperLU mit `permc_spec="NATURAL"` die vorab umsortierte Matrix. | COLAMD bei jeder der 101 Zerlegungen (27 ms) | 22 ms bei identischem Fill-in (8238 Einträge in L+U) |

Der Backward rechnet `V_bar`, `w_bar` und die Rückstreuung auf die Parameter jetzt außerdem vektorisiert statt in einer Python-Schleife über die Frequenzen.

**Fallback:** Wenn der Kotangent des Signals außerhalb der Detektorzeilen liegt, rechnet der Backward wie vorher mit einer LU pro Frequenz. Dasselbe gilt, wenn der Detektor mehr als zwei Zeilen liest. Die Ergebnisse sind dann identisch, nur langsamer.

**Ersparnis dieser Runde:**

| | macOS | Docker |
|---|---:|---:|
| Gradient | −61 % (57 → 22 ms) | −52 % |
| Forward | −19 % (53 → 43 ms) | −16 % |
| Warm gesamt | −40 % (0,110 → 0,066 s) | −35 % |

Die Ersparnis wurde nur gesamt gemessen und nicht pro Einzelmaßnahme a–d aufgeschlüsselt.

## Wo die verbleibende Zeit steckt

Im First-Run (ca. 0,4 s auf macOS) steckt kaum noch Physik:

- **Imports:** ca. 0,27–0,4 s, davon `import jax` allein ca. 0,22 s und `scipy.sparse.linalg` ca. 0,04 s.
- **JAX-Kompilieren:** ca. 0,09 s, für Readout/Sensitivität und die Gradienten-Verkettung im Benchmark.
- **Forward und Gradient:** zusammen ca. 0,065 s.

Weitere Optimierungen an der Physik bringen für den Gesamtwert also nur noch wenig.

## Korrektheit

- **Benchmark-Checks:** `benchmark.py` meldet in allen Läufen **PASS**. Der maximale Log-Fehler der Sensitivität zur Referenz liegt bei 5,4e-6, erlaubt sind 1e-5. Den gleichen Wert erreicht die Vorversion.
- **Vorher gegen nachher (letzte Runde):**
  - Carrier und Powers sind bitidentisch.
  - Signal weicht um 5,5e-20 ab.
  - Sensitivität weicht um 1,1e-10 ab, Gradient um 7,6e-10 (relativ zum Maximum).
- **Gegen das Original (dichtes JAX):**
  - Sensitivität 5,8e-9, Gradient 8,0e-8, Signal 1,9e-7.
  - Einzelne Gradienteneinträge in der Größenordnung 1e-13 bei einem Maximum von 5,8e4 sind numerisches Rauschen. Sie weichen schon zwischen dem Original und der ersten Fast-Path-Version um bis zu 100 % ab.
- **Andere Setups:** `runs/check_setups.py` vergleicht gegen JAX und läuft für aLIGO (Fast-Path), aLIGO mit Linking und Voyager mit Homodyn-Detektoren (beide über JAX). Die Abweichungen liegen bei höchstens ca. 1e-8.
- **Ableitung nach Frequenzen:** Wird nach etwas anderem als den Designvariablen abgeleitet, fällt der Dispatcher korrekt auf JAX zurück (`runs/check_fast.py --fallback`).

## Reproduzieren

```bash
# macOS / lokal (ohne CPU-Affinität: Threads begrenzen)
source runs/env.sh
python benchmark.py --repeats 10 --output runs/opt/after.json --save-arrays runs/opt/after.npz
python runs/compare_npz.py runs/opt/before.npz runs/opt/after.npz   # Arrays vergleichen
python runs/check_fast.py runs/opt/before.npz --fallback             # inkl. JAX-Fallback
python runs/check_setups.py                                          # aLIGO / Voyager

# Linux in Docker, eine CPU
runs/docker_bench.sh --repeats 10
```

Für den Vorher/Nachher-Vergleich in Docker lief die alte `fast.py` im selben Image, per `-v <alte fast.py>:/app/differometor/fast.py:ro` gemountet.
