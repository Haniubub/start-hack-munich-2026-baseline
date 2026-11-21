# Nachtlauf auf Branch `philip-nacht`: was geändert wurde und was es bringt

Stand: Sa 21.11.2026, 07:10. Der Branch enthält `origin/noel` bis `6635856` (05:38) und `origin/max/nacht` (`f1000e9`) und darauf die Änderungen dieser Nacht. Was nicht funktioniert hat, steht in [`FEHLVERSUCHE.md`](FEHLVERSUCHE.md).

## Ergebnis auf x86-Linux

Maschine: Intel Xeon 2,1 GHz (Emerald Rapids, Nachfolge-Generation des Xeon Platinum 8468 aus dem Brief), `python benchmark.py --cpus 1 --repeats 3`, ohne `XLA_FLAGS`. 10 Läufe pro Stand, abwechselnd, jeder in einem frischen Prozess. Angegeben sind Median / Minimum in ms. Alle Läufe `PASS`.

| Phase | `philip` (`ab82e42`) | `noel` (`6635856`) | `philip-nacht` (`327e82f`) |
| --- | ---: | ---: | ---: |
| **Erstlauf gesamt** | 1200,6 / 1148,9 | 796,9 / 742,5 | **655,4 / 599,1** |
| Imports | 385,9 / 379,3 | 304,7 / 285,8 | 294,8 / 268,2 |
| Build | 76,7 / 68,4 | 62,9 / 60,4 | 62,2 / 54,7 |
| Trace / Lower | 56,0 / 53,8 | 43,7 / 40,4 | **35,9 / 32,7** |
| Kompilieren | 415,9 / 385,3 | 241,7 / 228,5 | **190,6 / 161,6** |
| Vorwärtslauf | 140,8 / 127,0 | 82,2 / 75,9 | **59,7 / 50,9** |
| Gradient | 113,0 / 104,4 | 42,8 / 40,5 | **12,6 / 10,2** |
| Sensitivität und Übergabe | 1,3 / 1,2 | 3,6 / 3,4 | 3,3 / 2,5 |
| Warm gesamt | 200,9 / 183,9 | 97,6 / 85,1 | **63,3 / 57,5** |
| Warm Vorwärtslauf | 121,2 / 115,0 | 63,5 / 55,4 | 52,5 / 48,2 |
| Warm Gradient | 77,8 / 68,3 | 33,0 / 28,2 | 8,4 / 7,3 |
| Fehler gegen Feyers Referenz | 5,3742e-06 | 5,3743e-06 | 5,3743e-06 |

Alle drei Spalten stammen aus einer Messreihe (07:03). Der letzte Commit (`36b3ea7`, ein Callback weniger im Readout) kam danach und bringt im Minimum noch rund 5 ms.

**Die absoluten Zahlen hängen an der Tageszeit.** Die Messmaschine ist geteilt. Bis etwa 04:30 lief alles rund 25 % langsamer (damals `noel` 1045, dieser Branch 873), seit 06:50 läuft es schneller. Vergleichbar sind nur Stände aus derselben Reihe. Die Abstände innerhalb einer Reihe sind stabil, am genauesten in den Phasen Vorwärtslauf, Gradient und Lower.

Ohne Bytecode-Cache (frischer Export mit `git archive`, `PYTHONDONTWRITEBYTECODE=1` wie im Dockerfile, 8 Läufe): `noel` 781,1 / 749,4, `philip-nacht` 646,3 / 601,6. Der Import kostet dann bei beiden rund 300 ms.

Zwischenstand von 02:49 (langsame Phase der Maschine, 12 Läufe): `philip` 1508,5, `noel` (`c261b63`) 1089,6, `philip-nacht` (`ef2f1fc`) 928,2.

Die Zeiten sind anders verteilt als auf dem Mac. Auf x86 ist das Kompilieren der grösste Block nach den Imports, auf dem M4 sind es 75 ms. SuperLU ist 3,5-mal langsamer als auf dem M4.

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
| 9 | Mit Zwischenspeicher bleiben Y, MU und Z auf der Host-Seite | `fast.py` | Gradient 33,9 → 26,3 ms, Vorwärts 78,9 → 76,9 ms (14 Läufe) |
| 10 | Einheitsblöcke im Build ohne dichtes Zwischenfeld | `build.py` | Build 84,0 → 78,6 ms (12 Läufe) |
| 11 | Komponenten schreiben ihre Einträge direkt, drei Kopien im Vorwärtslauf entfallen | `fast.py` | Warm Vorwärts 68,1 → 65,5 ms, Erstlauf unverändert (14 Läufe) |
| 12 | Rückweg der Einträge nur für Gruppen mit Designvariablen | `fast.py` | Warm Gradient 18,2 → 17,2 ms, Erstlauf Gradient 25,4 → 24,9 ms (14 Läufe) |
| 13 | Struktur-Arrays des Builds sind Host-Konstanten (`Constant`) statt Laufzeit-Eingaben | `sparse.py`, `build.py`, `simulate.py` | Build 80,4 → 76,8, Lower 55,1 → 48,6, Kompilieren 296,5 → 293,5, Vorwärts 78,2 → 71,8, Gradient 25,3 → 20,7, Sensitivität 3,3 → 2,4 ms (10 Läufe). Zusammen rund 25 ms |
| 14 | Rückwärtslauf besucht nur gelesene Einträge und nimmt die Adjungierte aus dem Vorwärtslauf | `fast.py` | Gradient 20,6 → 14,9 ms, Vorwärts 71,5 → 69,5 ms, warm Gradient 16,0 → 9,5 ms (10 Läufe) |
| 15 | Komponentenfunktionen rechnen alles, was nicht von der Frequenz abhängt, einmal pro Aufruf | `fast.py` | Einträge mit Ableitungen 9,1 → 7,0 ms (Host, warm). Bit-identisch |
| 16 | Tape pro Funktionsausgabe, Pullback auf zusammenhängenden Zeilen | `fast.py` | Einträge 7,0 → 5,7 ms, Pullback 3,9 → 2,1 ms (Host, warm). 14 bis 16 zusammen: Vorwärts 72,5 → 69,6, Gradient 20,6 → 13,8 ms (10 Läufe) |
| 17 | Detektor-Ports werden auf dem Host ausgewählt, kein XLA-Gather und -Scatter im Readout | `components.py` | Kompilieren 232,9 → 179,3 ms, Lower 37,5 → 34,4 ms, **Erstlauf 701,0 → 637,3 ms** (10 Läufe). Bit-identisch |
| 18 | Detektor nur für die gewählten Ports auswerten, ein Callback statt zwei | `components.py` | Minimum von 10 Läufen: Kompilieren 161,7 → 157,6, Gradient 10,7 → 9,9, Sensitivität 2,5 → 1,9 ms |
| 19 | Nullensuche in den dichten Build-Matrizen mit einem Vergleich | `sparse.py` | 0,9 → 0,6 ms für die Signalmatrix |
| – | Übernommen: Noels Runden 2 und 3 (Tests, Notizen) und Max' Aufteilung in nachgeladene Module | `setups.py`, `simulate.py`, `_*.py`, `runs/noel/` | Ohne `.pyc` übersetzt Python beim Import rund ein Sechstel weniger Quelltext (in der langsamen Phase gemessen: 33 → 28 ms) |

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

- Folgt auf den Vorwärtslauf ein Pullback (`jax.vjp`, `jax.grad`), wertet er die Einträge gleich mit Ableitungen aus und legt Pläne, Einträge, Tape, Signalwerte, Carrier-LU und die Hilfsgrössen Y, MU, Z in einem Zwischenspeicher mit zwei Plätzen ab.
- An JAX gehen dann nur die drei Ausgaben und ein Schlüssel. Y, MU und Z (4,8 MB) wandern nicht mehr durch XLA und zurück (Änderung 9).
- Der Rückwärts-Callback benutzt den Eintrag, wenn er da ist und die Parameter übereinstimmen. Ist er verdrängt, rechnet er den Vorwärtslauf aus seinen eigenen Eingaben noch einmal.
- Reine Vorwärtsaufrufe von `simulate` geben nur noch die drei Ausgaben zurück.
- `fast.FORWARD_CACHE = False` (wird beim Tracen gelesen) schaltet auf das zustandslose Verhalten von vorher zurück: Y, MU und Z fliessen als Residuen durch JAX.
- Unter `vmap` laufen alle Vorwärtsläufe eines Stapels vor dem ersten Pullback. Zwei Plätze würden dann nur den letzten beiden nützen. Gestapelte Aufrufe nehmen deshalb automatisch den zustandslosen Weg.
- Die Resultate sind in allen Fällen bit-identisch. `test_forward_cache.py` prüft Treffer, verdrängte Einträge, mehrfach benutzte Pullbacks, `jit` und `vmap` gegen den zustandslosen Weg.

Damit sind die Callbacks im Normalfall nicht mehr zustandslos, wie es in Noels Beschreibung von `fast.py` stand. Der Zwischenspeicher ist aber nur eine Abkürzung: Fehlt ein Eintrag, kostet das Zeit, kein Resultat hängt davon ab.

Seit Änderung 14 gibt es die Detektor-Basis Z nicht mehr. Im Zwischenspeicher liegt stattdessen auch die Zerlegung der Signalsysteme.

### 6. bis 8. Weniger Speicherverkehr

- **Ein Feld für alle Stufen:** Jede Eliminationsstufe hat die behaltenen Einträge kopiert und ihre Beträge für alle Frequenzen neu bestimmt. Jetzt ändert eine Stufe nur die betroffenen Zeilen und hängt ihr Fill-in an.
- **Layout `(Einträge, F)`:** Einträge, Signalwerte und ihre Kotangenten lagen als `(F, Einträge)` im Speicher. Jede Auswahl von Einträgen lief damit quer durch den Speicher. Jetzt ist ein Eintrag eine zusammenhängende Zeile. Die Ausgaben bleiben bit-identisch, der Gradient ändert sich um 2,5e-16 relativ (andere Summationsreihenfolge).
- **Rückwärtsformeln an Ort und Stelle:** Der Kotangent der Signaleinträge entstand aus zehn Zwischenfeldern mit je 2 500 × 101 Elementen. Jetzt wird er direkt ins Ziel geschrieben.

### 10. Build

`np.eye(n, dtype=complex)` hat für die Einheitsblöcke eine zweite dichte Matrix angelegt (9 MB bei der Signalmatrix), nur um ihre Diagonale zu kopieren. `np.fill_diagonal` schreibt nur die Diagonale. Der unberührte Teil der Nullmatrix bekommt gar keinen echten Speicher, bevor die Einträge herausgezogen werden.

### 13. Struktur-Arrays als Host-Konstanten

`run_build_step` hat 34 JAX-Arrays geliefert. `benchmark.py` und `start_here.py` übergeben sie als Argument an ihre jit-Funktionen. Damit waren sie Laufzeit-Eingaben: Sie gingen in das kompilierte Programm, wurden in jeden Host-Callback kopiert (JAX wandelt jedes Callback-Argument einzeln um), kamen als Residuen des Pullbacks wieder heraus und gingen in das Gradientenprogramm.

30 dieser Arrays ändern sich für ein Setup nie: die Index-Arrays, die Auswahlvektoren der Detektoren und die Einträge der Basismatrizen. Sie sind jetzt `differometor.sparse.Constant`: schreibgeschützte NumPy-Arrays, die als Pytree-Knoten ohne Blätter registriert sind. JAX behandelt sie wie eine Python-Konstante. Eine jit-Funktion wird auf ihren Inhalt spezialisiert, so wie es schon immer war, wenn man die Arrays per Closure benutzt (so machen es die Beispiele).

- Laufzeit-Arrays bleiben `optimized_parameters`, die Basis-`parameters` und die Sweep-Werte (`carrier_changing_values`, `signal_changing_values`). Das sind Daten. Sie lassen sich weiter ersetzen, stapeln und ableiten.
- Zwei Konstanten gelten als gleich, wenn Form, Datentyp und Bytes übereinstimmen. Ein neu gebautes gleiches Setup übersetzt also nicht neu, ein anderes schon.
- Die Callbacks des schnellen Wegs bekommen die Konstanten selbst. Durch JAX wandern nur noch vier Arrays.
- `simulate_jax` wandelt Konstanten am Eingang in JAX-Arrays um.
- Abgeleitete Arrays (Ausschnitte, Rechenergebnisse, Kopien, Pickle) sind wieder gewöhnliche Arrays.
- Jeder Eintrag darf durch ein JAX-Array ersetzt werden, dann läuft er wie früher durch JAX. `sparse.STATIC = False` liefert überall JAX-Arrays.
- Ausgaben und Gradient sind bit-identisch. `test_static.py` prüft das alles.

Zwei Dinge sind anders als vorher: `jax.grad` und `jax.vmap` sehen in einer Konstante kein Array (dafür `jnp.asarray(np.asarray(konstante))` übergeben), und `jnp.asarray(konstante)` gibt die Konstante unverändert zurück.

### 14. Rückwärtslauf

Drei Dinge:

- **Nur gelesene Einträge.** Von den 2 668 Einträgen im Muster der Signalmatrix liest der Pullback 1 342 (die mit einer Designvariable dahinter). 480 weitere gehen nur in den Kotangenten des Carriers ein und sind bei jeder Frequenz gleich. Für sie reicht die Summe über die Frequenzen. Die konstanten Einträge werden gar nicht besucht (`_ReverseLayout`).
- **Adjungierte aus dem Vorwärtslauf.** Eine Sensitivität (Rauschen durch Signal am selben Detektor) legt auf das Signal einen Kotangenten, der bei jeder Frequenz ein Vielfaches α von conj(w) ist, weil Signal und Rauschen durch dieselben Gewichte w gelesen werden. Dann ist S⁻ᵀ·signal_bar = α·conj(Y), und Y hat der Vorwärtslauf schon. Der Kotangent eines Matrixeintrags ist damit ein einziges Produkt statt zwei. Geprüft wird das numerisch (Rest unter 1e-13 des Kotangenten, beim Benchmark sind es 2e-16).
- **Jeder andere Kotangent** kostet eine weitere Lösung mit der Zerlegung des Vorwärtslaufs. Ohne Zwischenspeicher wird neu zerlegt.

Damit entfällt die Detektor-Basis Z = S⁻ᵀ[e_d]. Der Vorwärtslauf löst Sᵀ·conj(Y) = conj(w) mit einer rechten Seite statt zwei. Die SuperLU-Reserve pro Frequenz (`signal_lu`, `_OrderedLU`) ist gelöscht.

Gegen die Fassung davor: Benchmark-Gradient 6,3e-13. 28 Gradienten von sechs Verlustfunktionen über die Wege mit Zwischenspeicher, ohne und unter `vmap` höchstens 2e-12, wo derselbe Löser rechnet. Allgemeine Kotangenten laufen jetzt durch die Vorab-Elimination statt durch SuperLU allein und weichen um 1e-8 bis 4e-8 ab. Gegen das Original-JAX bleibt es bei 2,2e-07 (`runs/noel/test_adjoint.py`). `test_reverse.py` prüft die Abkürzungen gegen die Rechnung ohne sie, auf drei Setups.

### 15. und 16. Auswertung der Komponenten

- **Broadcasting.** Eine Komponentenfunktion bekam jede Eingabe als Feld der Form (Aufrufe, Frequenzen). Jetzt hat eine Eingabe, die sich über den Sweep nicht ändert, die Form (Aufrufe, 1). Die Funktionen sind skalare Formeln. NumPy rechnet damit jeden Teilausdruck ohne Frequenz einmal pro Aufruf (eine Spiegelfläche hat sieben Eingaben, eine davon ist die Frequenz). Bit-identisch. Noel hat dasselbe in `1442ef3` gebaut.
- **Tape pro Ausgabe.** Werte und Ableitungen jeder Funktion wurden in Felder der Form (Aufrufe, Frequenzen, 4) und (Aufrufe, Frequenzen, 4, Eingaben) kopiert, obwohl die meisten Funktionen eine Ausgabe haben. Jetzt gehen die Werte direkt in die Einträge, und das Tape behält die Ableitung jeder gelesenen Ausgabe so, wie die Funktion sie liefert. Der Pullback rechnet pro Ausgabe auf zusammenhängenden Zeilen.

### 17. und 18. Detektor-Ports auf dem Host

Das ist der grösste einzelne Schritt der Nacht. Ein Readout nimmt mit einem Index-Array die Zeilen weniger Ports aus der Detektor-Ausgabe: `df.signal_detector(carrier, signal)[detector_ports]`. So steht es in `benchmark.py`, `start_here.py`, `utils.py` und allen Beispielen. Die Auswahl selbst kompiliert XLA schnell. Ihre Umkehrung im Rückwärtsmodus ist aber ein Scatter in ein Nullfeld, und daran kompiliert XLA rund 50 ms. Dazu kommt die Index-Arithmetik um beide Operationen.

Mit `components.HOST_DETECTOR_PORTS` (Standard: an, nur zusammen mit `HOST_DETECTORS`) behandelt die Ausgabe von `signal_detector` innerhalb einer getracten Funktion `ausgabe[Integer-Array]` selbst:

- Ein Callback wertet die Detektorformel in NumPy nur für die gewählten Ports aus.
- Die Rückwärtsregel schreibt deren Kotangenten in NumPy direkt in Nullfelder für Carrier und Signal.
- Das Ergebnis ist das von JAX-Indexierung: negative Ports zählen von hinten, Ports ausserhalb werden geklemmt und bekommen keinen Kotangenten, mehrfach genannte Ports addieren sich.
- Jede andere Verwendung der Ausgabe ist unverändert. Die vollständige Ausgabe wird nur berechnet, wenn etwas anderes sie liest.

Das Readout-Programm des Benchmarks hat damit 31 statt 42 Operationen, das Gradientenprogramm 19 statt 25, beide ohne Gather und Scatter. Ausgaben und Gradient sind bit-identisch.

**Der Haken:** Es braucht einen Eingriff in JAX-Interna. Der Tracer, den JAX zurückgibt, bekommt eine Unterklasse seiner eigenen Klasse, die sich nur in `__getitem__` unterscheidet (`_pick_ports_on_host`, über `tracer.__class__ = ...`). Abstrakter Typ, Pytree-Struktur und alle anderen Operationen bleiben. Zwei sauberere Wege sind gescheitert, siehe `FEHLVERSUCHE.md` Nr. 11 und 12. Mit konkreten Arrays (ohne Tracing) und unter `vmap` bleibt es bei der normalen JAX-Indexierung. `HOST_DETECTOR_PORTS = False` stellt überall die normale Indexierung wieder her.

`test_ports.py` vergleicht mit der normalen Indexierung und mit dem JAX-Detektor: sechs Arten von Port-Arrays, int32 und int64, getracte und konstante Indizes, `jit`, `grad`, `vjp` mit vorgegebenem Kotangenten, `vmap`, `cond`, andere Formen, und alle übrigen Arten, die Ausgabe zu benutzen.

## Wo die Zeit auf x86 jetzt steckt

Zahlen aus der Messreihe von 07:03 (schnelle Phase der Maschine).

| Block | ms | In unserer Hand |
| --- | ---: | --- |
| Imports | 295 | rund 4 (`import differometor`). Ohne `.pyc` kommen rund 19 ms für das Übersetzen der Quelltexte dazu. Der Rest ist `import jax` und `benchmark.py` |
| Build | 62 | rund 17. Der Rest sind drei kleine XLA-Programme, die `jnp.asarray(..., dtype=float64)` in `benchmark.py` kompiliert |
| Trace / Lower | 36 | wenig. Vorwärts 5, Readout 21, Gradient 9. Jedes Modul kostet einige ms allein für den MLIR-Kontext |
| Kompilieren | 190 | nichts mehr. Vorwärts 6, Readout rund 125, Gradient rund 35. Was bleibt, sind `abs`, Division, `log` und Mittelwert aus `benchmark.py` samt ihren Ableitungen |
| Vorwärtslauf | 60 | alles: SuperLU auf dem Restsystem 12 + 7 fürs Lösen, Elimination 12, Einträge mit Ableitungen 6, Signalwerte 3, Pläne 3, erster Aufruf rund 7 |
| Gradient | 13 | alles: Rückwärtsformeln 4, Pullback 2, Übergabe an XLA und Readout-Rückweg |
| Sensitivität | 3 | zwei Callbacks |

Rund 550 ms sind fix, solange `benchmark.py` unverändert bleibt und keine XLA-Flags erlaubt sind. In unserer Hand liegen noch rund 90 ms, fast alles im Vorwärtslauf.

## Schalter

| Schalter | Standard | Wirkung, wenn aus |
| --- | --- | --- |
| `components.HOST_DETECTOR_PORTS` | an | `signal_detector(...)[ports]` läuft wieder durch XLA (Gather und Scatter). Kostet rund 60 ms im Erstlauf. Kein Eingriff in JAX-Interna mehr |
| `components.HOST_DETECTORS` | an (von Noel) | Detektoren in JAX statt in NumPy. Schaltet auch die Port-Auswahl ab |
| `sparse.STATIC` | an | `run_build_step` liefert überall JAX-Arrays. Kostet rund 25 ms |
| `fast.FORWARD_CACHE` | an | Callbacks ohne Zustand, Y und MU fliessen durch JAX |

## Fragen an Feyer, die dazugekommen sind

Noels Liste steht in `FRAGEN_AN_FEYER.md`. Aus dieser Nacht kommen zwei dazu, beide in der Art von A9 (Host-Detektoren):

| Vorschlag | Frage | Warum es zählt | Bei Nein |
| --- | --- | --- | --- |
| A11 | Die Auswahl der Detektor-Ports (`signal_detector(...)[ports]`) läuft bei uns in NumPy statt durch XLA, mit eigener Rückwärtsregel. Dafür bekommt der zurückgegebene JAX-Tracer ein eigenes `__getitem__`. In Ordnung? | Spart rund 60 ms Kompilierzeit. Resultate bit-identisch. Nutzt eine JAX-Interna (`tracer.__class__`) | `HOST_DETECTOR_PORTS = False` in `components.py` |
| A12 | Die Index-Arrays und konstanten Matrixeinträge aus `run_build_step` sind bei uns Konstanten der jit-Funktionen statt Laufzeit-Eingaben. In Ordnung? | Spart rund 25 ms. Resultate bit-identisch | `STATIC = False` in `sparse.py` |

Der Regeltext in `judging-criteria.pdf` verlangt nur: vollständiges Setup, 101 Frequenzen, 193 Designvariablen, doppelte Genauigkeit, Resultate und Gradienten erhalten, `PASS`. Zu Callbacks oder Interna steht dort nichts.

## Prüfen und messen

```bash
source .venv/bin/activate
# Korrektheit (nach jeder Änderung). Referenz-Arrays wie in runs/noel/NOTIZEN.md beschrieben.
python benchmark.py --cpus 1 --repeats 3 --output check.json
python runs/check_fast.py runs/baseline.npz --fallback
python runs/check_setups.py
python runs/noel/test_detectors.py .
python runs/noel/test_adjoint.py .
python runs/noel/test_perturbed.py .
python runs/philip/test_condensed.py
python runs/philip/test_forward_cache.py
python runs/philip/test_compat.py
python runs/philip/test_static.py
python runs/philip/test_reverse.py
python runs/philip/test_ports.py
# Wechselvergleich unter Linux mit --cpus 1 (Vergleichsstand als Git-Worktree in einem zweiten Ordner)
python runs/philip/ab_linux.py 10 vorher=<ordner_vorher> neu=.
```

## Offene Ideen

| Nr. | Idee | Schätzung (x86) | Haken |
| --- | --- | --- | --- |
| 1 | Noels Frequenzblöcke (je 16 Frequenzen) auch hier: Plan der Vorab-Elimination einmal, Zahlen pro Block | 3–8 ms (weniger Seitenfehler im ersten Vorwärtslauf) | Plan und Zahlen in `_CondensedLU` trennen, grösserer Umbau |
| 2 | Vorab-Elimination nur dort für alle Frequenzen rechnen, wo sich Einträge über die Frequenz ändern | 2–4 ms | Buchführung pro Eintrag, ob er konstant ist |
| 3 | Noels balancierte Zerlegung auf dem Restsystem | höchstens 1,5 ms | `FEHLVERSUCHE.md` Nr. 10 |
| 4 | XLA-Flags oder `jax_disable_most_optimizations` | Joels Branches `sx*` melden 8,5 %. Hier früher gemessen: `--xla_backend_optimization_level=0` senkt Readout und Gradient zusammen um rund 90 ms (vor Änderung 17) | Regelfrage B6 |
