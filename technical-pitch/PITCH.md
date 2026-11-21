# Faster Physics: technischer Nutzen, durchgerechnet

Grundlage für den Pitch vor der Jury. Stand: 21.11.2026.

Alle Zahlen stammen aus Philips Bericht (`philip-nacht` / `philip-fastest`) und aus Feyers Baseline-README. Sie wurden auf verschiedenen Maschinen gemessen, deshalb sind Faktoren als Größenordnung zu lesen. Abschnitt 8 nennt, was vor dem Pitch noch nachgemessen werden sollte.

---

## Die drei Kernbotschaften

1. **Rund 145-mal schneller im gewerteten Lauf.** Der Erstlauf dauert 0,31 s statt 44,2 s, bestätigt auf 7 veränderten Setups.
2. **Eine Optimierung mit 10.000 Schritten dauert 9 Minuten statt 2 Tagen.** Der Gradient kostet nur noch 16 % eines Vorwärtslaufs.
3. **Genauer als das Original.** In einem schlecht konditionierten Testfall liegt das Original doppelt über der Toleranz. Unsere Rechnung ist dort rund 670.000-mal genauer.

---

## 1. Speedups

| Kennzahl | Baseline | Unsere Lösung | Faktor |
| --- | ---: | ---: | ---: |
| Erstlauf gesamt, gewertet (Median) | 44.204 ms | 305,9 ms | **≈ 145×** |
| Erstlauf auf 7 Setup-Varianten | – | 311,9–324,7 ms | ≈ 136–142× |
| Warmlauf (Simulation + Gradient) | 17.679 ms | 53,1 ms | **≈ 333×** |
| Tracing + Kompilieren | 22.918 ms | ≈ 61 ms | ≈ 374× |

- **Maschinen:** Die Baseline hat Feyer gemessen. Der Erstlauf stammt von einer Nebius-VM mit demselben CPU-Modell (Xeon Platinum 8468), der Warmlauf von einer anderen x86-Maschine (Stand `philip-nacht`, ohne die native Zerlegung aus Änderung 24).
- **Im Team-Vergleich:** Auf den Varianten ist die Lösung 2,2-mal schneller als der Branch `noel`.

---

## 2. Wo die 0,31 s jetzt stecken

| Phase | ca. ms | Anteil |
| --- | ---: | ---: |
| Imports (überwiegend `import jax`) | 183 | 60 % |
| Build | 17 | 5 % |
| Tracing / Lowering | 26 | 8 % |
| Kompilieren | 35 | 11 % |
| Vorwärtslauf (Physik) | 40 | 13 % |
| Gradient | 10 | 3 % |
| Sensitivität und Übergabe | 2 | 1 % |

Die Werte stammen aus mehreren Messreihen und ergeben zusammen etwa 313 ms statt 306 ms.

**Botschaft:** Die eigentliche Physik, also Vorwärtslauf plus Gradient, braucht noch etwa 50 ms. Der größte Teil ist inzwischen der Start von Python und JAX.

---

## 3. Was das für Feyers Forschung bedeutet

Bei Feyer sucht eine KI nach neuen Detektor-Designs. Jeder Suchschritt ist eine Simulation plus Gradient. Entscheidend ist deshalb vor allem der Warmlauf.

**Annahmen:** ein CPU-Kern; Baseline mit 44,2 s im ersten und 17,7 s in jedem weiteren Schritt; unsere Lösung mit 0,31 s und 53 ms.

### Optimierungslauf mit N Schritten

| Schritte | Baseline | Unsere Lösung | Kernstunden (Baseline → wir) |
| ---: | ---: | ---: | ---: |
| 100 | 30 min | 6 s | 0,50 → 0,002 |
| 1.000 | 4,9 h | 53 s | 4,9 → 0,015 |
| 10.000 | 2,0 Tage | 8,9 min | 49 → 0,15 |
| 100.000 | 20,5 Tage | 89 min | 491 → 1,5 |

### Größere Szenarien

| Szenario | Baseline | Unsere Lösung |
| --- | ---: | ---: |
| Auswertungen pro Kern und Tag | 4.887 | 1,63 Mio. |
| 100 Optimierungen à 10.000 Schritte | 205 Tage (4.912 Kernstunden) | 15 h (15 Kernstunden) |
| 1.000 verschiedene Setups, je 1 Auswertung inkl. Kompilieren | 12,3 h | 5,1 min |
| 1.000 Setups à 100 Schritte | 20,8 Tage | 93 min |

**Botschaft:** Was bisher einen Rechencluster oder Wochen gebraucht hat, läuft jetzt auf einem einzelnen Kern über Nacht oder in der Mittagspause. Bei gleichem Budget kann die KI etwa 330-mal mehr Designs prüfen.

---

## 4. Genauigkeit als eigener Gewinn

| Prüfung | Ergebnis | Anteil an der Toleranz (1e-5) |
| --- | --- | ---: |
| Benchmark-Setup gegen Feyers Referenz | 5,37e-6, `PASS` | 54 %* |
| 7 Setup-Varianten gegen den Original-Code | höchstens 3,6e-8, alle `PASS` | 0,36 % |
| Schlecht konditioniertes Setup (Konditionszahl 1e15), Abweichung von der exakt nachgerechneten Lösung | Original: 2e-5, unsere Lösung: 3e-11 | Original 200 %, wir ≈ 0 % |
| Zeilenweiser Rückwärtsfehler bei zufälligen rechten Seiten | SuperLU allein: 1e-3 bis 1e-1, mit unserer Vorab-Elimination: < 1e-7 | – |

\* Ältere, kaum optimierte Stände hatten schon denselben Wert. Die 5,37e-6 stammen also vermutlich nicht von unseren Änderungen, sondern sind ein Plattform-Unterschied. Das ist noch zu prüfen, siehe Abschnitt 8.

**Warum das für Feyer zählt:** Eine KI, die nach exotischen Designs sucht, landet genau in schlecht konditionierten Bereichen. Ist der Simulator dort ungenau, optimiert sie auf Rechenfehler statt auf echte Physik. Sie folgt dann gewissermaßen einem Navi mit falscher Karte. Ein genauerer Simulator verhindert solche Scheinlösungen.

---

## 5. Argumente für den Sonderpreis „überraschendste Lösung"

1. **Schneller durch weniger JAX.** Ausgerechnet in einem JAX-Projekt kam der größte Gewinn daher, die Lineare Algebra *aus* XLA herauszunehmen. Statt dichter Matrizen in JAX rechnen wir mit dünn besetzten Matrizen und einem direkten Löser, dazu eine selbst geschriebene Rückwärtsrechnung.
2. **Die Physik hat die Abkürzung geliefert.** Die Matrizen sind zu über 99 % leer, ihre Diagonale ist exakt 1 und alle anderen Einträge haben höchstens Betrag 1. Damit lassen sich die meisten Unbekannten exakt und nachweislich stabil eliminieren: beim Benchmark 751 → 200 (−73 %), bei aLIGO 82 → 20 (−76 %).
3. **Den Gradienten gibt es fast gratis.** Der Rückwärtslauf kostet 16 % eines Vorwärtslaufs (8,4 ms gegen 52,5 ms). Üblich ist das Ein- bis Dreifache. Der Grund: Signal und Rauschen werden über dieselben Detektor-Gewichte gelesen. Deshalb fällt die nötige Adjungierte schon im Vorwärtslauf mit ab.
4. **Eine Pivotfolge für alle 101 Frequenzen.** Alle Frequenzen haben dasselbe Besetzungsmuster. Ein eigener C-Kern zerlegt sie mit einer gemeinsamen Pivotfolge und prüft jeden Pivot an jeder Frequenz. Fällt eine Prüfung durch, wird wie bisher gerechnet.
5. **Schneller *und* genauer.** Normalerweise tauscht man Genauigkeit gegen Tempo. Hier wurde beides besser (Abschnitt 4).
6. **Saubere Methodik.**
   - Gemessen wurde im Wechsel in frischen Prozessen.
   - Fehlversuche sind dokumentiert.
   - Fast jede Maßnahme hat einen eigenen Schalter.
   - Die Build-Ausgaben sind für 27 Setups identisch (946 Arrays).
   - Getestet wurde auch mit aLIGO, Voyager und Zufalls-UIFOs.

---

## 6. Ehrlich trennen: Benchmark-Gewinn und Forschungs-Gewinn

| Bereich | Gewinn | Nutzen für die Forschung |
| --- | --- | --- |
| Numerischer Kern (Vorwärtslauf, Gradient) | Warmlauf ≈ 333× | **hoch:** zählt in jedem Optimierungsschritt |
| Kompilieren | ≈ 374× | **mittel:** zählt bei jedem neuen Setup, also bei der Suche nach neuen Topologien |
| Start-Maßnahmen (Imports, Bytecode usw.) | ≈ 144 ms, einmal pro Prozess | **gering:** 0,03 % eines Laufs mit 10.000 Schritten |

**Empfehlung:** Die Start-Maßnahmen offen als Benchmark-Optimierung benennen. Das Feyer-Team erkennt das ohnehin, und die klare Trennung macht den Rest glaubwürdiger.

---

## 7. Erwartbare Fragen der Jury

| Frage | Antwort |
| --- | --- |
| Hält das auch auf dem finalen Setup? | Auf 7 veränderten Setups alle `PASS`, Fehler höchstens 3,6e-8. Die Struktur wird zur Laufzeit aus dem Setup abgeleitet, nichts ist fest einprogrammiert. |
| Läuft das auf der GPU? | Der schnelle Weg ist für die CPU gebaut, so wie der Benchmark. Der ursprüngliche JAX-Weg bleibt erhalten und lässt sich jederzeit nutzen. |
| Was wird nicht abgedeckt? | Verknüpfte Parameter (Amplitudenmodulation), mehrere Carrier-Arbeitspunkte und mehr als ein Rauschdetektor. Diese Fälle laufen über den alten Weg: korrekt, aber langsamer. |
| Was passiert bei einer anderen JAX-Version? | Die Eingriffe sind auf jax 0.9.0 abgestimmt. Fast jede Maßnahme hat einen eigenen Schalter, der mitgelieferte Bytecode lässt sich entfernen. |
| Wie viel Speicher braucht es? | Noch nicht gemessen (siehe Abschnitt 8). Die Baseline braucht 14,4 GiB. |

---

## 8. Vor dem Pitch noch messen

| Messung | Warum | Wie |
| --- | --- | --- |
| Baseline auf der Nebius-VM | Faktor auf derselben Maschine statt gemischt | `python benchmark.py --cpus 1 --repeats 3` im unveränderten Kit |
| Warmlauf von `philip-fastest` auf der VM | Bisher stammt der Warmlauf von `philip-nacht` auf einer anderen Maschine; mit der nativen Zerlegung dürfte er niedriger liegen | aus dem JSON-Report des Benchmarks |
| Spitzenspeicher | Weniger Speicher heißt mehr parallele Läufe pro Maschine. Bei 14,4 GiB passen auf 32 GB nur 2 Läufe | `/usr/bin/time -v python benchmark.py --cpus 1 --repeats 3`, Zeile „Maximum resident set size" |
| Fehler der Baseline gegen die Referenz auf der VM | Klärt, ob die 5,37e-6 ein Plattform-Unterschied sind | Benchmark-Report der unveränderten Baseline |

---

## Sätze für den Pitch

- „Eine Detektor-Optimierung mit 10.000 Schritten dauert bei uns 9 Minuten statt 2 Tagen."
- „Die Physik selbst braucht noch 50 Millisekunden. Den Rest der Zeit verbringt Python mit `import jax`."
- „Der Gradient kostet uns 16 % einer Simulation. Wir haben die Struktur der Physik genutzt, nicht nur den Code beschleunigt."
- „Wir sind nicht nur schneller, sondern auch genauer: Im getesteten Fall, in dem das Original die Toleranz reißt, liegen wir bei 3e-11."
