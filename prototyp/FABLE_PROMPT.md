# Prompt für Fable: Feyer "Faster Physics"

Ich nehme am START Hack Munich teil, Challenge "Faster Physics" von Feyer. Dieses Repo ist eine
Kopie von feyerlabs/start-hack-munich-2026 und enthält den Simulator Differometor (Python + JAX).
Ziel: Die komplette Erstlauf-Zeit von `python benchmark.py --cpus 1` so weit wie möglich senken,
ohne die Ergebnisse zu verändern. Baseline laut README: 44,2 s, davon 22,9 s Kompilierung.
Lies zuerst README.md, challenge.pdf und judging-criteria.pdf.

## Regeln, die die Abgabe gültig halten

- benchmark.py und die Dateien in examples/data nicht ändern. Der Report speichert Hashes davon;
  geändert wird nur der Ordner differometor/.
- Alle 101 Frequenzen, alle 193 Stellgrößen, doppelte Genauigkeit. Nichts vorberechnen, nichts
  zwischen Läufen zwischenspeichern, keinen persistenten Kompilier-Cache einschalten.
- Sensitivität: |log(sens) - log(referenz)| <= 1e-5 an jeder Frequenz. Die Baseline liegt selbst
  schon bei 5,4e-6, die Reserve ist also klein.

## Was schon vorliegt

Ordner `prototyp/`. Bitte zuerst lesen und `python prototyp/test_numpy.py .` ausführen
(mit `--compare` zusätzlich der Vergleich gegen das Original-JAX auf 3 Frequenzen).

- sparse_prototyp.py: Vorwärtslauf mit einer dünn besetzten Zerlegung pro Frequenz
  (scipy.sparse.linalg.splu) und handgeschriebene Adjungierte. Die dritte Lösung des
  Vorwärtslaufs ist zugleich die Hilfsgröße für den Gradienten.
- numpy_eintraege.py: Matrixeinträge und ihre Ableitungen in NumPy. Die Originalformeln aus
  components.py werden unverändert wiederverwendet und mit dualen Zahlen abgeleitet.
- harness_stub.py: zeigt, dass simulate als jax.custom_vjp mit jax.pure_callback die drei
  jit-Phasen von benchmark.py übersteht (zusammen ca. 0,5 s).
- Gemessen (Linux, ein Kern): ganze Rechnung ca. 0,2 s (Erstlauf 0,55 s); Einträge identisch zu
  JAX; alle 193 Gradientenwerte auf 1e-13 gleich dem JAX-Pullback; gegen komplettes Original-JAX
  auf 3 Frequenzen 2,4e-6 relativ zum größten Eintrag.

## Aufgabe

Lege den Branch `fast-sparse` an und baue das in differometor/ ein.

1. Baseline sichern: `python benchmark.py --cpus 1 --repeats 3 --output baseline.json
   --save-arrays baseline.npz`. Das braucht ca. 14 GiB RAM und laut README Linux oder WSL2
   (`--cpus 1` nutzt Linux-Prozessaffinität). Wenn die Maschine das nicht schafft, sag es und
   nutze als Vergleich `python prototyp/test_numpy.py . --compare`.
2. df.simulate so umbauen, dass der schnelle Weg über custom_vjp + pure_callback läuft. Der
   bisherige JAX-Code bleibt als Rückfall erhalten und wird genommen, sobald ein Setup verknüpfte
   Parameter, Homodyn-Detektion (qhd) oder mehr als einen Detektor hat.
3. Callbacks ohne Python-Zustand: Was der Rückwärtslauf braucht, als Arrays durch JAX zurückgeben
   und dort neu zerlegen.
4. Danach den Build-Schritt (build.py, run_build_step) von einzelnen jnp-Aufrufen auf NumPy
   umstellen und den Erstlauf profilieren (er ist 0,35 s langsamer als die Folgeläufe).

## Bekannte Stolperstellen

- benchmark.py ruft simulate in jax.jit(jax.vjp(...)) auf, gibt den Pullback aus dem jit zurück
  und wendet ihn in einem zweiten jit an. Die Gradientenprüfung ruft simulate zusätzlich ohne VJP
  auf. Beides muss funktionieren.
- JAX-Kotangenten sind bilinear: für die Signal-Adjungierte A^-T lösen, nicht A^-H.
- 201 Parameterplätze werden aus 193 Werten gefüllt (vier gekoppelte Längengruppen): Beiträge
  addieren. Doppelte Indizes immer mit np.add.at oder np.bincount sammeln.
- Die Signal-Matrix und der Rausch-Vektor hängen von der Carrier-Lösung ab; ohne
  Carrier-Adjungierte ist der Gradient falsch, ohne dass ein Test es merkt.

## Prüfen nach jedem Schritt

- `python benchmark.py --cpus 1 --repeats 3 --output kandidat.json` muss PASS ausgeben.
- Alle 193 Gradientenwerte und carrier/signal/noise gegen baseline.npz vergleichen, relativ zum
  größten Eintrag. benchmark.py prüft nur einen einzigen Gradientenwert.
- Wenn eine Prüfung fehlschlägt: Ursache suchen und melden, keine Toleranz lockern.

## Arbeitsweise und Bericht

Code, Kommentare und Commits auf Englisch. Nach jedem Schritt committen, nicht pushen.
Am Ende berichte: Tabelle vorher/nachher (Erstlauf gesamt, je Phase, warmer Median), größte
Abweichungen bei Sensitivität und Gradient, was du geändert hast und was offen ist. Trenne
dabei, was du gemessen hast und was du nur annimmst.
