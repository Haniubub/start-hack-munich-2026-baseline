"""Test: NumPy-Eintraege + duale Zahlen gegen die JAX-Variante. Aufruf: python test_numpy.py REPO [--compare]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sparse_prototyp as sp
from sparse_prototyp import np, jnp, jax, pc
from numpy_eintraege import EntryPlan

idx = np.array([0, 37, 100]) if sp.COMPARE else None
if sp.COMPARE:
    truth = sp.original_jax(idx)
arrays, ports, parameters, freqs, ref, pairs = sp.load(idx=idx)
A = {k: np.asarray(v) for k, v in arrays.items() if k != "optimized_parameters"}
opt = np.asarray(parameters)

# JAX-Eintraege als Vergleich
ent_jit = jax.jit(lambda p, a: jax.vjp(lambda q: sp.entries_fn(q, a), p))
ents_jax, pull = ent_jit(parameters, arrays)
ents_jax = [np.asarray(e) for e in ents_jax]

for rep in range(3):
    t0 = pc(); plan_e = EntryPlan(A); plan = sp.Plan(arrays); t1 = pc()
    ents, tape = plan_e.entries(opt); t2 = pc()
    outs, res = sp.forward(plan, *ents, keep=True); t3 = pc()
    ro = jax.vjp(lambda o: sp.readout(o, ports, ref), tuple(jnp.asarray(o) for o in outs), has_aux=True)
    loss, ro_pull, (sens, powers) = ro
    cts = [np.asarray(z) for z in ro_pull(jnp.asarray(1.0))[0]]
    t4 = pc(); ebar = sp.backward(plan, res, outs, cts, None); t5 = pc()
    grad = plan_e.pullback(tape, *ebar); t6 = pc()
    print("Plan %.4f | Eintraege %.4f | Loesen %.4f | Adjungierte %.4f | Eintrags-Pullback %.4f | Summe Rechnung %.4f s"
          % (t1 - t0, t2 - t1, t3 - t2, t5 - t4, t6 - t5, (t3 - t0) + (t6 - t4)))

rel = lambda a, b: np.abs(a - b).max() / np.abs(b).max()
print("Eintraege NumPy vs JAX: carrier %.1e, signal %.1e, noise %.1e" % tuple(rel(a, b) for a, b in zip(ents, ents_jax)))
grad_jax = np.asarray(pull(tuple(jnp.asarray(e) for e in ebar))[0])
print("Gradient NumPy-Pullback vs JAX-Pullback (alle %d): max. Abweichung rel. zum groessten Eintrag %.2e, pro Wert max. %.2e"
      % (grad.size, rel(grad, grad_jax), np.max(np.abs(grad - grad_jax) / (np.abs(grad_jax) + 1e-300))))
sens = np.asarray(sens); err = np.abs(np.log(sens) - np.log(np.asarray(ref))).max()
print("max |log sens - log ref| = %.3e (Grenze 1e-5)" % err)
print("Gradient: Norm %.6f, endlich %s, ungleich null %s, mb11.tuning %.10e"
      % (np.linalg.norm(grad), np.isfinite(grad).all(), bool(np.any(grad != 0)), grad[pairs.index(["mb11", "tuning"])]))
if sp.COMPARE:
    print("gegen komplettes Original-JAX (3 Frequenzen): sens %.2e, Gradient %.2e" % (rel(sens, truth["sens"]), rel(grad, truth["grad"])))
