"""Feyer "Faster Physics" - Prototyp: duenn besetzter SciPy-Vorwaertslauf + handgeschriebene Adjungierte.

Aufruf (im venv des Hackathon-Repos, Python 3.12, requirements installiert):
    python sparse_prototyp.py /pfad/zu/start-hack-munich-2026            # alle 101 Frequenzen, Check gegen Referenz
    python sparse_prototyp.py /pfad/zu/start-hack-munich-2026 --compare  # 3 Frequenzen, Vergleich mit Original-JAX (alle 193 Gradienten)

Stand: Die Matrixeintraege und ihre Ableitung kommen hier noch aus einer kleinen JAX-Funktion
(entries_fn, ~15 s Kompilierung). Loesen und Adjungierte laufen bereits komplett in NumPy/SciPy.
Offen: entries_fn durch NumPy-Formeln + Ableitungen ersetzen und alles per jax.custom_vjp +
jax.pure_callback in differometor.simulate einhaengen (siehe harness_stub.py).
"""
import sys, os, json, time
if len(sys.argv) < 2: sys.exit(__doc__)
REPO = os.path.abspath(sys.argv[1]); COMPARE = "--compare" in sys.argv[2:]
os.environ["JAX_PLATFORMS"]="cpu"; os.environ["JAX_ENABLE_X64"]="true"
sys.path.insert(0, REPO)
import jax, jax.numpy as jnp, numpy as np
import differometor as df
from differometor.setups import constrain_inter_grid_cell_spaces
jax.config.update("jax_enable_compilation_cache", False)
PROPERTIES = ["reflectivity", "tuning", "db", "angle", "power", "mass", "length"]
def load(nfreq=None, idx=None):
    data=os.path.join(REPO, "examples", "data") + os.sep
    fixture=json.load(open(data+"uifo_800_3000.json")); reference=json.load(open(data+"uifo_800_3000_test_data.json"))
    setup = df.Setup.from_data(fixture)
    pairs = constrain_inter_grid_cell_spaces(setup.parameters, PROPERTIES)
    initial=[]
    for pair in pairs:
        members = pair if isinstance(pair[0], list) else [pair]
        values = [fixture["nodes" if name in fixture["nodes"] else "edges"][name]["properties"][prop] for name, prop in members]
        initial.append(values[0])
    parameters = jnp.asarray(initial, dtype=jnp.float64)
    freqs=np.asarray(reference["frequencies"]); ref=np.asarray(reference["sensitivities"])
    if idx is not None: freqs=freqs[idx]; ref=ref[idx]
    arrays, ports, *rest = df.run_build_step(setup, [("f", "frequency")], jnp.asarray(freqs), pairs)
    return arrays, jnp.asarray(ports), parameters, jnp.asarray(freqs), jnp.asarray(ref), pairs

def readout(outputs, detector_ports, reference):
    carrier, signal, noise = outputs
    signals = df.signal_detector(carrier, signal)[detector_ports]
    detected = signals.squeeze() if detector_ports.shape[0] == 1 else signals[0] - signals[1]
    sensitivity = (noise / jnp.abs(detected)).squeeze()
    loss = jnp.mean(jnp.log10(sensitivity / reference))
    return loss, (sensitivity, df.power_detector(carrier))

from time import perf_counter as pc
from time import perf_counter as pc
from jax import lax, vmap
import scipy.sparse as sp, scipy.sparse.linalg as spl
from differometor.components import FUNCTIONS, UNIT_VACUUM, H_PLANCK, F0
from differometor.build import LINKING_FUNCTIONS
from differometor.simulate import expand_parameters
K = UNIT_VACUUM*H_PLANCK*F0*0.25

def entries_fn(opt, A):
    P = A["parameters"].at[[0], A["optimized_parameter_indices"]].set(opt[A["optimized_value_indices"]])
    out = vmap(lambda i, x: lax.switch(i, LINKING_FUNCTIONS, x))(A["linking_function_indices"], P[0, A["linked_indices"]])
    P = P.at[[0], A["indices_to_link"]].set(out)
    Pc = expand_parameters(P, A["carrier_changing_parameter_indices"], A["carrier_changing_values"])
    Ps = expand_parameters(Pc, A["signal_changing_parameter_indices"], A["signal_changing_values"])
    ev = lambda p, fii, fi, oi: vmap(lambda i, x: lax.switch(i, FUNCTIONS, x))(fi, p[fii])[oi[0], oi[1]]
    c_ent = ev(Pc[0], A["carrier_function_input_indices"], A["carrier_function_indices"], A["carrier_output_indices"])
    s_ent = vmap(lambda p: ev(p, A["signal_function_input_indices"], A["signal_function_indices"], A["signal_output_indices"]))(Ps)
    n_ent = vmap(lambda p: ev(p, A["noise_function_input_indices"], A["noise_function_indices"], A["noise_output_indices"]))(Ps)
    return c_ent, s_ent, n_ent

class Plan:
    """Static symbolic data, computed once in the (timed) build step."""
    def __init__(s, A):
        g = lambda k: np.asarray(A[k])
        s.Cb = g("carrier_matrix"); s.Sb = g("signal_matrix"); s.n = n = s.Cb.shape[0]; s.m = m = s.Sb.shape[0]
        s.cmi = g("carrier_matrix_indices") % np.array([[n],[n+1]])
        smi = g("signal_matrix_indices") % np.array([[m],[m+1]]); pl = g("signal_carrier_placing_indices") % np.array([[m],[m+1]])
        s.sci = g("signal_carrier_indices").ravel()
        s.nmi = g("noise_system_matrix_indices") % m; s.Nb = g("noise_matrix"); assert not s.Nb.any()
        s.sel = g("noise_selection_vectors")[0, 0]; assert g("qhd_parameter_indices").size == 0
        # carrier pattern
        mask = s.Cb[:, :-1] != 0; mask[s.cmi[0][s.cmi[1] < n], s.cmi[1][s.cmi[1] < n]] = True
        s.c_r, s.c_c = np.nonzero(mask.T)[::-1]; s.c_ptr = np.concatenate([[0], np.cumsum(mask.sum(0))])
        # signal pattern (incl. rhs column m)
        mask = s.Sb != 0; mask[smi[0], smi[1]] = True; mask[pl[0], pl[1]] = True
        cc, rr = np.nonzero(mask.T); s.r, s.c = rr, cc; s.Kn = len(rr)
        pos = -np.ones(mask.shape, int); pos[rr, cc] = np.arange(s.Kn)
        s.base = s.Sb[rr, cc]; s.upd = pos[smi[0], smi[1]]; s.plk = pos[pl[0], pl[1]]
        s.cm = ((rr >= n) & (rr < 2*n) & (cc >= n) & (cc < 2*n)) | ((rr >= 2*n) & (cc < n)) | ((rr >= n) & (rr < 2*n) & (cc >= 2*n))
        s.mat = cc < m; s.rhs = ~s.mat
        s.ptr = np.concatenate([[0], np.cumsum(mask[:, :m].sum(0))]); s.mr = rr[s.mat].astype(np.int32); s.mc = cc[s.mat]
        s.rhs_r = rr[s.rhs]

def forward(plan, c_ent, s_ent, n_ent, keep=False):
    s = plan; n, m = s.n, s.m; F = s_ent.shape[0]
    Mc = s.Cb.copy(); Mc[s.cmi[0], s.cmi[1]] = c_ent
    a = sp.csc_matrix((Mc[s.c_r, s.c_c], s.c_r, s.c_ptr), shape=(n, n)); clu = spl.splu(a)
    c = clu.solve(Mc[:, -1])
    V = np.broadcast_to(s.base, (F, s.Kn)).copy(); V[:, s.upd] = s_ent
    pre = V[:, s.plk].copy(); V[:, s.plk] = -pre * c[s.sci]
    V[:, s.cm] = np.conj(V[:, s.cm])
    w = s.sel.copy(); w[:n] = s.sel[:n]*c; w[n:2*n] = np.conj(s.sel[n:2*n]*c)
    X = np.empty((F, m), complex); Y = np.empty((F, m), complex); MU = np.empty((F, m), complex); S = np.empty(F)
    lus = []
    Vm = V[:, s.mat]; B = np.zeros((F, m), complex); B[:, s.rhs_r] = V[:, s.rhs]
    for f in range(F):
        lu = spl.splu(sp.csc_matrix((Vm[f], s.mr, s.ptr), shape=(m, m)))
        X[f] = lu.solve(B[f]); y = lu.solve(w, trans='H'); Y[f] = y
        q = np.zeros(m, complex); np.add.at(q, s.nmi[0], n_ent[f]*y[s.nmi[1]])
        S[f] = (np.conj(y) @ q).real
        qh = np.zeros(m, complex); np.add.at(qh, s.nmi[1], np.conj(n_ent[f])*y[s.nmi[0]])
        MU[f] = lu.solve(0.5*(q+qh))          # = baseline "covariances" (for Hermitian N); reused as adjoint
        if keep: lus.append(lu)
    noise = np.sqrt(2*S*K)
    res = dict(c=c, clu=clu, pre=pre, Y=Y, MU=MU, w=w, lus=lus, V=V)
    return (c[:, None], X.T, noise), res

def backward(plan, res, outs, cts, s_ent_shape):
    s = plan; n, m = s.n, s.m; cbar, xbar, nbar = cts; cbar = cbar[:, 0].astype(complex).copy(); xbar = xbar.T
    c = res["c"]; X = outs[1].T; noise = outs[2]; F = X.shape[0]
    Vbar = np.empty((F, s.Kn), complex); wbar_sum = np.zeros(m, complex)
    g = nbar*K/noise
    r, cc = s.r[s.mat], s.c[s.mat]
    for f in range(F):
        lu = res["lus"][f] if res["lus"] else spl.splu(sp.csc_matrix((res["V"][f, s.mat], s.mr, s.ptr), shape=(m, m)))
        lam = lu.solve(xbar[f].astype(complex), trans='T')
        y, mu = res["Y"][f], res["MU"][f]
        Vbar[f, s.mat] = -lam[r]*X[f, cc] - 2*g[f]*np.conj(y[r])*mu[cc]
        Vbar[f, s.rhs] = lam[s.rhs_r]
        wbar_sum += 2*g[f]*np.conj(mu)
    nbar_ent = g[:, None]*np.conj(res["Y"][:, s.nmi[0]])*res["Y"][:, s.nmi[1]]
    Vbar[:, s.cm] = np.conj(Vbar[:, s.cm])
    np.add.at(cbar, s.sci, -(Vbar[:, s.plk]*res["pre"]).sum(0))
    Vbar[:, s.plk] = -Vbar[:, s.plk]*c[s.sci]
    sbar_ent = Vbar[:, s.upd]
    cbar[:n] += wbar_sum[:n]*s.sel[:n] + np.conj(wbar_sum[n:2*n])*s.sel[n:2*n]
    lamc = res["clu"].solve(cbar, trans='T')
    cbar_ent = np.where(s.cmi[1] == n, lamc[s.cmi[0]], -lamc[s.cmi[0]]*c[np.minimum(s.cmi[1], n-1)])
    return cbar_ent, sbar_ent, nbar_ent


def original_jax(idx):
    """Referenz: unveraenderter Differometor-Pfad mit JAX-Autodiff auf wenigen Frequenzen."""
    arrays, ports, parameters, freqs, ref, pairs = load(idx=idx)
    def loss_fn(p, a, ports, ref):
        out = df.simulate(**dict(a, optimized_parameters=p))
        loss, (sens, powers) = readout(out, ports, ref)
        return loss, (out, sens, powers)
    (loss, (out, sens, powers)), g = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))(parameters, arrays, ports, ref)
    return dict(grad=np.asarray(g), sens=np.asarray(sens), carrier=np.asarray(out[0]), signal=np.asarray(out[1]), noise=np.asarray(out[2]))

if __name__ == "__main__":
    idx = np.array([0, 37, 100]) if COMPARE else None
    if COMPARE:
        print("Rechne Original-JAX auf 3 Frequenzen (dauert wegen Kompilierung) ...", flush=True); tr = original_jax(idx)
    arrays, ports, parameters, freqs, ref, pairs = load(idx=idx)
    t = pc(); plan = Plan(arrays); print("Plan (Besetzungsmuster) %.4f s" % (pc()-t))
    ent_jit = jax.jit(lambda p, A: jax.vjp(lambda q: entries_fn(q, A), p))
    t = pc(); low = ent_jit.lower(parameters, arrays); t1 = pc(); comp = low.compile(); t2 = pc()
    print("Matrixeintraege + Pullback (noch JAX): lower %.2f s, compile %.2f s" % (t1-t, t2-t1))
    ents, pull = comp(parameters, arrays); ents = [np.asarray(e) for e in ents]
    for rep in range(3):
        t = pc(); outs, res = forward(plan, *ents, keep=True); tf = pc()-t
        ro = jax.vjp(lambda o: readout(o, ports, ref), tuple(jnp.asarray(o) for o in outs), has_aux=True)
        loss, ro_pull, (sens, powers) = ro
        cts = [np.asarray(z) for z in ro_pull(jnp.asarray(1.0))[0]]
        t = pc(); ebar = backward(plan, res, outs, cts, None); tb = pc()-t
        print("Vorwaertslauf duenn besetzt %.4f s | Adjungierte %.4f s" % (tf, tb))
    bw = jax.jit(lambda pull, e: pull(e)[0]); t = pc(); grad = np.asarray(bw(pull, tuple(jnp.asarray(e) for e in ebar)))
    print("Pullback der Eintraege (noch JAX, inkl. Kompilierung) %.2f s" % (pc()-t))
    sens = np.asarray(sens); err = np.abs(np.log(sens)-np.log(np.asarray(ref))).max()
    print("max |log sens - log ref| = %.3e  (Grenze 1e-5: %s)" % (err, "OK" if err <= 1e-5 else "VERLETZT"))
    print("Gradient: Norm %.6f, endlich %s, mb11.tuning %.10e" % (np.linalg.norm(grad), np.isfinite(grad).all(), grad[pairs.index(["mb11", "tuning"])]))
    if COMPARE:
        rel = lambda a, b: np.abs(a-b).max()/np.abs(b).max()
        print("gegen Original-JAX: carrier %.2e, signal %.2e, noise %.2e, sens %.2e" % (rel(outs[0], tr["carrier"]), rel(outs[1], tr["signal"]), rel(outs[2], tr["noise"]), rel(sens, tr["sens"])))
        print("Gradient, alle 193 Werte: max. Abweichung relativ zum groessten Eintrag %.3e" % rel(grad, tr["grad"]))
