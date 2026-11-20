"""Matrixeintraege und ihre Ableitungen in reinem NumPy - ohne die 19 Bauteil-Formeln neu zu schreiben.

Idee: Die Originalfunktionen aus differometor/components.py werden unveraendert wiederverwendet.
Sie bekommen nur ein anderes `jnp` untergeschoben, das mit dualen Zahlen (Wert + Ableitung) auf
NumPy-Arrays rechnet. Dadurch entfallen XLA-Kompilierung UND handgeschriebene Ableitungen.

    plan = EntryPlan(arrays)                      # einmal, im (gemessenen) Build-Schritt
    ents, tape = plan.entries(opt)                # Eintraege fuer Carrier, Signal, Noise
    grad = plan.pullback(tape, cbar, sbar, nbar)  # Kotangenten der Eintraege -> Gradient (193,)
"""
import types
import numpy as np
import differometor.components as comp


class Dual:
    """Wert plus Ableitungen nach k reellen Eingaengen; tan hat Form (k, *val.shape) oder ist 0."""
    __slots__ = ("val", "tan")
    __array_priority__ = 1000

    def __init__(self, val, tan=0.0):
        self.val, self.tan = val, tan

    @staticmethod
    def lift(x):
        return x if isinstance(x, Dual) else Dual(x, 0.0)

    def __neg__(self): return Dual(-self.val, -self.tan)
    def __add__(self, o): o = Dual.lift(o); return Dual(self.val + o.val, self.tan + o.tan)
    __radd__ = __add__
    def __sub__(self, o): o = Dual.lift(o); return Dual(self.val - o.val, self.tan - o.tan)
    def __rsub__(self, o): o = Dual.lift(o); return Dual(o.val - self.val, o.tan - self.tan)
    def __mul__(self, o): o = Dual.lift(o); return Dual(self.val * o.val, self.tan * o.val + self.val * o.tan)
    __rmul__ = __mul__
    def __truediv__(self, o):
        o = Dual.lift(o); q = self.val / o.val
        return Dual(q, (self.tan - q * o.tan) / o.val)
    def __rtruediv__(self, o): return Dual.lift(o).__truediv__(self)
    def __pow__(self, n):
        assert not isinstance(n, Dual)
        return Dual(self.val ** n, n * self.val ** (n - 1) * self.tan)


def _unary(f, df):
    def g(x):
        if not isinstance(x, Dual):
            return f(x)
        y = f(x.val)
        return Dual(y, df(x.val, y) * x.tan)
    return g


class _Shim:
    """Ersatz fuer `jax.numpy` innerhalb der Bauteil-Funktionen."""
    pi, e, complex128 = np.pi, np.e, np.complex128
    log10, log = staticmethod(np.log10), staticmethod(np.log)
    exp = staticmethod(_unary(np.exp, lambda x, y: y))
    sqrt = staticmethod(_unary(np.sqrt, lambda x, y: 0.5 / y))
    sin = staticmethod(_unary(np.sin, lambda x, y: np.cos(x)))
    cos = staticmethod(_unary(np.cos, lambda x, y: -np.sin(x)))
    sinh = staticmethod(_unary(np.sinh, lambda x, y: np.cosh(x)))
    cosh = staticmethod(_unary(np.cosh, lambda x, y: np.sinh(x)))
    arcsin = staticmethod(_unary(np.arcsin, lambda x, y: 1.0 / np.sqrt(1.0 - x * x)))
    radians = staticmethod(_unary(np.radians, lambda x, y: np.pi / 180.0))

    @staticmethod
    def conj(x):  # gueltig, weil alle Eingaenge reell sind
        return Dual(np.conj(x.val), np.conj(x.tan)) if isinstance(x, Dual) else np.conj(x)
    conjugate = conj

    @staticmethod
    def abs(x):
        if not isinstance(x, Dual):
            return np.abs(x)
        a = np.abs(x.val)
        return Dual(a, (np.conj(x.val) * x.tan).real / a)

    @staticmethod
    def array(items, dtype=None):
        return list(items)


class _Out(list):
    """Ausgabevektor eines Bauteils; unterstuetzt das Vorzeichen `-standardize_output(...)`."""
    def __neg__(self):
        return _Out(-x for x in self)


def _rebind():
    """Kopien der Originalfunktionen, deren globales `jnp` der Shim ist."""
    g = dict(vars(comp))
    g["jnp"] = _Shim
    g["standardize_output"] = lambda out: _Out(list(out) + [0.0] * (4 - len(out)))
    for name, f in vars(comp).items():
        if isinstance(f, types.FunctionType) and f.__module__ == comp.__name__ and name != "standardize_output":
            g[name] = types.FunctionType(f.__code__, g, name, f.__defaults__, f.__closure__)
    return [g[f.__name__] for f in comp.FUNCTIONS]


FUNCTIONS_NP = _rebind()
_SEED = np.eye(7)


def _evaluate(fidx, X):
    """X: (..., rows, 7) -> Werte (..., rows, 4) und Jacobi (..., rows, 4, 7)."""
    lead = X.shape[:-1]
    params = [Dual(X[..., j], _SEED[j].reshape((7,) + (1,) * len(lead))) for j in range(7)]
    out = FUNCTIONS_NP[fidx](params)
    val = np.empty(lead + (4,), complex)
    jac = np.zeros(lead + (4, 7), complex)
    for o, item in enumerate(out):
        if isinstance(item, Dual):
            val[..., o] = item.val
            if not np.isscalar(item.tan):
                jac[..., o, :] = np.moveaxis(np.broadcast_to(item.tan, (7,) + lead), 0, -1)
        else:
            val[..., o] = item
    return val, jac


class _Block:
    """Eine der drei Eintragsgruppen (Carrier, Signal, Noise)."""

    def __init__(self, A, name, out_key):
        self.fii = np.asarray(A[name + "_function_input_indices"])
        fi = np.asarray(A[name + "_function_indices"])
        self.oi = np.asarray(A[out_key])
        self.groups = [(int(k), np.nonzero(fi == k)[0]) for k in np.unique(fi)]

    def run(self, P):
        """P: (R, n_par). Liefert Eintraege (R, n_ent) und das Band fuer den Rueckweg."""
        R, calls = P.shape[0], self.fii.shape[0]
        val = np.empty((R, calls, 4), complex)
        jac = np.empty((R, calls, 4, 7), complex)
        for k, rows in self.groups:
            val[:, rows], jac[:, rows] = _evaluate(k, P[:, self.fii[rows]])
        return val[:, self.oi[0], self.oi[1]], jac

    def back(self, jac, ebar, n_par):
        """ebar: (R, n_ent) -> Kotangente der Parameter (R, n_par), reell."""
        R, calls = jac.shape[:2]
        outbar = np.zeros((R, calls * 4), complex)
        flat = self.oi[0] * 4 + self.oi[1]
        for r in range(R):
            outbar[r] = np.bincount(flat, ebar[r].real, calls * 4) + 1j * np.bincount(flat, ebar[r].imag, calls * 4)
        inbar = np.einsum("rco,rcoj->rcj", outbar.reshape(R, calls, 4), jac).real
        Pbar = np.zeros((R, n_par))
        cols = self.fii.ravel()
        for r in range(R):
            Pbar[r] = np.bincount(cols, inbar[r].ravel(), n_par)
        return Pbar


class EntryPlan:
    def __init__(self, A):
        g = lambda k: np.asarray(A[k])
        self.base = g("parameters")[0].real.copy()
        assert not g("parameters").imag.any()
        self.opi, self.ovi = g("optimized_parameter_indices"), g("optimized_value_indices")
        # Nur der triviale Fall "keine verknuepften Parameter" wird hier abgedeckt.
        assert g("linked_indices").tolist() == [0] and g("indices_to_link").tolist() == [0], "verknuepfte Parameter: Originalpfad nutzen"
        self.cci, self.ccv = g("carrier_changing_parameter_indices"), g("carrier_changing_values").real
        self.sci, self.scv = g("signal_changing_parameter_indices"), g("signal_changing_values").real
        self.carrier = _Block(A, "carrier", "carrier_output_indices")
        self.signal = _Block(A, "signal", "signal_output_indices")
        self.noise = _Block(A, "noise", "noise_output_indices")
        self.n_opt = int(self.ovi.max()) + 1

    def entries(self, opt):
        P = self.base.copy()
        P[self.opi] = np.asarray(opt, float)[self.ovi]
        Pc = np.tile(P, (self.ccv.shape[1], 1)); Pc[:, self.cci] = self.ccv.T
        Ps = np.tile(Pc[0], (self.scv.shape[1], 1)); Ps[:, self.sci] = self.scv.T
        c, jc = self.carrier.run(Pc[:1])
        s, js = self.signal.run(Ps)
        n, jn = self.noise.run(Ps)
        return (c[0], s, n), (jc, js, jn)

    def pullback(self, tape, cbar, sbar, nbar):
        jc, js, jn = tape
        n_par = self.base.size
        Pbar_s = self.signal.back(js, sbar, n_par) + self.noise.back(jn, nbar, n_par)
        Pbar_s[:, self.sci] = 0.0                      # von der Frequenz ueberschriebene Parameter
        Pbar = self.carrier.back(jc, cbar[None], n_par)[0] + Pbar_s.sum(0)
        Pbar[self.cci] = 0.0
        return np.bincount(self.ovi, Pbar[self.opi], self.n_opt)
