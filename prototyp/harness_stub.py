"""Stub: bildet die drei jit-Phasen aus benchmark.py nach, mit simulate als jax.custom_vjp + jax.pure_callback.
Die Callbacks geben hier nur gespeicherte Arrays zurueck - gemessen wird allein der JAX-Overhead dieser Struktur.
Aufruf: python harness_stub.py
"""
from time import perf_counter as pc
T0=pc()
import os, sys
os.environ["JAX_PLATFORMS"]="cpu"; os.environ["JAX_ENABLE_X64"]="true"
import jax, jax.numpy as jnp, numpy as np
T1=pc()
jax.config.update("jax_enable_x64", True); jax.config.update("jax_enable_compilation_cache", False); jax.devices("cpu")
T2=pc()
n,m,F,NP=286,751,101,193
rng=np.random.default_rng(0)
OUT=(rng.normal(size=(n,1))+1j*rng.normal(size=(n,1)), rng.normal(size=(m,F))+1j*rng.normal(size=(m,F)), np.abs(rng.normal(size=F))+1)
GRAD=rng.normal(size=NP)
shapes=(jax.ShapeDtypeStruct((n,1),np.complex128), jax.ShapeDtypeStruct((m,F),np.complex128), jax.ShapeDtypeStruct((F,),np.float64))
def host_fwd(p, f): return OUT
def host_bwd(p, f, c, x, nz): return GRAD
@jax.custom_vjp
def _sim(p, freqs):
    return jax.pure_callback(host_fwd, shapes, p, freqs)
def _fwd(p, freqs):
    out=_sim(p,freqs); return out,(p,freqs)
def _bwd(res, ct):
    p,freqs=res
    g=jax.pure_callback(host_bwd, jax.ShapeDtypeStruct((NP,),np.float64), p, freqs, *ct)
    return g, jnp.zeros_like(freqs)
_sim.defvjp(_fwd,_bwd)
def simulate(optimized_parameters, frequencies): return _sim(optimized_parameters, frequencies)
def signal_detector(carrier, signal):
    return jnp.conj(carrier)*signal[:carrier.shape[0]] + carrier*signal[carrier.shape[0]:carrier.shape[0]*2]
def power_detector(s): return 0.5*jnp.abs(s)**2
def forward_with_pullback(parameters, arrays):
    return jax.vjp(lambda p: simulate(**dict(arrays, optimized_parameters=p)), parameters)
def readout(outputs, detector_ports, reference):
    carrier, signal, noise = outputs
    signals = signal_detector(carrier, signal)[detector_ports]
    detected = signals.squeeze() if detector_ports.shape[0] == 1 else signals[0] - signals[1]
    sensitivity = (noise / jnp.abs(detected)).squeeze()
    loss = jnp.mean(jnp.log10(sensitivity / reference))
    return loss, (sensitivity, power_detector(carrier))
def sensitivity_with_pullback(outputs, detector_ports, reference):
    return jax.vjp(lambda values: readout(values, detector_ports, reference), outputs, has_aux=True)
def backward(simulation_pullback, readout_pullback, cotangent):
    return simulation_pullback(readout_pullback(cotangent)[0])[0]
tot={}
def timed(k,fn):
    t=pc(); r=fn(); jax.block_until_ready(r); tot[k]=pc()-t; return r
def compile_phase(k, fn, *v):
    low=timed(k+"_lower", lambda: jax.jit(fn).lower(*v)); return timed(k+"_compile", low.compile)
parameters=jnp.asarray(rng.normal(size=NP)); arrays={"frequencies": jnp.linspace(800.,3000.,F), "optimized_parameters": parameters}
ports=jnp.asarray([279]); expected=jnp.ones(F); one=jnp.asarray(1.0)
fw=compile_phase("forward", forward_with_pullback, parameters, arrays)
outputs, spull = timed("forward", lambda: fw(parameters, arrays))
sf=compile_phase("sensitivity", sensitivity_with_pullback, outputs, ports, expected)
loss, rpull, (sens,powers) = timed("sensitivity", lambda: sf(outputs, ports, expected))
gf=compile_phase("gradient", backward, spull, rpull, one)
grad=timed("gradient", lambda: gf(spull, rpull, one))
host=timed("host", lambda: jax.device_get((outputs, sens, powers, loss, grad)))
assert np.allclose(host[-1], GRAD)
print("import jax+numpy %.3f  init %.3f"%(T1-T0, T2-T1))
for k,v in tot.items(): print("  %-22s %.4f"%(k,v))
print("harness jit overhead total %.3f s"%sum(tot.values()))
