"""
lib_mace_readout.py — MACE-MPA-0 readout as a linear basis (Constrain_Mb_Elastics.pdf).

ScaleShiftMACE assembles the energy as (mace/modules/models.py)

    E = Σ_atoms E0(Z) + Σ_atoms [ scale·( pair + r_0(h⁰) + r_1(h¹) ) + shift ]

    r_0 = LinearReadoutBlock:    w₀ · x₀,   x₀ = 0e channels of the layer-0 node features (128)
    r_1 = NonLinearReadoutBlock: w₁ · x₁,   x₁ = σ(linear_1(h¹))                         (16)

Everything upstream of w₀, w₁ and shift is frozen, so with

    D = Σ_atoms [ x₀ , x₁ , 1 ]                                  (145 columns)
    Θ₀ = [ scale·w₀_eff , scale·w₁_eff , shift ]

E_MACE = E_frozen + D·Θ₀ exactly, and a readout correction δΘ gives E_MACE + D·δΘ
(Eq. 2 of the spec).  `w_eff` is the e3nn Linear's weight times its path normalisation;
it is probed numerically, never assumed.  `check_linear_identity` verifies the identity.

Strain convention: cell' = cell·(I + ε)ᵀ with ε symmetric, Voigt ε₄ = 2ε_yz etc.
C_ij = (1/V₀) ∂²E/∂ε_i∂ε_j at zero stress.  1 eV/Å³ = 160.21766 GPa.
"""

import copy
import numpy as np
import torch
from ase.build import bulk

EV_A3_TO_GPA = 160.21766208


# ── model ────────────────────────────────────────────────────────────────────
def load_calc(model="medium-mpa-0", device="cpu"):
    from mace.calculators import mace_mp
    return mace_mp(model=model, default_dtype="float64", device=device)


def load_calc_from_file(path, device="cpu"):
    from mace.calculators import MACECalculator
    return MACECalculator(model_paths=path, device=device, default_dtype="float64")


def _readout_layers(model):
    ro = model.readouts
    assert len(ro) == 2, f"expected 2 readouts, got {len(ro)}"
    assert hasattr(ro[0], "linear") and hasattr(ro[1], "linear_2"), "unexpected readout blocks"
    assert len(getattr(model, "heads", ["default"])) == 1, "multi-head model not supported"
    return ro[0].linear, ro[1].linear_2


def feature_dims(model):
    lin0, lin2 = _readout_layers(model)
    n0 = lin0.irreps_in.count("0e")
    n1 = lin2.irreps_in.count("0e")
    # the 0e block must come first in lin0's input so x[:, :n0] are the scalars
    assert str(lin0.irreps_in).startswith(f"{n0}x0e"), lin0.irreps_in
    assert lin2.irreps_in.dim == n1
    return n0, n1


def effective_weights(model):
    """w_eff such that Linear(x) = w_eff · x[:, :n] (probed with unit vectors)."""
    lin0, lin2 = _readout_layers(model)
    out = []
    for lin in (lin0, lin2):
        n = lin.irreps_in.count("0e")
        X = torch.zeros(n, lin.irreps_in.dim, dtype=torch.float64)
        X[:, :n] = torch.eye(n, dtype=torch.float64)
        with torch.no_grad():
            out.append(lin(X).reshape(n, -1)[:, 0].clone())
    return out


def theta0(model):
    w0, w1 = effective_weights(model)
    scale = float(model.scale_shift.scale.reshape(-1)[0])
    shift = float(model.scale_shift.shift.reshape(-1)[0])
    return np.concatenate([scale * w0.numpy(), scale * w1.numpy(), [shift]])


# ── descriptors ──────────────────────────────────────────────────────────────
class ReadoutHooks:
    """Captures the inputs of the two final readout linears during a MACE forward."""

    def __init__(self, model):
        self.model = model
        self.n0, self.n1 = feature_dims(model)
        lin0, lin2 = _readout_layers(model)
        self.buf = {}
        self.handles = [
            lin0.register_forward_pre_hook(self._grab("x0")),
            lin2.register_forward_pre_hook(self._grab("x1")),
        ]

    def _grab(self, key):
        def hook(mod, args):
            self.buf[key] = args[0].detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def energy_and_descriptor(calc, atoms, hooks):
    """E_MACE (eV) and D = Σ_atoms [x₀, x₁, 1] for one structure."""
    at = atoms.copy()
    at.calc = calc
    hooks.buf.clear()
    E = at.get_potential_energy()
    x0 = hooks.buf["x0"][:, : hooks.n0].sum(0).numpy()
    x1 = hooks.buf["x1"].sum(0).numpy()
    return E, np.concatenate([x0, x1, [len(at)]])


def check_linear_identity(calc, atoms, hooks, th0=None):
    """E_MACE − D·Θ₀ must equal E with the readouts and shift switched off."""
    model = calc.models[0]
    th0 = theta0(model) if th0 is None else th0
    E, D = energy_and_descriptor(calc, atoms, hooks)
    zeroed = copy.deepcopy(model)
    lin0, lin2 = _readout_layers(zeroed)
    with torch.no_grad():
        lin0.weight.zero_()
        lin2.weight.zero_()
        zeroed.scale_shift.shift.zero_()
    calc.models[0] = zeroed
    try:
        at = atoms.copy()
        at.calc = calc
        E_frozen = at.get_potential_energy()
    finally:
        calc.models[0] = model
    return E - D @ th0 - E_frozen


# ── strain ───────────────────────────────────────────────────────────────────
def voigt_to_matrix(e):
    e1, e2, e3, e4, e5, e6 = e
    return np.array([[e1, e6 / 2, e5 / 2], [e6 / 2, e2, e4 / 2], [e5 / 2, e4 / 2, e3]])


def strained(atoms, voigt):
    at = atoms.copy()
    F = np.eye(3) + voigt_to_matrix(voigt)
    at.set_cell(at.cell.array @ F.T, scale_atoms=True)
    return at


def bcc_cubic(a, element="Mo"):
    return bulk(element, "bcc", a=a, cubic=True)


def strain_derivatives(fn, atoms, h):
    """
    Finite-difference strain derivatives of fn(atoms) (scalar or vector) for a cubic crystal.
    Returns dict: d1 = ∂/∂ε₁, d11 = ∂²/∂ε₁², d12 = ∂²/∂ε₁∂ε₂, d44 = ∂²/∂ε₄².
    BCC has every atom on an inversion centre, so no internal relaxation is needed.
    """
    def f(v):
        return np.asarray(fn(strained(atoms, v)), dtype=np.float64)

    z = [0.0] * 6
    def e(**kw):
        v = list(z)
        for k, val in kw.items():
            v[int(k[1]) - 1] = val
        return v

    f0 = f(z)
    fp1, fm1 = f(e(e1=h)), f(e(e1=-h))
    fpp = f(e(e1=h, e2=h)); fpm = f(e(e1=h, e2=-h))
    fmp = f(e(e1=-h, e2=h)); fmm = f(e(e1=-h, e2=-h))
    fp4, fm4 = f(e(e4=h)), f(e(e4=-h))
    return {
        "d1": (fp1 - fm1) / (2 * h),
        "d11": (fp1 - 2 * f0 + fm1) / h**2,
        "d12": (fpp - fpm - fmp + fmm) / (4 * h**2),
        "d44": (fp4 - 2 * f0 + fm4) / h**2,
    }


def relaxed_bcc_a(calc, a_guess=3.16, element="Mo"):
    """Lattice constant minimising E(a) (5-point quadratic refine, twice)."""
    a = a_guess
    for da in (0.02, 0.002):
        grid = a + da * np.arange(-2, 3)
        Es = []
        for x in grid:
            at = bcc_cubic(x, element)
            at.calc = calc
            Es.append(at.get_potential_energy())
        c = np.polyfit(grid, Es, 2)
        a = -c[1] / (2 * c[0])
    return float(a)


# ── corrected model ──────────────────────────────────────────────────────────
def apply_correction(model, dtheta):
    """Return a deep copy of `model` whose energy is E_MACE + D·δΘ."""
    new = copy.deepcopy(model)
    n0, n1 = feature_dims(new)
    assert len(dtheta) == n0 + n1 + 1
    lin0, lin2 = _readout_layers(new)
    scale = float(new.scale_shift.scale.reshape(-1)[0])
    for lin, d in ((lin0, dtheta[:n0]), (lin2, dtheta[n0 : n0 + n1])):
        # w_eff is linear in the raw weight: find the per-entry normalisation α (w_eff = α·w_raw)
        raw = lin.weight.detach().clone()
        with torch.no_grad():
            lin.weight.fill_(1.0)
        alpha = effective_weights(new)[0 if lin is lin0 else 1]
        assert alpha.numel() == raw.numel(), "0e→0e block must be the only weight block"
        with torch.no_grad():
            lin.weight.copy_(raw + torch.as_tensor(d, dtype=torch.float64) / (scale * alpha))
    with torch.no_grad():
        new.scale_shift.shift.add_(float(dtheta[-1]))
    return new
