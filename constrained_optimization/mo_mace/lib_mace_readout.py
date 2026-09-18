"""
lib_mace_readout.py — MACE-MPA-0 readout descriptors as a linear basis (Constrain_Mb_Elastics.pdf).

ScaleShiftMACE assembles the energy as (mace/modules/models.py)

    E = Σ_atoms E0(Z) + Σ_atoms [ scale·( ZBL pair + r_0(h⁰) + r_1(h¹) ) + shift ]

    r_0 = LinearReadoutBlock     = w₀ · h⁰[0e]            (128 scalars of layer 0)
    r_1 = NonLinearReadoutBlock  = w₁ · σ(W h¹)           (h¹: 128 scalars of layer 1; σ(W h¹): 16)

Two descriptor choices, both exactly linear in δΘ:

  DESCRIPTOR=perez   (default)  D = Σ_atoms [ h⁰[0e], h¹, 1 ]            257 columns
      The scalar INPUT to the readout layer, as in Perez et al., npj Comput. Mater. 11, 263
      (2025), §III.G / Eq. (10) (D_i ∈ R²⁵⁶ for MACE-MPA-0) — the spec's Eq. (2).  The
      correction E_MACE + D·δΘ is an additive linear corrector; the h¹ part is realised by
      wrapping readouts[1] (LinearCorrectedReadout).

  DESCRIPTOR=readout             D = Σ_atoms [ h⁰[0e], σ(W h¹), 1 ]       145 columns
      The input to the LAST linear map of each readout: δΘ is a pure change of the existing
      readout weights (w₀, w₁, shift).  Less expressive (16 layer-1 directions, not 128).

The trailing 1 is a per-atom constant (added to scale_shift.shift); it absorbs the
difference between the DFT set's PBE reference and MPtrj's.  It is not in Perez Eq. (10).

The ZBL pair term is live for Mo (cutoff 2·r_cov = 3.08 Å > NN 2.73 Å) and is frozen: it is
part of E_MACE and of C_MACE, never of D.

Strain convention: cell' = cell·(I + ε)ᵀ with ε symmetric, Voigt ε₄ = 2ε_yz etc.
C_ij = (1/V₀) ∂²E/∂ε_i∂ε_j at zero stress.  1 eV/Å³ = 160.21766 GPa.
"""

import copy
import os

import numpy as np
import torch
from ase.build import bulk

EV_A3_TO_GPA = 160.21766208
MODES = ("perez", "readout")


def descriptor_mode():
    mode = os.environ.get("DESCRIPTOR", "perez")
    assert mode in MODES, f"DESCRIPTOR must be one of {MODES}, got {mode}"
    return mode


# ── model ────────────────────────────────────────────────────────────────────
def load_calc(model="medium-mpa-0", device="cpu"):
    from mace.calculators import mace_mp
    return mace_mp(model=model, default_dtype="float64", device=device)


def _readouts(model):
    ro = model.readouts
    assert len(ro) == 2, f"expected 2 readouts, got {len(ro)}"
    assert hasattr(ro[0], "linear") and hasattr(ro[1], "linear_2"), "unexpected readout blocks"
    assert len(getattr(model, "heads", ["default"])) == 1, "multi-head model not supported"
    n0 = ro[0].linear.irreps_in.count("0e")
    assert str(ro[0].linear.irreps_in).startswith(f"{n0}x0e"), ro[0].linear.irreps_in
    nh = ro[1].linear_1.irreps_in.dim
    assert ro[1].linear_1.irreps_in.count("0e") == nh, "layer-1 readout input must be scalars"
    n1 = ro[1].linear_2.irreps_in.dim
    return ro, n0, nh, n1


def n_columns(model, mode):
    _, n0, nh, n1 = _readouts(model)
    return n0 + (nh if mode == "perez" else n1) + 1


def _probe(lin, n):
    """w_eff such that lin(x) = w_eff · x[:, :n] (unit-vector probe)."""
    X = torch.zeros(n, lin.irreps_in.dim, dtype=torch.float64)
    X[:, :n] = torch.eye(n, dtype=torch.float64)
    with torch.no_grad():
        return lin(X).reshape(n, -1)[:, 0].clone()


def theta0(model):
    """Current Θ₀ for DESCRIPTOR=readout (E_MACE = E_frozen + D·Θ₀)."""
    ro, n0, _, n1 = _readouts(model)
    scale = float(model.scale_shift.scale.reshape(-1)[0])
    shift = float(model.scale_shift.shift.reshape(-1)[0])
    return np.concatenate([scale * _probe(ro[0].linear, n0).numpy(),
                           scale * _probe(ro[1].linear_2, n1).numpy(), [shift]])


# ── descriptors ──────────────────────────────────────────────────────────────
class ReadoutHooks:
    """Captures h⁰[0e], h¹ and σ(W h¹) during a MACE forward."""

    def __init__(self, model, mode):
        self.mode = mode
        self.keep_graph = False
        ro, self.n0, self.nh, self.n1 = _readouts(model)
        self.buf = {}
        self.handles = [
            ro[0].register_forward_pre_hook(self._grab("h0")),
            ro[1].register_forward_pre_hook(self._grab("h1")),
            ro[1].linear_2.register_forward_pre_hook(self._grab("x1")),
        ]

    def _grab(self, key):
        def hook(mod, args):
            self.buf[key] = args[0] if self.keep_graph else args[0].detach()
        return hook

    def remove(self):
        for h in self.handles:
            h.remove()


def energy_and_descriptor(calc, atoms, hooks):
    """E_MACE (eV) and D (summed over atoms) for one structure."""
    at = atoms.copy()
    at.calc = calc
    hooks.buf.clear()
    E = at.get_potential_energy()
    h0 = hooks.buf["h0"][:, : hooks.n0].sum(0).numpy()
    last = hooks.buf["h1"] if hooks.mode == "perez" else hooks.buf["x1"]
    return E, np.concatenate([h0, last.sum(0).numpy(), [len(at)]])


def descriptor_force_jacobian(calc, atoms, hooks):
    """
    G = −∂D/∂r, shape (3N, n_columns), row order atom-major (x₁,y₁,z₁,x₂,…) as in F.ravel().
    The force of the correction is G·δΘ, so the corrected forces are F_MACE + G·δΘ.
    One reverse pass per descriptor column, batched in blocks of JAC_BLOCK columns (default 8).
    The constant column has zero gradient.

    Memory scales with the block size, speed barely does.  Measured on a 54-atom AIMD config
    (identical G to 4e-16 in every case):  B = 8 → 30.6 s, 3.8 GB;  B = 32 → 29.0 s, 11.5 GB;
    B = 128 → 28.5 s, 41.4 GB.  The unblocked 256-column pass OOM-killed a 4-worker job.
    """
    block = int(os.environ.get("JAC_BLOCK", 8))
    model = calc.models[0]
    batch = calc._atoms_to_batch(atoms)
    data = batch.to_dict()
    n = len(atoms)
    hooks.buf.clear()
    hooks.keep_graph = True
    try:
        with torch.enable_grad():
            model(data, training=False, compute_force=False, compute_virials=False,
                  compute_stress=False)
            pos = data["positions"]
            assert pos.requires_grad and pos.shape[0] == n, "unexpected batch (padding?)"
            h0 = hooks.buf["h0"][:, : hooks.n0].sum(0)
            last = (hooks.buf["h1"] if hooks.mode == "perez" else hooks.buf["x1"]).sum(0)
            Dvec = torch.cat([h0, last])                               # (ncol − 1,)
            m = Dvec.numel()
            eye = torch.eye(m, dtype=Dvec.dtype)
            chunks = []
            for lo in range(0, m, block):
                hi = min(lo + block, m)
                try:
                    (Jb,) = torch.autograd.grad(Dvec[lo:hi], pos, grad_outputs=eye[lo:hi, lo:hi],
                                                is_grads_batched=True, retain_graph=True)
                except Exception:
                    Jb = torch.stack([torch.autograd.grad(Dvec[j], pos, retain_graph=True)[0]
                                      for j in range(lo, hi)])
                chunks.append(Jb.detach())
            J = torch.cat(chunks)
    finally:
        hooks.keep_graph = False
        hooks.buf.clear()
    G = np.zeros((3 * n, m + 1))
    G[:, :m] = -J.detach().reshape(m, 3 * n).T.numpy()
    return G


# ── per-config blocks, for a process pool ────────────────────────────────────
_WORKER = {}


def init_worker(mode, threads):
    torch.set_num_threads(max(1, int(threads)))
    calc = load_calc()
    _WORKER.update(calc=calc, hooks=ReadoutHooks(calc.models[0], mode))


def config_blocks(args):
    """
    args = (atoms without calc, E_DFT, F_DFT (N×3) or None).
    Returns E_MACE, D and — if forces given — GᵀG, GᵀΔF, ΔFᵀΔF, 3N with ΔF = F_DFT − F_MACE.
    """
    atoms, E_dft, F_dft = args
    calc, hooks = _WORKER["calc"], _WORKER["hooks"]
    E_mace, D = energy_and_descriptor(calc, atoms, hooks)
    out = {"E_dft": E_dft, "E_mace": E_mace, "D": D, "n": len(atoms)}
    if F_dft is not None:
        x = atoms.copy()
        x.calc = calc
        dF = (np.asarray(F_dft) - x.get_forces()).ravel()
        G = descriptor_force_jacobian(calc, atoms, hooks)
        out.update(GtG=G.T @ G, Gtf=G.T @ dF, ftf=float(dF @ dF), nF=dF.size)
    return out


# ── corrected model ──────────────────────────────────────────────────────────
class LinearCorrectedReadout(torch.nn.Module):
    """readouts[1] + c·h¹ (per node).  `c` is in pre-scale units (δΘ/scale)."""

    def __init__(self, inner, c):
        super().__init__()
        self.inner = inner
        self.register_buffer("c", torch.as_tensor(c, dtype=torch.float64).clone())

    def forward(self, x, heads=None):
        return self.inner(x, heads) + (x[:, : self.c.numel()] @ self.c).unsqueeze(-1)


def _add_to_linear(lin, n, d_eff):
    """Add d_eff to the effective weights of an e3nn 0e→0e Linear (w_eff = α·w_raw)."""
    raw = lin.weight.detach().clone()
    assert raw.numel() == n, "0e→0e block must be the only weight block"
    with torch.no_grad():
        lin.weight.fill_(1.0)
    alpha = _probe(lin, n)
    with torch.no_grad():
        lin.weight.copy_(raw + torch.as_tensor(d_eff, dtype=torch.float64) / alpha)


def apply_correction(model, dtheta, mode):
    """Deep copy of `model` whose energy is exactly E_MACE + D·δΘ."""
    new = copy.deepcopy(model)
    ro, n0, nh, n1 = _readouts(new)
    assert len(dtheta) == n_columns(new, mode)
    scale = float(new.scale_shift.scale.reshape(-1)[0])
    dtheta = np.asarray(dtheta, dtype=np.float64)
    _add_to_linear(ro[0].linear, n0, dtheta[:n0] / scale)
    if mode == "perez":
        new.readouts[1] = LinearCorrectedReadout(ro[1], dtheta[n0 : n0 + nh] / scale)
    else:
        _add_to_linear(ro[1].linear_2, n1, dtheta[n0 : n0 + n1] / scale)
    with torch.no_grad():
        new.scale_shift.shift.add_(float(dtheta[-1]))
    return new


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
    BCC is a Bravais lattice (every atom an inversion centre): no internal relaxation.
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
