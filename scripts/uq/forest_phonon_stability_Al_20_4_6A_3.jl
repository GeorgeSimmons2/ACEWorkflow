# forest_phonon_stability_Al_20_4_6A_3.jl
#
# QUESTION.  different_filter.jl draws its committee from a hypercube fitted to the
# POPS delta forest.  Its samples contain dynamically-unstable members (2/50 real,
# after unstable_recheck_native showed 3/5 of the flagged ones were 3x3x3 periodic-
# image artefacts).  Did that instability come from the FOREST the box was fitted
# to, or is the box itself manufacturing it by reaching into corners no actual
# correction occupies?
#
#   forest clean, samples unstable   -> box-corner artefact (percentile_clipping=0,
#                                       and a d-dimensional box is nearly all corner)
#   comparable unstable fractions    -> inherited from the cloud; the filter is
#                                       selecting genuinely soft corrections
#
# Both clouds are therefore evaluated here on the SAME 4x4x4 band path, which also
# closes a gap left by the recheck: only the 5 flagged samples were ever re-tested
# at a converged cell size, and contamination is not one-directional -- it inflated
# three members, but could equally have masked a soft one among the other 45.
#
# ── METHOD ───────────────────────────────────────────────────────────────────
#
# Two approximations are removed relative to different_filter.jl:
#
#   N_cell 3 -> 4.  The 3x3x3 cubic cell is 12.06 A, box half-width 6.03 A against
#   a 6 A cutoff: atoms very nearly interact with their own images.  4x4x4 gives
#   8.04 A of clearance.
#
#   fixed a_eq -> each member at its own.  The forest members are not a_eq-pinned,
#   so a spectrum at the mean model's lattice constant carries residual stress.
#   Handled quasiharmonically: D(q; a) ~= D(q; a_ref) + (a - a_ref) dD/da.
#
# WHY THE QHA SHIFT IS WELL-POSED.  fcc_band_path builds q from FIXED FRACTIONAL
# high-symmetry points times B = 2*pi*inv(L'), so under uniform scaling a -> a' the
# Cartesian q scales as 1/a while R scales as a: the Bloch phase q.R is invariant.
# At fixed fractional q, dD/da is therefore just the Bloch transform of dPhi/da,
# and a finite difference between two builds is a clean derivative.  (Same pattern
# as the existing grun_m / grun_p caches.)
#
# ── WHY THIS DOES NOT BUILD THE FULL UNDOTTED HESSIAN ────────────────────────
#
# A full 4x4x4 undotted H_basis at 1829 parameters is 768^2 x 1829 x 8 = 8.63 GB,
# and lib.jl's undotted_hessian() allocates one such buffer PER THREAD -- 69 GB at
# 8 threads.  That is why every Al_20 cache on disk is 3x3x3 while Al_12 (91 params,
# 429 MB) has dozens at 4x4x4.
#
# It is not needed.  FCC has a ONE-ATOM primitive cell (Np = 1; different_filter.jl
# printed "3 branches"), and dynamical_matrix_from_fc reads only
#
#     H[3*p2s_map[i+1] + alpha, 3k + beta]        (phonon_bands.jl:142)
#
# which for Np = 1 is THREE ROWS of the 768x768 supercell Hessian.  Those rows are
# touched only by site Hessians of atoms i having r in their neighbour list, and
# neighbour lists are symmetric, so the contributing set is {r} U neighbours(r) --
# about 57 of the 256 atoms.
#
#     memory          8.63 GB (x threads)  ->  33.7 MB, no per-thread copy
#     site Hessians   256                  ->  ~57
#
# so a 4x4x4 build is CHEAPER than the existing 3x3x3 full build (57 vs 108 site
# Hessians).  Only Bq (37 MB) is cached, since nothing downstream reads anything else.
#
# ── VALIDATION (all three run automatically) ─────────────────────────────────
#   1. undotted vs native at 4x4x4.  The 3x3x3 control failed by up to 0.20 THz
#      (unstable_recheck_native), in a way that tracked cell-size sensitivity
#      exactly -- consistent with cutoff marginality rather than a bug.  This
#      settles it at the size that matters, and validates the 3-row build.
#   2. QHA-shifted vs a full rebuild at that member's own a, for the members with
#      the LARGEST |da| (worst case, so it bounds the error).
#   3. Newton da vs relax_lattice_constant, for a random sample.
#
# Run (REPL, with pwise already in scope):   include("scripts/uq/forest_phonon_stability_Al_20_4_6A_3.jl")
# Run (batch):                               julia --project -t 8 scripts/uq/forest_phonon_stability_Al_20_4_6A_3.jl [n_members]
#
#   n_members > 0 processes only the first n by index as a pilot.  NOTE: forest
#   members are ordered by observation index, not randomly, so a pilot is a biased
#   sample of both cost and stability -- do not extrapolate a rate from it.

using ACEWorkflow        # load_model, bulk_prim_super, precompute_force_constants,
                         # fcc_band_path, relax_lattice_constant, FREQ_THz, Elasticity
using ACEpotentials      # Models.set_linear_parameters!, Models.potential_energy_basis
import ACEpotentials.Models: evaluate_basis
import AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius, hessian
using StaticArrays, ForwardDiff, Unitful
using LinearAlgebra, Statistics, Random, Serialization, DelimitedFiles, Printf
using CairoMakie

Random.seed!(1234)

element        = :Al
N_cell         = 4                        # 8.04 A clearance vs the 6 A cutoff
N_per_seg      = [20, 20, 20, 20, 60]     # identical path to different_filter.jl
unstable_tol   = -0.05                    # THz
qΓtol          = 5e-2
δa_fd          = 0.02                     # Å, forward step for dD/da (2-point linear)
n_qha_check    = 3                        # members re-built exactly to bound QHA error
n_relax_check  = 20                       # members relaxed exactly to check the Newton step
n_plot         = 300                      # forest members drawn on the band figure
n_members_arg  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0

# ── model, WITHOUT the design matrix ─────────────────────────────────────────
# load_model() reads A.csv — 5.2 GB of text into a 2.15 GB Matrix — and nothing in
# this script touches A.  Load the potential straight from its json instead.
MODELDIR   = "models/Al_20_4_6A_3_"
model, _   = ACEpotentials.load_model("$MODELDIR/Al_20_4_6A_3.json")
lin_params = vec(readdlm("$MODELDIR/lin_params.csv", ','))
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
n_params   = length(lin_params)
result     = (; dir = MODELDIR, name = "Al_20_4_6A_3", lin_params, model)
RES        = "$MODELDIR/results"
outdir     = "$RES/forest_phonon_stability"; mkpath(outdir)
FOREST_CSV = "$MODELDIR/pops_corrections_different_filter.csv"
FOREST_JLS = "$MODELDIR/pops_corrections_different_filter.jls"

# Threads used for the Hessian BUILD are capped separately from Threads.nthreads():
# each build thread holds a site Hessian of N_basis × (3nR)² Float64 ≈ 0.4 GB plus
# ForwardDiff dual-number overhead, so 40 build threads needs ~60-80 GB and gets OOM-
# killed.  The build saturates at 55 contributing atoms anyway.  Everything else
# (Bloch transform, the 73k sweep) still uses all threads.
const BUILD_THREADS = parse(Int, get(ENV, "BUILD_THREADS", "8"))

@printf("Model %s: %d params | %d threads (%d for Hessian builds)\n",
        result.name, n_params, Threads.nthreads(), BUILD_THREADS)

# ── the delta forest ─────────────────────────────────────────────────────────
# Prefer whatever is already in the REPL (pwise): re-deriving it costs the 5.2 GB
# A.csv read plus the C\X' solve.  It is deterministic (no RNG in the filter), so
# an in-memory copy and a recomputed one are identical.
forest = if @isdefined(pwise)
    println("using `pwise` from the current session")
    pwise isa Matrix{Float64} ? pwise : Matrix{Float64}(pwise)
elseif isfile(FOREST_JLS)
    println("reading $FOREST_JLS")          # binary: ~1.1 GB, no parse, no copy
    deserialize(FOREST_JLS)
elseif isfile(FOREST_CSV)
    # readdlm on 2.96 GB of text needs several GB of transient parse buffers on top
    # of the 1.07 GB result, so convert once and use the binary from then on.
    @printf("reading %s (%.2f GB of text — slow, single-threaded) …\n",
            FOREST_CSV, filesize(FOREST_CSV)/1e9); flush(stdout)
    f = readdlm(FOREST_CSV, ',', Float64)
    println("  caching as $FOREST_JLS for future runs"); serialize(FOREST_JLS, f)
    f
else
    error("no forest available: `pwise` is not defined and neither\n" *
          "  $FOREST_JLS nor $FOREST_CSV exists.\nRun different_filter.jl first.")
end
size(forest, 2) == n_params ||
    error("forest has $(size(forest,2)) columns, model has $n_params parameters")
N_forest = size(forest, 1)
@printf("delta forest: %d members x %d params (%.2f GB)\n", N_forest, n_params, sizeof(forest)/1e9)

if @isdefined(pwise) && !isfile(FOREST_CSV)
    @printf("writing %s (~%.1f GB of text, this takes a few minutes) …\n",
            FOREST_CSV, N_forest*n_params*20/1e9); flush(stdout)
    writedlm(FOREST_CSV, forest, ',')
    println("  done")
end

sel = n_members_arg > 0 ? (1:min(n_members_arg, N_forest)) : (1:N_forest)
n_members_arg > 0 && @warn "PILOT: first $(length(sel)) members by index — biased, do not extrapolate a rate"

# ═════════════════════════════════════════════════════════════════════════════
#  3-row undotted band-path builder
# ═════════════════════════════════════════════════════════════════════════════
function _site_basis_hessian(m, Rs, Zs, z0, ps, st)
    nR = length(Rs)
    x0 = collect(Float64, reinterpret(Float64, Rs))
    tovec(x) = [SVector{3,eltype(x)}(x[3i-2], x[3i-1], x[3i]) for i in 1:nR]
    Bfun(x)  = evaluate_basis(m, tovec(x), Zs, z0, ps, st)
    Hflat    = ForwardDiff.jacobian(x -> vec(ForwardDiff.jacobian(Bfun, x)), x0)
    return reshape(Hflat, length(Bfun(x0)), 3nR, 3nR)
end

"""
Rows `3r-2 : 3r` of the undotted per-basis supercell Hessian, where `r` is the
supercell index of the primitive-cell representative atom.  Returns 3 × 3Nat × NB.

Only atoms in {r} ∪ neighbours(r) contribute (neighbour lists are symmetric), and
for i ≠ r only the neighbour slots with j1 == r do any work — the rest of the
scatter writes into rows we are discarding.
"""
function undotted_rows(sys_super, V, r)
    nlist = PairList(sys_super, cutoff_radius(V))
    Nat   = length(sys_super); D = 3
    ps, st, m = V.ps, V.st, V.model
    _, Rs0, Zs0, z00 = get_neighbours(sys_super, V, nlist, 1)
    NB = length(evaluate_basis(m, Rs0, Zs0, z00, ps, st))

    Js_r, = get_neighbours(sys_super, V, nlist, r)
    contributors = unique(vcat(r, Js_r))
    nR_ref = length(Js_r)
    hi_gb  = NB * (3nR_ref)^2 * 8 / 1e9
    nt     = clamp(BUILD_THREADS, 1, min(Threads.nthreads(), length(contributors)))
    @printf("    %d of %d atoms contribute; %d build threads, %d deep\n",
            length(contributors), Nat, nt, cld(length(contributors), nt))
    @printf("    site Hessian %.2f GB/thread → ~%.0f-%.0f GB peak (accumulators %.2f GB)\n",
            hi_gb, nt*hi_gb, 2*nt*hi_gb, nt*D*D*Nat*NB*8/1e9)
    flush(stdout)

    # Bounded concurrency with explicit per-task buffers: `Threads.@threads` would
    # spawn one task per thread and one accumulator per thread, and at 40 threads the
    # transient site Hessians alone are ~60-80 GB.  Round-robin so the chunks balance.
    bufs = [zeros(D, D*Nat, NB) for _ in 1:nt]
    @sync for t in 1:nt
        Threads.@spawn begin
            Hl = bufs[t]
            for i in contributors[t:nt:end]
                Js, Rs, Zs, z0 = get_neighbours(sys_super, V, nlist, i)
                Hi = _site_basis_hessian(m, Rs, Zs, z0, ps, st)
                Ji = (i-1)*D .+ (1:D)
                is_ref = (i == r)
                for (α1, j1) in enumerate(Js)
                    hit1 = (j1 == r)
                    (hit1 || is_ref) || continue   # rows of r untouched by this slot
                    A1 = (α1-1)*D .+ (1:D)
                    for (α2, j2) in enumerate(Js)
                        A2 = (α2-1)*D .+ (1:D)
                        J2 = (j2-1)*D .+ (1:D)
                        @views for k in 1:NB
                            blk = Hi[k, A1, A2]
                            if hit1                # H[J1,J2] += blk ; H[J1,Ji] -= blk
                                Hl[:, J2, k] .+= blk
                                Hl[:, Ji, k] .-= blk
                            end
                            if is_ref              # H[Ji,J2] -= blk ; H[Ji,Ji] += blk
                                Hl[:, J2, k] .-= blk
                                Hl[:, Ji, k] .+= blk
                            end
                        end
                    end
                end
                Hi = nothing                        # release before the next atom
            end
        end
    end
    H = bufs[1]
    for t in 2:nt; H .+= bufs[t]; bufs[t] = zeros(0,0,0); end   # free as we reduce
    return H
end

"""
Band-path per-basis dynamical matrices from the 3 reference rows.  Mirrors
`dynamical_matrix_from_fc` exactly (minimum-image wrapping in supercell fractional
coords, Hermitian symmetrisation), specialised to Np = 1.
"""
function bandpath_rows(model, element, a, N_cell; N_per_seg=N_per_seg)
    sys_prim, sys_super = bulk_prim_super(element; a=a, N_cell=N_cell)
    fc = precompute_force_constants(sys_prim, sys_super, model)   # geometry + maps (+ native H)
    fc.Np == 1 || error("this builder assumes a 1-atom primitive cell; got Np = $(fc.Np)")
    r  = fc.p2s_map[1] + 1                                        # p2s_map is 0-based
    Ns = fc.Ns

    @printf("  building 3-row undotted Hessian at a = %.5f Å (%d atoms) …\n", a, length(sys_super))
    t = @elapsed Hrows = undotted_rows(sys_super, model, r)
    @printf("    done in %.1f min\n", t/60); flush(stdout)

    q_list, x_vals, x_ticks, labels, _ = fcc_band_path(fc.L; N_per_seg=N_per_seg)
    NB = size(Hrows, 3)
    Bq = Vector{Matrix{ComplexF64}}(undef, length(q_list))

    # minimum-image lattice vector from the origin-cell atom to supercell atom k
    Rmi = Vector{SVector{3,Float64}}(undef, Ns)
    for k in 1:Ns
        R_cart  = fc.L * (fc.frac_super[k] - fc.frac_prim[1])
        R_sfrac = fc.Linv_super * R_cart
        R_sfrac = R_sfrac .- round.(R_sfrac)
        Rmi[k]  = fc.L_super * R_sfrac
    end
    mass = fc.masses[1]

    Threads.@threads for iq in eachindex(q_list)
        q = q_list[iq]
        M = zeros(ComplexF64, 9, NB)
        for k in 1:Ns
            eph = exp(im * dot(q, Rmi[k]))
            cols = 3(k-1) .+ (1:3)
            @views for kb in 1:NB
                blk = Hrows[:, cols, kb]           # 3×3
                for β in 1:3, α in 1:3
                    M[3(β-1)+α, kb] += blk[α, β] * eph
                end
            end
        end
        M ./= mass
        # Hermitian symmetrisation, as dynamical_matrix_from_fc does
        for kb in 1:NB
            D3 = reshape(@view(M[:, kb]), 3, 3)
            H3 = (D3 .+ D3') ./ 2
            M[:, kb] = vec(H3)
        end
        Bq[iq] = M
    end
    return (; Bq, q_list, x_vals, x_ticks, labels, Np=fc.Np,
              qnorm=norm.(q_list), Hrows, r, sys_super, a)
end

bands_from(Bq, θ, qn) = begin
    F = Matrix{Float64}(undef, 3, length(Bq))
    for iq in eachindex(Bq)
        ev = eigvals(Hermitian(reshape(Bq[iq]*θ, 3, 3)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    F
end

# min ω away from Γ, for D = Bq0 + s*dBq  (s = a − a_ref; s = 0 gives the base build)
function min_freq_shift(Bq0, dBq, θ, qn, s; qΓtol=qΓtol)
    m = Inf
    for iq in eachindex(Bq0)
        qn[iq] < qΓtol && continue
        v = Bq0[iq]*θ
        s != 0 && (v = v .+ s .* (dBq[iq]*θ))
        ev = eigvals(Hermitian(reshape(v, 3, 3)))
        m = min(m, minimum(sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz))
    end
    return m
end

# ═════════════════════════════════════════════════════════════════════════════
#  builds
# ═════════════════════════════════════════════════════════════════════════════
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
a_ref = ACEWorkflow.relax_lattice_constant(model, element)
@printf("\na_ref (mean model) = %.5f Å;  box = %.2f Å, half-box = %.2f Å vs cutoff 6 Å\n",
        a_ref, N_cell*a_ref, N_cell*a_ref/2); flush(stdout)

bp0 = bandpath_rows(model, element, a_ref,          N_cell)
bpp = bandpath_rows(model, element, a_ref + δa_fd,  N_cell)
dBq = [(bpp.Bq[iq] .- bp0.Bq[iq]) ./ δa_fd for iq in eachindex(bp0.Bq)]
@printf("band path: %d q-points, %d branches; dD/da by forward difference at δa = %.3f Å\n",
        length(bp0.Bq), 3bp0.Np, δa_fd); flush(stdout)

# Cached HERE rather than at the end of the run: these two builds are the expensive
# part (~20 min each) and everything downstream — including the pinned-forest script —
# only needs Bq and dBq.  Writing it now means a later failure costs nothing to redo.
serialize("$outdir/bandpath_4x4x4.jls",
          (; Bq=bp0.Bq, dBq, q_list=bp0.q_list, x_vals=bp0.x_vals, x_ticks=bp0.x_ticks,
             labels=bp0.labels, qnorm=bp0.qnorm, a_ref, δa_fd, N_cell, Np=bp0.Np))
println("band-path cache → $outdir/bandpath_4x4x4.jls (37 MB, reusable)"); flush(stdout)

# ── VALIDATION 1: undotted vs native at 4x4x4 ────────────────────────────────
println("\n── validation 1: undotted vs native Hessian rows at 4×4×4 ──")
rows_r = 3*(bp0.r - 1) .+ (1:3)
val1 = NamedTuple[]
for (tag, θ) in (("mean", lin_params),
                 ("forest member 1", lin_params .+ forest[1, :]),
                 ("forest member $(N_forest)", lin_params .+ forest[N_forest, :]))
    H_und = reshape(reshape(bp0.Hrows, :, n_params) * θ, 3, :)
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    H_nat = ustrip.(hessian(bp0.sys_super, model))[rows_r, :]
    d = maximum(abs.(H_und .- H_nat)); s = maximum(abs.(H_nat))
    @printf("  %-22s max|Δ| = %.3e  (%.2e relative)  %s\n", tag, d, d/s,
            d/s < 1e-8 ? "✓ agree" : "← DISAGREE")
    push!(val1, (; tag, rel = d/s))
end
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
val1_ok = all(v -> v.rel < 1e-8, val1)
val1_ok || @warn "undotted and native Hessians disagree at 4×4×4 — the 3×3×3 discrepancy was NOT cutoff marginality; treat everything below as suspect"
flush(stdout)

# ── per-member equilibrium: Newton step on E(a) = b(a)·θ ─────────────────────
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
                       ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_ref)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_ref)

# ═════════════════════════════════════════════════════════════════════════════
#  sweep
# ═════════════════════════════════════════════════════════════════════════════
n_sel = length(sel)
minω_ref = fill(NaN, n_sel)      # at a_ref
minω_own = fill(NaN, n_sel)      # QHA-shifted to the member's own a
Δa_all   = fill(NaN, n_sel)
println("\n── sweeping $(n_sel) forest members ──"); flush(stdout)
t_sweep = @elapsed Threads.@threads for t in 1:n_sel
    θ  = lin_params .+ @view(forest[sel[t], :])
    K  = dot(b_double_prime, θ)
    Δa = K > 0 ? -dot(b_prime, θ)/K : NaN      # K ≤ 0: no minimum, Newton meaningless
    Δa_all[t]   = Δa
    minω_ref[t] = min_freq_shift(bp0.Bq, dBq, θ, bp0.qnorm, 0.0)
    minω_own[t] = isnan(Δa) ? NaN : min_freq_shift(bp0.Bq, dBq, θ, bp0.qnorm, Δa)
end
@printf("  swept in %.1f s (%.2f ms/member)\n", t_sweep, 1000*t_sweep/n_sel); flush(stdout)

n_badK = count(isnan, Δa_all)
n_unst_ref = count(x -> !isnan(x) && x < unstable_tol, minω_ref)
n_unst_own = count(x -> !isnan(x) && x < unstable_tol, minω_own)
ok = .!isnan.(Δa_all)

@printf("\nforest Δa            : [%+.5f, %+.5f] Å, median %+.5f  (%d members with b″·θ ≤ 0)\n",
        minimum(Δa_all[ok]), maximum(Δa_all[ok]), median(Δa_all[ok]), n_badK)
@printf("forest min ω @a_ref  : [%+.4f, %+.4f] THz, median %+.4f — %d/%d unstable (%.3f%%)\n",
        minimum(minω_ref), maximum(minω_ref), median(minω_ref), n_unst_ref, n_sel, 100n_unst_ref/n_sel)
@printf("forest min ω @a_own  : [%+.4f, %+.4f] THz, median %+.4f — %d/%d unstable (%.3f%%)\n",
        minimum(minω_own[ok]), maximum(minω_own[ok]), median(minω_own[ok]), n_unst_own, n_sel, 100n_unst_own/n_sel)
flush(stdout)

# ── the hypercube samples on the SAME footing (closes the 45-untested gap) ───
samples = Matrix(readdlm("$RES/different_filter/samples.csv", ',')')      # p × 50
n_s = size(samples, 2)
s_ref = Float64[]; s_own = Float64[]; s_Δa = Float64[]
for j in 1:n_s
    θ = samples[:, j]; K = dot(b_double_prime, θ)
    Δa = K > 0 ? -dot(b_prime, θ)/K : NaN
    push!(s_Δa, Δa)
    push!(s_ref, min_freq_shift(bp0.Bq, dBq, θ, bp0.qnorm, 0.0))
    push!(s_own, isnan(Δa) ? NaN : min_freq_shift(bp0.Bq, dBq, θ, bp0.qnorm, Δa))
end
n_s_unst = count(x -> !isnan(x) && x < unstable_tol, s_own)
@printf("\nhypercube samples (all %d, 4×4×4): min ω @a_own ∈ [%+.4f, %+.4f], %d unstable (%.1f%%)\n",
        n_s, minimum(skipmissing(s_own)), maximum(skipmissing(s_own)), n_s_unst, 100n_s_unst/n_s)
flush(stdout)

# ── VALIDATION 2: QHA vs exact rebuild, at the largest |Δa| (worst case) ─────
println("\n── validation 2: QHA linear shift vs exact rebuild ──")
worst = sel[sortperm(abs.(Δa_all); rev=true)[1:min(n_qha_check, n_sel)]]
val2 = NamedTuple[]
for i in worst
    θ  = lin_params .+ forest[i, :]
    Δa = -dot(b_prime, θ)/dot(b_double_prime, θ)
    pred = min_freq_shift(bp0.Bq, dBq, θ, bp0.qnorm, Δa)
    ACEpotentials.Models.set_linear_parameters!(model, lin_params)
    bpx  = bandpath_rows(model, element, a_ref + Δa, N_cell)
    exact = min_freq_shift(bpx.Bq, dBq, θ, bpx.qnorm, 0.0)
    @printf("  member %-7d Δa = %+.5f Å : QHA %+.4f vs exact %+.4f THz  (Δ = %.4f)\n",
            i, Δa, pred, exact, abs(pred-exact))
    push!(val2, (; i, Δa, pred, exact, err = abs(pred-exact)))
    flush(stdout)
end
qha_err = isempty(val2) ? 0.0 : maximum(v -> v.err, val2)
@printf("  worst-case QHA error over the sampled |Δa| range: %.4f THz\n", qha_err)
qha_err > 0.1 && @warn "QHA linear shift is off by >0.1 THz at the largest |Δa| — consider a quadratic (3-point) fit"

# ── VALIDATION 3: Newton Δa vs relax_lattice_constant ───────────────────────
println("\n── validation 3: Newton Δa vs relax_lattice_constant ──")
rc = randperm(n_sel)[1:min(n_relax_check, n_sel)]
d_relax = Float64[]
for t in rc
    θ = lin_params .+ forest[sel[t], :]
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    a_true = ACEWorkflow.relax_lattice_constant(model, element)
    push!(d_relax, (a_ref + Δa_all[t]) - a_true)
end
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
@printf("  over %d members: |Newton − relax| max %.5f Å, median %.5f Å\n",
        length(rc), maximum(abs.(d_relax)), median(abs.(d_relax)))
flush(stdout)

# ═════════════════════════════════════════════════════════════════════════════
#  verdict
# ═════════════════════════════════════════════════════════════════════════════
println("\n══ VERDICT ═══════════════════════════════════════════════════════")
f_forest = 100n_unst_own/n_sel; f_samp = 100n_s_unst/n_s
@printf("  forest   : %.3f%% unstable (%d / %d)\n", f_forest, n_unst_own, n_sel)
@printf("  samples  : %.3f%% unstable (%d / %d)\n", f_samp, n_s_unst, n_s)
@printf("  forest min ω floor  = %+.4f THz;  sample min ω floor = %+.4f THz\n",
        minimum(minω_own[ok]), minimum(filter(!isnan, s_own)))
if f_forest < 0.5 * f_samp
    println("  → samples are MORE unstable than the cloud they were fitted to:")
    println("    the hypercube is manufacturing instability in its corners, not inheriting it.")
elseif f_forest > 2 * f_samp
    println("  → the forest is MORE unstable than the samples; rejection/box clipping is")
    println("    filtering some of it out. Instability originates in the corrections.")
else
    println("  → comparable rates: the instability is INHERITED from the delta forest.")
end
println("  (Compare floors too: a sample below the forest floor cannot have been inherited.)")

# ═════════════════════════════════════════════════════════════════════════════
#  figure + outputs  (MLST sizing: built at final display width, 11–13 pt text)
# ═════════════════════════════════════════════════════════════════════════════
BLU = RGBf(0.0,0.447,0.698); ORG = RGBf(0.835,0.369,0.0)
GRY = RGBAf(0.45,0.45,0.45,0.25); RED = RGBAf(0.80,0.15,0.15,0.55)
TITLE, LAB, TICK = 13, 12, 11

plot_idx = randperm(n_sel)[1:min(n_plot, n_sel)]
fig = Figure(size=(540, 350), figure_padding=(6,10,4,6))
ax = Axis(fig[1,1]; xlabel="Wave vector", ylabel="Frequency (THz)",
          title="delta forest, 4×4×4, each at its own a_eq",
          titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
          xticklabelsize=TICK, yticklabelsize=TICK,
          xticks=(bp0.x_ticks, bp0.labels), xgridvisible=false, ygridvisible=false,
          xtickalign=1, ytickalign=1, xticksize=4, yticksize=4)
for t in plot_idx
    θ = lin_params .+ forest[sel[t], :]
    s = isnan(Δa_all[t]) ? 0.0 : Δa_all[t]
    Bs = [bp0.Bq[iq] .+ s .* dBq[iq] for iq in eachindex(bp0.Bq)]
    F  = bands_from(Bs, θ, bp0.qnorm)
    col = (!isnan(minω_own[t]) && minω_own[t] < unstable_tol) ? RED : GRY
    for b in 1:3; lines!(ax, bp0.x_vals, F[b,:]; color=col, linewidth=0.6); end
end
Fm = bands_from(bp0.Bq, lin_params, bp0.qnorm)
for b in 1:3; lines!(ax, bp0.x_vals, Fm[b,:]; color=BLU, linewidth=1.8); end
hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
vlines!(ax, bp0.x_ticks; color=(:black,0.22), linewidth=0.7)
xlims!(ax, first(bp0.x_vals), last(bp0.x_vals))
text!(ax, 0.97, 0.97; text="$(length(plot_idx)) of $n_sel shown", space=:relative,
      align=(:right,:top), fontsize=TICK-1, color=:gray30)

ax2 = Axis(fig[1,2]; xlabel="min ω (THz)", ylabel="density",
           title="forest vs samples", titlesize=TITLE, xlabelsize=LAB, ylabelsize=LAB,
           xticklabelsize=TICK, yticklabelsize=TICK,
           xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
hist!(ax2, filter(!isnan, minω_own); bins=60, normalization=:pdf,
      color=(BLU,0.55), strokewidth=0.4, strokecolor=:white, label="forest")
vlines!(ax2, filter(!isnan, s_own); color=(ORG,0.8), linewidth=0.8)
vlines!(ax2, [0.0]; color=:black, linestyle=:dash, linewidth=1.0)
axislegend(ax2; position=:lt, labelsize=TICK-1, framevisible=true)
text!(ax2, 0.97, 0.80; text="orange = 50 samples", space=:relative,
      align=(:right,:top), fontsize=TICK-2, color=:gray30)
colsize!(fig.layout, 1, Relative(0.62)); colgap!(fig.layout, 20)
save("$outdir/forest_phonon_stability.pdf", fig)
save("$outdir/forest_phonon_stability.png", fig; px_per_unit=4)

writedlm("$outdir/forest_min_freq.csv",
         hcat(collect(sel), Δa_all, minω_ref, minω_own), ',')
println("\nfigure  → $outdir/forest_phonon_stability.{pdf,png}")
println("min ω   → $outdir/forest_min_freq.csv  (member, Δa, min ω @a_ref, min ω @a_own)")
println("band path cache (written earlier) → $outdir/bandpath_4x4x4.jls")
