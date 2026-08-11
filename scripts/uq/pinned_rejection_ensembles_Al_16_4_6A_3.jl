# pinned_rejection_ensembles_Al_16_4_6A_3.jl
#
# STAGE 1 of three.  Builds the two ensembles that stages 2 and 3 consume:
#
#   ensemble_unconstrained.csv   pinned hypercube, drawn with NO predicate
#   ensemble_constrained.csv     same box, same seed, accepted only if the member is
#                                dynamically stable
#
#   stage 2  scripts/qoi/bands_two_ensembles_Al_16_4_6A_3.jl   per-member geom opt,
#            individual native Hessians, the two band figures on one frequency axis
#   stage 3  scripts/uq/parity_calibration_Al_16_4_6A_3.jl     parity and calibration
#
# ── WHY PIN FIRST ───────────────────────────────────────────────────────────
# The undotted per-basis Hessian is built at ONE geometry, so Σ_k θ_k D_k(q) is the
# exact operator only for members whose equilibrium sits there.  Pinning b′·δ = 0 puts
# the whole forest at a_eq, and then a single Hessian screens every proposal at ~ms
# each — which is what makes rejection sampling affordable at all.
#
# The pin is a closed-form rank-one correction to the inverse, not a per-member QP:
#     g = P′\b′,  u = C\g,  v = Ap·u,  gu = g′u
#     δ_i = (Ainv[:,i] − u·v_i/gu) · r_i^c / h_i^c
# g′δ_i = 0 identically.  Verified against explicitly solved KKT systems to 3.9e-16.
#
# It then survives the sampler unaided: b′ becomes an exact null eigenvector of PC′PC,
# hypercube()'s own rank cut discards it, and every retained direction is orthogonal to
# b′.  Rejection therefore only ever works on the phonons, never on the pin.  (The kept
# count is NOT n_params-1 — the cloud is numerically rank deficient on its own account.
# Verify the pin with max |b′·eigvec|, not by counting directions.)
#
# ── WHY THE UNCONSTRAINED ENSEMBLE IS ALSO PINNED ───────────────────────────
# Both draws come from the SAME box with the SAME seed, differing only in the predicate,
# so the pair is paired and any difference is the predicate alone.  An unpinned control
# would confound the pin with the phonon constraint.  Note this makes "unconstrained"
# mean "not phonon-constrained" — it is still a_eq-pinned.
#
# ── 3-ROW UNDOTTED HESSIAN ──────────────────────────────────────────────────
# For a 1-atom primitive cell only 3 rows of H are ever read, and only the reference
# atom's neighbours contribute.  Full H_basis here would be 768×768×684×8 = 3.2 GB;
# three rows is 12.6 MB.  Verified against a full cache at machine precision.
#
# Run:  BUILD_THREADS=16 julia --project -t 40 scripts/uq/pinned_rejection_ensembles_Al_16_4_6A_3.jl
#   N_MEMBERS=20  PIN_TOL=1e-6  ACCEPT_TOL=-0.05  LEV_PCT=0.5  N_CELL=4  MAX_ATTEMPTS=

using ACEWorkflow, ACEpotentials
import ACEpotentials.Models: evaluate_basis
import AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius
using AtomsBuilder, Unitful, ForwardDiff, StaticArrays
using LinearAlgebra, Statistics, Random, Serialization, DelimitedFiles, Printf
using CairoMakie
Random.seed!(1234)

element      = :Al
N_MEMBERS    = parse(Int,     get(ENV, "N_MEMBERS", "20"))
PIN_TOL      = parse(Float64, get(ENV, "PIN_TOL", "1e-6"))     # max |b′·θ| we accept
ACCEPT_TOL   = parse(Float64, get(ENV, "ACCEPT_TOL", "-0.05")) # THz
LEV_PCT      = parse(Float64, get(ENV, "LEV_PCT", "0.5"))
N_CELL       = parse(Int,     get(ENV, "N_CELL", "4"))
qΓtol        = 5e-2
N_PER_SEG    = [20, 20, 20, 20, 60]
MAX_ATTEMPTS = parse(Int, get(ENV, "MAX_ATTEMPTS", string(5000 * N_MEMBERS)))
const BUILD_THREADS = parse(Int, get(ENV, "BUILD_THREADS", "8"))

MODELDIR = "models/Al_16_4_6A_3_"
outdir   = get(ENV, "OUTDIR", "$MODELDIR/results/pinned_ensembles"); mkpath(outdir)
BPCACHE  = "$outdir/bandpath_$(N_CELL)x$(N_CELL)x$(N_CELL)_aref.jls"

# ═════════════════════════════════════════════════════════════════════════════
#  3-row undotted band-path builder.  Duplicated from
#  scripts/uq/pinned_hypercube_rejection_Al_20_4_6A_3.jl rather than included, so this
#  script has no path dependency — keep the copies in step if either is edited.
# ═════════════════════════════════════════════════════════════════════════════
function _site_basis_hessian(m, Rs, Zs, z0, ps, st)
    nR = length(Rs)
    x0 = collect(Float64, reinterpret(Float64, Rs))
    tovec(x) = [SVector{3,eltype(x)}(x[3i-2], x[3i-1], x[3i]) for i in 1:nR]
    Bfun(x)  = evaluate_basis(m, tovec(x), Zs, z0, ps, st)
    Hflat    = ForwardDiff.jacobian(x -> vec(ForwardDiff.jacobian(Bfun, x)), x0)
    return reshape(Hflat, length(Bfun(x0)), 3nR, 3nR)
end

function undotted_rows(sys_super, V, r)
    nlist = PairList(sys_super, cutoff_radius(V))
    Nat   = length(sys_super); D = 3
    ps, st, m = V.ps, V.st, V.model
    _, Rs0, Zs0, z00 = get_neighbours(sys_super, V, nlist, 1)
    NB = length(evaluate_basis(m, Rs0, Zs0, z00, ps, st))
    Js_r, = get_neighbours(sys_super, V, nlist, r)
    contributors = unique(vcat(r, Js_r))
    nt = clamp(BUILD_THREADS, 1, min(Threads.nthreads(), length(contributors)))
    @printf("    %d of %d atoms contribute; %d threads, %.2f GB/thread\n",
            length(contributors), Nat, nt, NB*(3length(Js_r))^2*8/1e9); flush(stdout)
    bufs = [zeros(D, D*Nat, NB) for _ in 1:nt]
    @sync for t in 1:nt
        Threads.@spawn begin
            Hl = bufs[t]
            for i in contributors[t:nt:end]
                Js, Rs, Zs, z0 = get_neighbours(sys_super, V, nlist, i)
                Hi = _site_basis_hessian(m, Rs, Zs, z0, ps, st)
                Ji = (i-1)*D .+ (1:D); is_ref = (i == r)
                for (α1, j1) in enumerate(Js)
                    hit1 = (j1 == r); (hit1 || is_ref) || continue
                    A1 = (α1-1)*D .+ (1:D)
                    for (α2, j2) in enumerate(Js)
                        A2 = (α2-1)*D .+ (1:D); J2 = (j2-1)*D .+ (1:D)
                        @views for kb in 1:NB
                            blk = Hi[kb, A1, A2]
                            if hit1; Hl[:, J2, kb] .+= blk; Hl[:, Ji, kb] .-= blk; end
                            if is_ref; Hl[:, J2, kb] .-= blk; Hl[:, Ji, kb] .+= blk; end
                        end
                    end
                end
                Hi = nothing
            end
        end
    end
    H = bufs[1]
    for t in 2:nt; H .+= bufs[t]; bufs[t] = zeros(0,0,0); end
    return H
end

function bandpath_rows(model, a)
    sys_prim, sys_super = bulk_prim_super(element; a=a, N_cell=N_CELL)
    fc = precompute_force_constants(sys_prim, sys_super, model)
    fc.Np == 1 || error("this builder assumes a 1-atom primitive cell; got Np = $(fc.Np)")
    r = fc.p2s_map[1] + 1
    @printf("  building 3-row undotted Hessian at a = %.5f Å (%d atoms) …\n", a, length(sys_super))
    t = @elapsed Hrows = undotted_rows(sys_super, model, r)
    @printf("    done in %.1f min\n", t/60); flush(stdout)
    q_list, x_vals, x_ticks, labels, _ = fcc_band_path(fc.L; N_per_seg=N_PER_SEG)
    NB = size(Hrows, 3)
    Rmi = Vector{SVector{3,Float64}}(undef, fc.Ns)
    for k in 1:fc.Ns
        R_cart  = fc.L * (fc.frac_super[k] - fc.frac_prim[1])
        R_sfrac = fc.Linv_super * R_cart; R_sfrac = R_sfrac .- round.(R_sfrac)
        Rmi[k]  = fc.L_super * R_sfrac
    end
    mass = fc.masses[1]
    Bq = Vector{Matrix{ComplexF64}}(undef, length(q_list))
    Threads.@threads for iq in eachindex(q_list)
        q = q_list[iq]; M = zeros(ComplexF64, 9, NB)
        for k in 1:fc.Ns
            eph = exp(im * dot(q, Rmi[k])); cols = 3(k-1) .+ (1:3)
            @views for kb in 1:NB
                blk = Hrows[:, cols, kb]
                for β in 1:3, α in 1:3; M[3(β-1)+α, kb] += blk[α, β] * eph; end
            end
        end
        M ./= mass
        for kb in 1:NB
            D3 = reshape(@view(M[:, kb]), 3, 3); M[:, kb] = vec((D3 .+ D3') ./ 2)
        end
        Bq[iq] = M
    end
    return (; Bq, q_list, x_vals, x_ticks, labels, Np=fc.Np, qnorm=norm.(q_list), a_ref=a)
end

# ── model, design matrix, pin direction ─────────────────────────────────────
result = load_model(element, 16, 4, 6, 3; dataset_name="")
model, lin_params = result.model, result.lin_params
n_params = length(lin_params)
P, W, Y  = result.P, result.W, result.Y
Ap = Diagonal(W) * result.A / P
Yw = W .* Y
N  = size(Ap, 1)
@printf("Model %s: %d params, %d observations, %d threads\n",
        result.name, n_params, N, Threads.nthreads()); flush(stdout)

a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@printf("a_eq = %.6f Å;  %d×%d×%d cell, half-box %.2f Å\n",
        a_eq, N_CELL, N_CELL, N_CELL, N_CELL*a_eq/2); flush(stdout)

lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
                       ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
@printf("b′·lin_params = %.3e (≈0 ⇒ a_eq is the mean model's relaxed constant)\n",
        dot(b_prime, lin_params))
@printf("b″·lin_params = %+.6e\n", dot(b_double_prime, lin_params)); flush(stdout)

# ── pinned delta forest, closed form ────────────────────────────────────────
λ  = 1.0 / N
C  = Symmetric(P'*P .* λ .+ Ap'*Ap); Cf = cholesky(C)
Ainv = Cf \ Matrix(Ap')
x̃    = Cf \ (Ap' * Yw)
lev  = vec(sum(Ap' .* Ainv; dims=1))
g  = transpose(P) \ b_prime
u  = Cf \ g; gu = dot(g, u); v = Ap * u
abs(gu) > 1e-30 || error("g′u ≈ 0 — the pin direction lies in the null space of C⁻¹")
x̃_c   = x̃ .- u .* (dot(g, x̃)/gu)
lev_c = lev .- (v .^ 2) ./ gu
r_c   = Yw .- Ap * x̃_c
scale = r_c ./ lev_c
PCx   = transpose(Ainv .* transpose(scale))
PCx .-= (scale .* v ./ gu) * transpose(u)
Ainv = nothing; GC.gc()
@printf("\nPIN CHECK (forest): max |b′·δ| = %.3e over %d corrections (tolerance %.0e)\n",
        maximum(abs.(PCx * g)), N, PIN_TOL)
maximum(abs.(PCx * g)) < PIN_TOL ||
    error("the pin did not hold on the forest — everything downstream is invalid")

# top-leverage subset feeds the hypercube, matching POPSRegression.corrections
keep_rows = lev_c .>= quantile(lev_c, 1 - LEV_PCT)
@printf("cloud: %d of %d corrections (top %.0f%% constrained leverage)\n",
        count(keep_rows), N, 100LEV_PCT)
pwise = PCx[keep_rows, :] / transpose(P)          # → θ-space: δθ′ = δx′P⁻ᵀ
PCx = nothing; GC.gc()
@printf("PIN CHECK (θ-space): max |b′·δθ| = %.3e\n", maximum(abs.(pwise * b_prime)))
flush(stdout)

# ── band path at a_eq (all members live there, so one build serves everything) ──
bp = isfile(BPCACHE) ? (println("band path: reusing $BPCACHE"); deserialize(BPCACHE)) :
     begin
         println("band path: building (this is the only Hessian build)")
         b = bandpath_rows(model, a_eq); serialize(BPCACHE, b)
         println("  cached → $BPCACHE"); b
     end
keep_q = findall(bp.qnorm .>= qΓtol)
@printf("band path: %d q-points, %d after the near-Γ cut\n", length(bp.Bq), length(keep_q))

# ── hypercube on the pinned cloud ───────────────────────────────────────────
eig_pops, bound_pops = hypercube(pwise)
n_dir = size(eig_pops, 2)
# NOTE the kept-direction count is NOT n_params-1.  hypercube() cuts at
# eigvals > max*1e-8, and the POPS delta cloud is numerically rank deficient far beyond
# the single direction the pin removes — its covariance spectrum decays fast under a
# strong preconditioner.  So the COUNT says little about the pin; what matters is that
# the RETAINED directions are orthogonal to b′, which is what makes every draw pinned.
@printf("\nhypercube: %d / %d directions kept (rank cut at max·1e-8; the deficiency is\n",
        n_dir, n_params)
@printf("           mostly intrinsic to the cloud, not the pin)\n")
@printf("           max |b′·eigvec| over kept directions = %.3e  ← 0 ⇒ every draw is pinned\n",
        maximum(abs.(transpose(eig_pops)*b_prime)))
flush(stdout)

# ── the predicate: cheap test first ─────────────────────────────────────────
n_prop = Ref(0); n_rej_K = Ref(0); n_rej_ω = Ref(0)
function is_physical(θ)
    n_prop[] += 1
    if dot(b_double_prime, θ) <= 0; n_rej_K[] += 1; return false; end
    for iq in keep_q
        ev = eigvals(Hermitian(reshape(bp.Bq[iq]*θ, 3bp.Np, 3bp.Np)))
        if minimum(sign.(ev).*sqrt.(abs.(ev)).*FREQ_THz) < ACCEPT_TOL
            n_rej_ω[] += 1; return false                     # early exit
        end
    end
    return true
end

println("\n── draw 1: UNCONSTRAINED (pinned, no predicate) ──")
Random.seed!(1234)
mat_u, _ = sample_hypercube(eig_pops, bound_pops, lin_params;
                            number_of_committee_members=N_MEMBERS)
println("── draw 2: CONSTRAINED (same box, same seed, phonon-positive) ──"); flush(stdout)
Random.seed!(1234)
t = @elapsed mat_c, _ = rejection_sample_hypercube(eig_pops, bound_pops, lin_params,
                        is_physical; number_of_committee_members=N_MEMBERS,
                        max_attempts=MAX_ATTEMPTS)
@printf("  %d proposals for %d members in %.1f s → acceptance %.3f%%\n",
        n_prop[], N_MEMBERS, t, 100N_MEMBERS/n_prop[])
@printf("  → %.2f%% of the pinned box is dynamically unphysical\n", 100*(1 - N_MEMBERS/n_prop[]))
@printf("  rejected on b″·θ ≤ 0 : %d\n  rejected on a soft mode: %d\n", n_rej_K[], n_rej_ω[])

# ── verify both ensembles, then save ────────────────────────────────────────
mem_u = [mat_u[:, i] for i in 1:N_MEMBERS]
mem_c = [mat_c[:, i] for i in 1:N_MEMBERS]
minw(θ) = minimum(minimum(sign.(ev).*sqrt.(abs.(ev)).*FREQ_THz)
                  for ev in (eigvals(Hermitian(reshape(bp.Bq[iq]*θ, 3bp.Np, 3bp.Np)))
                             for iq in keep_q))
wu = minw.(mem_u); wc = minw.(mem_c)
# The pin constrains b′·δ, NOT b′·θ.  θ = lin_params + δ, and b′·lin_params is the mean
# model's own relaxation residual — whatever tolerance relax_lattice_constant stopped at
# — which every member inherits equally and which says nothing about the pin.  Test the
# quantity the pin actually controls, and express it as a lattice-constant shift in Å so
# the tolerance means something physical:  Δa_i = −b′·δ_i / (b″·θ_i).
da(θ) = -dot(b_prime, θ .- lin_params) / dot(b_double_prime, θ)
du = abs.(da.(mem_u)); dc = abs.(da.(mem_c))
da_mean = -dot(b_prime, lin_params) / dot(b_double_prime, lin_params)
@printf("\nmean model's own relaxation residual: Δa = %+.3e Å  (relax_lattice_constant\n", da_mean)
@printf("  tolerance; shared by every member, not a pin failure)\n")
@printf("PIN CHECK (samples): max |Δa − Δa_mean| = %.3e Å unconstrained, %.3e Å constrained  (tol %.0e Å)\n",
        maximum(du), maximum(dc), PIN_TOL)
@printf("  raw: max |b′·δ| = %.3e / %.3e eV/Å\n",
        maximum(abs(dot(b_prime, θ .- lin_params)) for θ in mem_u),
        maximum(abs(dot(b_prime, θ .- lin_params)) for θ in mem_c))
max(maximum(du), maximum(dc)) < PIN_TOL ||
    error("drawn members are not pinned to the mean model's a_eq — the undotted Hessian " *
          "is the wrong operator for them")
@printf("min ω  unconstrained: [%+.4f, %+.4f], %d/%d below %.2f THz\n",
        minimum(wu), maximum(wu), count(<(ACCEPT_TOL), wu), N_MEMBERS, ACCEPT_TOL)
@printf("min ω  constrained  : [%+.4f, %+.4f], %d/%d below %.2f THz %s\n",
        minimum(wc), maximum(wc), count(<(ACCEPT_TOL), wc), N_MEMBERS, ACCEPT_TOL,
        count(<(ACCEPT_TOL), wc) == 0 ? "✓" : "← PREDICATE LEAKED")

writedlm("$outdir/ensemble_unconstrained.csv", mat_u', ',')
writedlm("$outdir/ensemble_constrained.csv",   mat_c', ',')
serialize("$outdir/pinned_ensembles.jls",
          (; mem_u, mem_c, wu, wc, a_eq, n_dir, n_params, N_MEMBERS, PIN_TOL, ACCEPT_TOL,
             LEV_PCT, N_CELL, acceptance = N_MEMBERS/n_prop[],
             n_rej_K = n_rej_K[], n_rej_ω = n_rej_ω[], seed = 1234))
println("\nensembles → $outdir/ensemble_{unconstrained,constrained}.csv")
println("            (20 × $n_params, one full coefficient vector per row)")
println("next: scripts/qoi/bands_two_ensembles_Al_16_4_6A_3.jl, then")
println("      scripts/uq/parity_calibration_Al_16_4_6A_3.jl")
