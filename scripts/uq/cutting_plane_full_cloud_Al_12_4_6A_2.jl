# cutting_plane_full_cloud_Al_12_4_6A_2.jl
#
# Cutting-plane constrain the WHOLE top-leverage POPS cloud, not just a 30-member
# forest. At leverage_percentile = 0.5 that is 73,479 of 146,958 observations for
# Al_12_4_6A_2_ -- the same cloud `corrections(...; leverage_percentile=0.5)`
# builds.
#
# Per observation i the constrained POPS member solves
#
#     min  1/2 x'Hx + q'x        (x = P*theta, OSQP's variable)
#     s.t. Ap[i,:]·x = Yw[i]     POPS interpolation, exact
#          b'·theta  = 0         equilibrium pinned to a_eq
#          Born rows >= bounds   elastic stability
#          + cutting planes      e'D(q;theta)e >= omega_cut^2, added on demand
#
# COST. Measured in scripts/uq/sdp_vs_cuts_benchmark_Al_12_4_6A_2.jl: ~10 ms for a
# member needing no cuts, ~0.3 s for one needing 3. Five of six needed none, so
# ~10 ms x 73,479 ~ 750 s single-threaded is the right order. Two things can break
# that estimate, so both are instrumented and reported:
#   * members that hit max_cuts pay 40 solves against a constraint matrix that
#     grows by one row per soft mode -- if these are common they dominate;
#   * OSQP may return non-Solved for observations where the interpolation equality
#     is infeasible against Born + phonon rows. Status is recorded, never ignored.
#
# THREADING. The legacy `osqp = OSQP.Model()` at module scope is mutated by
# setup!, so it is NOT thread-safe. One model per thread here.
#
# OUTPUT SIZE. 73,479 x 91 as text CSV would be ~130 GB. The committee is written
# with `serialize` (~53 MB) instead; only per-member diagnostics go to CSV.
#
# CHECKPOINTING. Partial results are serialised every `ckpt_every` members so a
# walltime kill does not lose the run.
#
# Run a PILOT first to calibrate before committing to the full cloud:
#   julia --project -t 16 scripts/uq/cutting_plane_full_cloud_Al_12_4_6A_2.jl 500
# then the full thing (n_members <= 0 means "all"):
#   sbatch scripts/uq/run_cutting_plane_full_cloud.slurm

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
using SparseArrays, LinearAlgebra, OSQP, Serialization, Printf

element, dataset  = :Al, ""
N_cell_fc         = 4
N_per_seg         = [20, 20, 20, 20, 60]
cut_margin_THz    = 0.15
qΓtol             = 5e-2
max_cuts          = 40
lev_percentile    = 0.5
n_members_arg     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 0     # <=0 → all
ckpt_every        = 5_000

BLAS.set_num_threads(1)          # Julia threads own the parallelism here
@printf("julia threads = %d, BLAS threads = %d\n", Threads.nthreads(), BLAS.get_num_threads())

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params; n_params = length(lin_params)
P = result.P; Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y
M = size(Ap, 1); λ = 1.0/M
outdir = "$(result.dir)/results/cutting_plane_full_cloud"; mkpath(outdir)
@printf("%s: %d observations x %d params\n", result.name, M, n_params); flush(stdout)

# ── reference geometry, Born + a_eq rows (verbatim from the legacy script) ────
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))
eV_to_GPa = 160.2176621/abs(det(L0))
H_el = elastic_hessian_basis(model; element=element, a=a_eq)
c11_0 = reshape(H_el,36,n_params)[1,:]; c12_0 = reshape(H_el,36,n_params)[7,:]; c44_0 = reshape(H_el,36,n_params)[22,:]
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(ACEWorkflow.Elasticity.reference_system(element; a=a), model))
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
born_rows  = vcat(c44_0', (c11_0.-c12_0)', (c11_0.+2 .*c12_0)', b_double_prime')
born_lower = [0.1, 1.0, 0.1, 1e-9]
@printf("a_eq = %.5f Å\n", a_eq); flush(stdout)

bp = bandpath_Dk(result, model, element, a_eq, N_cell_fc; N_per_seg=N_per_seg)
ω2_cut = (cut_margin_THz / FREQ_THz)^2
@printf("band path: %d q-points (%d kept above qΓtol)\n",
        length(bp.Bq), count(>=(qΓtol), bp.qnorm)); flush(stdout)

# ── leverage → the top-percentile cloud ──────────────────────────────────────
println("── leverage (Cholesky + 91 × M solve) ──"); flush(stdout)
t = @elapsed begin
    C   = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C)
    AtX = Cf \ Matrix(Ap')                      # 91 × M
    lev = vec(sum(Ap' .* AtX; dims=1))
end
AtX = nothing; GC.gc()                          # 107 MB, not needed past selection
thresh   = quantile(lev, 1 - lev_percentile)
selected = findall(>=(thresh), lev)
n_members_arg > 0 && (selected = selected[1:min(n_members_arg, length(selected))])
N = length(selected)
@printf("leverage in [%.3e, %.3e]; threshold %.3e → %d members  [%.1f s]\n",
        minimum(lev), maximum(lev), thresh, N, t); flush(stdout)

# ── precompute everything constant in the hot loop ───────────────────────────
# The legacy constrain_member re-does `born_rows/P` and `b_prime'/P` on EVERY
# call; over 73k members that is pure waste, so hoist them.
Hqp        = sparse(Ap'*Ap .+ λ.*(P'*P))
qqp        = -(Ap'*Yw)
born_rows_P = sparse(born_rows / P)
b_prime_P   = sparse(reshape(b_prime' / P, 1, n_params))
Pfac        = factorize(P)                       # reused for every `P \ x`

models = [OSQP.Model() for _ in 1:Threads.nthreads()]

function solve_one!(osqp, i, extra_rows_P, extra_lower)
    A_full = vcat(sparse(reshape(Ap[i, :], 1, n_params)), b_prime_P,
                  born_rows_P, extra_rows_P)
    l = vcat([Yw[i]], [0.0], born_lower, extra_lower)
    u = vcat([Yw[i]], [0.0], fill(Inf, length(born_lower) + length(extra_lower)))
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=A_full, l=l, u=u, max_iter=4_000_000,
                check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    r = OSQP.solve!(osqp)
    return Pfac \ r.x, r.info.status
end

# ── the run ──────────────────────────────────────────────────────────────────
Θ       = zeros(Float64, n_params, N)
ncuts   = zeros(Int, N); niters = zeros(Int, N)
minω    = fill(NaN, N); status = fill(:unset, N); secs = zeros(Float64, N)
done    = Threads.Atomic{Int}(0)
t_start = time()

Threads.@threads for k in 1:N
    i    = selected[k]
    osqp = models[Threads.threadid()]
    tk = @elapsed begin
        extra_rows_P = zeros(0, n_params); extra_lower = Float64[]
        θ, st = solve_one!(osqp, i, extra_rows_P, extra_lower)
        for it in 0:max_cuts
            soft = soft_modes(θ, bp, ω2_cut; qΓtol=qΓtol)
            isempty(soft) && break
            niters[k] += 1
            it == max_cuts && break              # give up; θ stays soft, flagged below
            for (iq, e) in soft
                extra_rows_P = vcat(extra_rows_P, (cut_row(iq, e, bp)' / P))
                push!(extra_lower, ω2_cut)
            end
            ncuts[k] += length(soft)
            θ, st = solve_one!(osqp, i, extra_rows_P, extra_lower)
        end
        Θ[:, k] = θ; status[k] = st; minω[k] = min_freq_stable(θ, bp; qΓtol=qΓtol)
    end
    secs[k] = tk
    d = Threads.atomic_add!(done, 1) + 1
    if d % 1000 == 0
        el = time() - t_start
        @printf("  %6d / %d  (%.1f%%)  elapsed %.1f min  eta %.1f min  mean %.1f ms/member\n",
                d, N, 100d/N, el/60, el/60*(N-d)/d, 1000*el*Threads.nthreads()/d)
        flush(stdout)
    end
    if d % ckpt_every == 0
        serialize("$outdir/checkpoint.jls",
                  (Θ=Θ, selected=selected, ncuts=ncuts, niters=niters,
                   minω=minω, status=status, done=d))
    end
end

wall = time() - t_start
@printf("\n── done in %.1f min (%d threads) ──\n", wall/60, Threads.nthreads())

# ── diagnostics ──────────────────────────────────────────────────────────────
solved   = count(==(:Solved), status)
hitmax   = count(k -> niters[k] > max_cuts, 1:N)
stillbad = count(<(cut_margin_THz - 1e-6), filter(!isnan, minω))
@printf("  OSQP :Solved            %d / %d (%.2f%%)\n", solved, N, 100solved/N)
@printf("  needed >=1 cut          %d / %d (%.2f%%)\n", count(>(0), ncuts), N, 100count(>(0),ncuts)/N)
@printf("  hit max_cuts (%d)       %d\n", max_cuts, hitmax)
@printf("  short of margin         %d   (tolerance miss, not necessarily unstable —\n", stillbad)
@printf("                               see the min ω > 0 count below)\n")
@printf("  cuts   : mean %.2f max %d\n", mean(ncuts), maximum(ncuts))
@printf("  min ω  : [%.4f, %.4f] THz\n", minimum(filter(!isnan,minω)), maximum(filter(!isnan,minω)))
@printf("  time   : mean %.1f ms, median %.1f ms, max %.1f s\n",
        1000mean(secs), 1000median(secs), maximum(secs))

# ── filtered committees ──────────────────────────────────────────────────────
# Two distinct criteria, kept separate because they mean different things:
#   dyn_stable : min ω > 0     — genuinely dynamically stable (no imaginary mode)
#   at_margin  : min ω >= ω_cut − 1e-6 — also met the imposed cutting-plane margin
# A member can be the first without being the second: hitting max_cuts leaves it
# converged to just under the margin, which is a tolerance miss, NOT an instability.
dyn_stable = (!isnan).(minω) .& (minω .> 0.0)
at_margin  = (!isnan).(minω) .& (minω .>= cut_margin_THz - 1e-6)
@printf("  dynamically stable (min ω > 0)        %d / %d (%.3f%%)\n",
        count(dyn_stable), N, 100count(dyn_stable)/N)
@printf("  also met margin (min ω ≥ %.2f THz)    %d / %d (%.3f%%)\n",
        cut_margin_THz, count(at_margin), N, 100count(at_margin)/N)

tag = n_members_arg > 0 ? "_pilot$(N)" : ""      # keep pilots from clobbering the full run
serialize("$outdir/committee_full_cloud$tag.jls",
          (Θ=Θ, selected=selected, ncuts=ncuts, niters=niters, minω=minω,
           status=status, secs=secs, a_eq=a_eq, lev_percentile=lev_percentile,
           cut_margin_THz=cut_margin_THz, N_cell_fc=N_cell_fc, N_per_seg=N_per_seg,
           dyn_stable=dyn_stable, at_margin=at_margin))
# Stable-only committee: this is the one to feed hypercube()/rejection sampling.
serialize("$outdir/committee_stable$tag.jls",
          (Θ=Θ[:, at_margin], selected=selected[at_margin], minω=minω[at_margin],
           a_eq=a_eq, cut_margin_THz=cut_margin_THz, criterion="min ω ≥ ω_cut − 1e-6"))
@printf("  → committee_stable%s.jls holds %d members\n", tag, count(at_margin))

open("$outdir/member_diagnostics$tag.csv", "w") do io
    println(io, "# leverage_percentile=$lev_percentile a_eq=$a_eq cut_margin_THz=$cut_margin_THz")
    println(io, "k,obs_index,n_cuts,n_iters,min_omega_THz,status,seconds")
    for k in 1:N
        @printf(io, "%d,%d,%d,%d,%.6f,%s,%.4f\n",
                k, selected[k], ncuts[k], niters[k], minω[k], status[k], secs[k])
    end
end
isfile("$outdir/checkpoint.jls") && rm("$outdir/checkpoint.jls")
@printf("\nall members  → %s/committee_full_cloud%s.jls  (%.1f MB)\n",
        outdir, tag, filesize("$outdir/committee_full_cloud$tag.jls")/1e6)
@printf("stable only  → %s/committee_stable%s.jls  (%.1f MB)\n",
        outdir, tag, filesize("$outdir/committee_stable$tag.jls")/1e6)
println("diagnostics  → $outdir/member_diagnostics$tag.csv")
