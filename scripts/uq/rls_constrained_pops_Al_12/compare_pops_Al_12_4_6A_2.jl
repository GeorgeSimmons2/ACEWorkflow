# compare_pops_Al_12_4_6A_2.jl — standard POPS on the RLS fit vs rejection-sampled POPS
# on a phonon-constrained RLS fit.  Machinery from ../bandpath_committee_undotted_Al_12_4_6A_2_multivolume.jl.
#
#   naive        hypercube of the POPS corrections about lin_params, sample_hypercube.
#                Members are NOT pinned, so each sits at its own a_k (root of b′(a)·θ = 0 on the
#                same hydrostatic E(a) that defines the pin) and gets its OWN H_basis at
#                1.00 and 1.02 a_k — 2 builds per member, cached as undotted_Hbasis_naive_*.jls
#   constrained  mean = RLS with b′(a_eq)·θ = 0, Born, ω > 0 at 1.00 and 1.02 a_eq (cutting plane);
#                POPS corrections about it, pinned so b′·δ = 0 (every member keeps a_eq);
#                rejection-sampled with ω > 0 at both volumes as the predicate
#
# Run:  julia --project -t 20 scripts/uq/rls_constrained_pops_Al_12/compare_pops_Al_12_4_6A_2.jl
#   N_MEMBERS=30  OUTDIR=<this dir>/results

include(joinpath(@__DIR__, "..", "..", "bandpath_phonon_uq", "lib.jl"))
using SparseArrays, OSQP, Random
Random.seed!(1234)

N_MEMBERS  = parse(Int, get(ENV, "N_MEMBERS", "30"))
outdir     = get(ENV, "OUTDIR", joinpath(@__DIR__, "results")); mkpath(outdir)
vol_scales = [1.00, 1.02]
margin     = 0.01                                    # THz
ω2_cut     = (margin / FREQ_THz)^2

result = load_model(:Al, 12, 4, 6, 2; dataset_name="")
model, lin_params, P = result.model, result.lin_params, result.P
Ap = Diagonal(result.W)*result.A/P; Yw = result.W.*result.Y; λ = 1.0/size(Ap,1)

# ── a_eq pin, Born rows, band paths at both volumes ──────────────────────────
a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(ACEWorkflow.Elasticity.reference_system(:Al; a=a), model))
b_prime = ForwardDiff.derivative(lattice_basis, a_eq)
b2      = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
H_el = reshape(elastic_hessian_basis(model; element=:Al, a=a_eq), 36, :)
c11, c12, c44 = H_el[1,:], H_el[7,:], H_el[22,:]
born_rows  = vcat(c44', (c11.-c12)', (c11.+2 .*c12)', b2')
born_lower = [0.1, 1.0, 0.1, 1e-9]
bps   = [bandpath_Dk(result, model, :Al, s*a_eq, 4) for s in vol_scales]
minω(θ) = [min_freq_stable(θ, bp) for bp in bps]

# ── constrained RLS: QP in θ̃ = Pθ, cutting plane over both volumes ──────────
osqp = OSQP.Model(); Hqp = sparse(Ap'*Ap .+ λ.*(P'*P)); qqp = -(Ap'*Yw)
function mean_fit(rows, lower)
    A = vcat(sparse(b_prime'/P), sparse(rows/P)); l = vcat(0.0, lower); u = vcat(0.0, fill(Inf, length(lower)))
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=A, l=l, u=u, max_iter=4_000_000, adaptive_rho_interval=25,
                verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    P \ OSQP.solve!(osqp).x
end
rows, lower = born_rows, born_lower
θ_con = mean_fit(rows, lower)
for it in 1:80
    global θ_con, rows, lower
    soft = [(v, iq, e) for v in eachindex(bps) for (iq, e) in soft_modes(θ_con, bps[v], ω2_cut)]
    isempty(soft) && break
    rows  = vcat(rows, reduce(vcat, [cut_row(iq, e, bps[v])' for (v, iq, e) in soft]))
    lower = vcat(lower, fill(ω2_cut, length(soft)))
    θ_con = mean_fit(rows, lower)
end
@printf("min ω (THz) at %s a_eq:  RLS %s   constrained RLS %s\n", vol_scales,
        round.(minω(lin_params); digits=3), round.(minω(θ_con); digits=3))

# ── POPS corrections (top-50% leverage, as POPSRegression.corrections) ──────
C = Symmetric(Ap'*Ap .+ λ.*(P'*P)); Cf = cholesky(C)
Ainv = Cf \ Matrix(Ap'); lev = vec(sum(Ap' .* Ainv; dims=1))
function pops_cloud(θ; pin=false)
    L, r, M = lev, Yw .- Ap*(P*θ), copy(Ainv)
    if pin                                           # rank-one pin: g′δ = 0 for every correction
        g = P' \ b_prime; u = Cf \ g; gu = dot(g, u); v = Ap*u
        L = lev .- v.^2 ./ gu; M .-= u * (v' ./ gu)
    end
    keep = L .>= quantile(L, 0.5)
    Matrix((M[:, keep] .* (r[keep] ./ L[keep])')' / P')   # rows = θ-space corrections
end

eig_n, bnd_n = hypercube(pops_cloud(lin_params))
naive = sample_hypercube(eig_n, bnd_n, lin_params; number_of_committee_members=N_MEMBERS)[1]

cloud_c = pops_cloud(θ_con; pin=true)
@printf("pin check: max |b′·δ| = %.2e\n", maximum(abs.(cloud_c * b_prime)))
eig_c, bnd_c = hypercube(cloud_c)
con = rejection_sample_hypercube(eig_c, bnd_c, θ_con, θ -> all(minω(θ) .>= margin);
                                 number_of_committee_members=N_MEMBERS, max_attempts=5_000_000)[1]

# ── naive members at their own geometry: own a_k, own Hessians ──────────────
b′(a) = ForwardDiff.derivative(lattice_basis, a)
own_a(θ; a=a_eq) = (for _ in 1:30; a -= dot(b′(a), θ) / dot(ForwardDiff.derivative(b′, a), θ); end; a)
a_naive   = [own_a(θ) for θ in eachcol(naive)]
naive_bps = map(a_naive) do a
    bp = [bandpath_Dk(result, model, :Al, s*a, 4; tag="naive_") for s in vol_scales]; GC.gc(); bp
end

# ── compare + save ──────────────────────────────────────────────────────────
M_n = reduce(vcat, [[min_freq_stable(θ, b) for b in bp]' for (θ, bp) in zip(eachcol(naive), naive_bps)])
M_c = reduce(vcat, [minω(θ)' for θ in eachcol(con)])
for (tag, M, E) in (("naive", M_n, naive), ("constrained", M_c, con))
    @printf("%-12s unstable (min ω < -0.05 THz) at 100%%: %2d/%d  102%%: %2d/%d   worst %+.3f THz\n",
            tag, count(<(-0.05), M[:,1]), N_MEMBERS, count(<(-0.05), M[:,2]), N_MEMBERS, minimum(M))
    writedlm("$outdir/committee_$tag.csv", E', ',')
    writedlm("$outdir/minomega_$tag.csv", vcat(["minomega_100" "minomega_102"], M), ',')
end
writedlm("$outdir/theta_rls_constrained.csv", θ_con, ',')
writedlm("$outdir/a_naive.csv", vcat(["a_own"], a_naive), ',')
@printf("naive own a: [%.4f, %.4f] Å  (a_eq = %.4f)\n", extrema(a_naive)..., a_eq)

for (v, s) in enumerate(vol_scales)
    # naive: each member's bands on its own path; same q-index layout, so drawn on the RLS x-axis
    fig = Figure(size=(340, 300)); ax = Axis(fig[1,1]; ylabel="Frequency (THz)", title="naive POPS, $(round(Int,100s))% own a",
                                             xticks=(bps[v].x_ticks, bps[v].labels), xgridvisible=false, ygridvisible=false)
    for (θ, bp) in zip(eachcol(naive), naive_bps)
        F = bands(θ, bp[v]); col = min_freq_stable(θ, bp[v]) < -0.05 ? RGBAf(0.8,0.15,0.15,0.45) : RGBAf(0.45,0.45,0.45,0.3)
        for b in axes(F, 1); lines!(ax, bps[v].x_vals, F[b,:]; color=col, linewidth=0.7); end
    end
    Fm = bands(lin_params, bps[v]); for b in axes(Fm, 1); lines!(ax, bps[v].x_vals, Fm[b,:]; color=RGBf(0.0,0.447,0.698), linewidth=1.6); end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8); xlims!(ax, extrema(bps[v].x_vals)...)
    save("$outdir/bands_naive_$(round(Int,100s)).pdf", fig); save("$outdir/bands_naive_$(round(Int,100s)).png", fig; px_per_unit=4)
    plot_committee_bands(collect(eachcol(con)), θ_con, bps[v], "constrained POPS, $(round(Int,100s))% a_eq", "$outdir/bands_constrained_$(round(Int,100s)).png")
end
println("outputs → $outdir")
