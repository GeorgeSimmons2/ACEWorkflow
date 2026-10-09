# compare_rls_Al_12_4_6A_2.jl — phonons of the RLS fit vs the constrained RLS fit, no POPS.
#
#   RLS          lin_params
#   constrained  RLS with b′(a_eq)·θ = 0 (a_eq held), Born, ω > 0 at 1.00 and 1.02 a_eq (cutting plane)
#
# Both relax to a_eq, so one Hessian per volume serves both.
# Run:  julia --project -t 20 scripts/uq/rls_constrained_pops_Al_12/compare_rls_Al_12_4_6A_2.jl   (OUTDIR=<this dir>/results_rls)

include(joinpath(@__DIR__, "..", "..", "bandpath_phonon_uq", "lib.jl"))
using SparseArrays, OSQP

outdir     = get(ENV, "OUTDIR", joinpath(@__DIR__, "results_rls")); mkpath(outdir)
vol_scales = [1.00, 1.02, 1.04, 1.06, 1.08, 1.10]
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
bps   = [bandpath_Dk(result, model, :Al, s*a_eq, 5) for s in vol_scales]
minω(θ) = [min_freq_stable(θ, bp) for bp in bps]

# ── constrained RLS: QP in θ̃ = Pθ, cutting plane over both volumes ──────────
osqp = OSQP.Model(); Hqp = sparse(Ap'*Ap .+ λ.*(P'*P)); qqp = -(Ap'*Yw)
function fit(rows, lower)
    A = vcat(sparse(b_prime'/P), sparse(rows/P)); l = vcat(0.0, lower); u = vcat(0.0, fill(Inf, length(lower)))
    OSQP.setup!(osqp; P=Hqp, q=qqp, A=A, l=l, u=u, max_iter=4_000_000, adaptive_rho_interval=25,
                verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    P \ OSQP.solve!(osqp).x
end
rows, lower = born_rows, born_lower
θ_con = fit(rows, lower)
for it in 1:80
    global θ_con, rows, lower
    soft = [(v, iq, e) for v in eachindex(bps) for (iq, e) in soft_modes(θ_con, bps[v], ω2_cut)]
    isempty(soft) && break
    rows  = vcat(rows, reduce(vcat, [cut_row(iq, e, bps[v])' for (v, iq, e) in soft]))
    lower = vcat(lower, fill(ω2_cut, length(soft)))
    θ_con = fit(rows, lower)
end

writedlm("$outdir/theta_rls_constrained.csv", θ_con, ',')
rmse(θ) = sqrt(mean((Ap*(P*θ) .- Yw).^2))
ωstr(θ) = join([@sprintf("@%d%% %+.3f", round(Int, 100s), w) for (s, w) in zip(vol_scales, minω(θ))], "  ")
@printf("%-16s min ω (THz) %s   weighted RMSE %.4g\n", "RLS", ωstr(lin_params), rmse(lin_params))
@printf("%-16s min ω (THz) %s   weighted RMSE %.4g   (%d cut rows, ‖Δθ‖ = %.3g)\n",
        "constrained RLS", ωstr(θ_con), rmse(θ_con), size(rows, 1) - size(born_rows, 1), norm(θ_con - lin_params))

# ── figure: one panel per volume, RLS (red) vs constrained RLS (blue) ───────
fig = Figure(size=(190*length(vol_scales), 260), figure_padding=(4, 10, 4, 4)); axs = Axis[]
for (v, s) in enumerate(vol_scales)
    bp = bps[v]
    ax = Axis(fig[1, v]; title="a = $(round(Int,100s))% a_eq", ylabel=v == 1 ? "Frequency (THz)" : "",
              xticks=(bp.x_ticks, bp.labels), xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    for (θ, col, lab) in ((lin_params, RGBf(0.80,0.15,0.15), "RLS"), (θ_con, RGBf(0.0,0.447,0.698), "constrained RLS"))
        F = bands(θ, bp)
        for b in axes(F, 1); lines!(ax, bp.x_vals, F[b,:]; color=col, linewidth=1.4, label=lab); end
    end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8); vlines!(ax, bp.x_ticks; color=(:black, 0.22), linewidth=0.6)
    xlims!(ax, extrema(bp.x_vals)...)
    push!(axs, ax); v == 1 && axislegend(ax; position=:rb, framevisible=false, labelsize=9, unique=true)
end
linkyaxes!(axs...)
save("$outdir/bands_rls_vs_constrained.pdf", fig); save("$outdir/bands_rls_vs_constrained.png", fig; px_per_unit=4)
println("outputs → $outdir")
