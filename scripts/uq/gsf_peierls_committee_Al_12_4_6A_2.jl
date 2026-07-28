# gsf_peierls_committee_Al_12_4_6A_2.jl
#
# Generalized stacking-fault energy and Peierls–Nabarro stress across the POPS
# ensembles: constrained + phonon-rejection versus naive.
#
# WHY THIS IS CHEAP.  The rigid (unrelaxed) GSF energy is a LINEAR functional of the
# parameters, exactly like the equation-of-state and Born rows:
#
#     γ(t;c) = [E(t) − E(0)]/A = ([B(t) − B(0)]·c)/A  ≡  (Δb(t)·c)/A
#
# because the fault displacement moves atoms rigidly and the geometry does not depend
# on c.  So instead of 30 members × 21 displacements = 630 energy evaluations, we
# evaluate the BASIS once per displacement (21 evaluations) and obtain every member by
# a dot product.  The script verifies this against direct energy evaluation.
#
# The same linearity means γ_us is directly CONSTRAINABLE as a QP row,
#     Δb(t*)·c / A = γ_us^target,
# in exactly the machinery already used for b′·c = 0 and the Born criteria — see the
# note at the end of this file.
#
# NOTE: linearity holds only for the RIGID γ-surface.  A relaxed γ-surface is not
# linear in c, because the relaxed geometry itself depends on the parameters.  That is
# both why this works and the honest limitation to state.
#
# Peierls stress follows from the sinusoidal Peierls–Nabarro closed form using each
# member's OWN elastic constants, which are themselves linear in c:
#     C_ij(c) = Σ_k c_k H^el_{ij,k}
# so τ_P is a nonlinear function of two exactly-linear functionals of c.
#
# Reuses scripts/qoi/peierls_stress_gsf_pn.jl unmodified.
#
# Outputs → results/gsf_peierls_committee/
#   gsf_curves.pdf/png        γ(u) per committee, mean ± 1σ
#   gamma_us_hist.pdf/png     distribution of γ_us across members
#   peierls_scatter.pdf/png   τ_P against γ_us, coloured by committee
#   gsf_peierls_summary.csv   per-member γ_us, C11, C12, C44, G, ν, τ_P
#
# Run:  julia --project -t 4 scripts/uq/gsf_peierls_committee_Al_12_4_6A_2.jl

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
include(joinpath(@__DIR__, "..", "qoi", "peierls_stress_gsf_pn.jl"))
@async while true; flush(stdout); sleep(5); end

element, dataset = :Al, ""
layers   = 24
vacuum   = 15.0
n_steps  = 20
character = :edge
n_check  = 3          # members re-done by direct energy evaluation as a cross-check

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model, lin = result.model, result.lin_params
d = length(lin)
RES = "$(result.dir)/results"
SRC = "$RES/aeq_cheap_vs_expensive"
outdir = "$RES/gsf_peierls_committee"; mkpath(outdir)
θ_con = vec(readdlm("$RES/bandpath_undotted_multivolume/theta_mean.csv", ','))

ACEpotentials.Models.set_linear_parameters!(model, lin)
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@printf("a_eq = %.5f Å; slab %d layers, %.1f Å vacuum, %d displacements\n",
        a_eq, layers, vacuum, n_steps+1); flush(stdout)

# ── the GSF basis: Δb(t) = B(slab at t) − B(slab at 0), one vector per t ─────
# Evaluated ONCE; every member is then a dot product.  A common a_eq is required for
# this (a per-member geometry would break the shared linear functional); the
# constrained members are pinned to a_eq by b′·c = 0 anyway.
ts = collect(range(0.0, 1.0, length=n_steps+1))
sys0, area = fcc111_gsf_slab(element, a_eq, layers, vacuum, 0.0)
b0 = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys0, model))
Δb = Matrix{Float64}(undef, length(ts), d)
for (k, t) in enumerate(ts)
    syst = t == 0.0 ? sys0 : fcc111_gsf_slab(element, a_eq, layers, vacuum, t)[1]
    Δb[k, :] = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(syst, model)) .- b0
end
a2d = a_eq/sqrt(2)
b_partial = norm(((a2d*[1.0,0.0,0.0]) .+ (a2d*[0.5,sqrt(3)/2,0.0])) ./ 3)
@printf("  basis evaluated at %d displacements; area = %.4f Å², b_partial = %.4f Å\n\n",
        length(ts), area, b_partial); flush(stdout)

gsf_gamma(c) = (Δb * c) ./ area              # eV/Å² at each t — one matvec per member

# ── elastic constants, also linear in c ─────────────────────────────────────
sys_el = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0 = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys_el.cell.cell_vectors))
eV_to_GPa = 160.2176621/abs(det(L0))
H_el = elastic_hessian_basis(model; element=element, a=a_eq)
c11r = reshape(H_el,36,d)[1,:]; c12r = reshape(H_el,36,d)[7,:]; c44r = reshape(H_el,36,d)[22,:]
elastic(c) = (dot(c11r,c)*eV_to_GPa, dot(c12r,c)*eV_to_GPa, dot(c44r,c)*eV_to_GPa)

# ── cross-check the linear route against direct energy evaluation ───────────
println("── cross-check: linear Δb·c/A against direct E(t)−E(0) ──"); flush(stdout)
Θchk = readdlm("$SRC/committee_A_cheap_phononreject.csv", ',')
worst = 0.0
for k in 1:min(n_check, size(Θchk,1))
    c = collect(Float64, Θchk[k,:])
    ACEpotentials.Models.set_linear_parameters!(model, c)
    direct = gsf_curve(model, element, a_eq; layers=layers, vacuum=vacuum, n_steps=n_steps).gamma
    lin_g  = gsf_gamma(c)
    e = maximum(abs.(direct .- lin_g))
    worst = max(worst, e)
    @printf("  member %d: max|direct − linear| = %.3e eV/Å²  (γ_us = %.5f)\n", k, e, maximum(lin_g))
end
@printf("  worst over %d members: %.3e eV/Å²  →  %s\n\n", min(n_check,size(Θchk,1)), worst,
        worst < 1e-9 ? "LINEARITY CONFIRMED" : "*** DISCREPANCY — investigate ***"); flush(stdout)
ACEpotentials.Models.set_linear_parameters!(model, lin)

# ── evaluate both committees ────────────────────────────────────────────────
committees = [
 ("constrained", "$SRC/committee_A_cheap_phononreject.csv", θ_con, RGBf(0.0,0.447,0.698)),
 ("naive",       "$SRC/committee_B_naive.csv",              lin,   RGBf(0.80,0.15,0.15)),
]
store = Dict{String,Any}(); rows = String[]
for (tag, csv, centre, col) in committees
    isfile(csv) || (@warn "missing $csv"; continue)
    Θ = readdlm(csv, ','); N = size(Θ,1)
    G = reduce(hcat, [gsf_gamma(collect(Float64, Θ[k,:])) for k in 1:N])   # (n_t × N)
    γus = vec(maximum(G; dims=1))
    Cs  = [elastic(collect(Float64, Θ[k,:])) for k in 1:N]
    pn  = [peierls_nabarro_stress(C[1], C[2], C[3], γus[k], b_partial; character=character)
           for (k,C) in enumerate(Cs)]
    gc  = gsf_gamma(centre); γus_c = maximum(gc)
    Cc  = elastic(centre)
    pnc = peierls_nabarro_stress(Cc[1], Cc[2], Cc[3], γus_c, b_partial; character=character)
    store[tag] = (; G, γus, Cs, pn, gc, γus_c, pnc, col, N)
    @printf("── %s (%d members) ──\n", tag, N)
    @printf("   γ_us   : centre %.5f | members %.5f ± %.5f eV/Å²  (%.1f ± %.1f mJ/m²)\n",
            γus_c, mean(γus), std(γus), mean(γus)*16021.77, std(γus)*16021.77)
    @printf("   C11/C12/C44 (GPa): centre %.1f/%.1f/%.1f\n", Cc...)
    @printf("   τ_P    : centre %.4f | members %.4f ± %.4f GPa  (range %.4f – %.4f)\n",
            pnc.tau_P, mean(getfield.(pn,:tau_P)), std(getfield.(pn,:tau_P)),
            minimum(getfield.(pn,:tau_P)), maximum(getfield.(pn,:tau_P)))
    @printf("   negative γ_us (unphysical): %d / %d\n\n", count(<(0), γus), N); flush(stdout)
    for k in 1:N
        C = Cs[k]; p = pn[k]
        push!(rows, @sprintf("%s,%d,%.6f,%.4f,%.2f,%.2f,%.2f,%.2f,%.4f,%.5f",
                             tag, k, γus[k], γus[k]*16021.77, C[1], C[2], C[3],
                             p.G, p.ν, p.tau_P))
    end
end

# ── plots ────────────────────────────────────────────────────────────────────
let fig = Figure(size=(520,340), figure_padding=(6,10,4,6))
    ax = Axis(fig[1,1]; xlabel="partial displacement  u / b", ylabel="γ (eV/Å²)",
              title="Rigid {111}⟨112⟩ γ-surface, ensemble mean ± 1σ", titlesize=10,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    for (tag, st) in store
        μ = vec(mean(st.G; dims=2)); σ = vec(std(st.G; dims=2))
        band!(ax, ts, μ.-σ, μ.+σ; color=(st.col, 0.20))
        lines!(ax, ts, μ; color=st.col, linewidth=1.6)
        lines!(ax, ts, st.gc; color=st.col, linewidth=1.0, linestyle=:dash)
    end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    elem = [LineElement(color=st.col) for (_, st) in store]
    Legend(fig[1,1], elem, collect(keys(store)); tellwidth=false, tellheight=false,
           halign=:left, valign=:top, margin=(8,8,8,8), framevisible=true, labelsize=8)
    save("$outdir/gsf_curves.pdf", fig); save("$outdir/gsf_curves.png", fig; px_per_unit=4)
end

let fig = Figure(size=(460,320), figure_padding=(6,10,4,6))
    ax = Axis(fig[1,1]; xlabel="γ_us (mJ/m²)", ylabel="members", titlesize=10,
              title="Unstable stacking-fault energy across the ensemble",
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    for (tag, st) in store
        hist!(ax, st.γus .* 16021.77; bins=15, color=(st.col,0.55), strokewidth=0.4)
    end
    vlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    text!(ax, 0.98, 0.95; text="DFT/expt Al: γ_us ≈ 160–180 mJ/m²", space=:relative,
          align=(:right,:top), fontsize=8, color=(:black,0.7))
    save("$outdir/gamma_us_hist.pdf", fig); save("$outdir/gamma_us_hist.png", fig; px_per_unit=4)
end

let fig = Figure(size=(460,340), figure_padding=(6,10,4,6))
    ax = Axis(fig[1,1]; xlabel="γ_us (mJ/m²)", ylabel="Peierls stress τ_P (GPa)",
              title="Peierls–Nabarro τ_P from each member's own γ_us and C_ij",
              titlesize=10, xgridvisible=false, ygridvisible=false,
              xtickalign=1, ytickalign=1)
    for (tag, st) in store
        scatter!(ax, st.γus .* 16021.77, getfield.(st.pn, :tau_P);
                 color=(st.col,0.65), markersize=7)
        scatter!(ax, [st.γus_c*16021.77], [st.pnc.tau_P];
                 color=st.col, marker=:diamond, markersize=12)
    end
    save("$outdir/peierls_scatter.pdf", fig); save("$outdir/peierls_scatter.png", fig; px_per_unit=4)
end

open("$outdir/gsf_peierls_summary.csv", "w") do io
    println(io, "# rigid {111}<112> GSF at a_eq = $a_eq A, $layers layers, $vacuum A vacuum")
    println(io, "# gamma linear in c: gamma(t) = (Delta_b(t) . c)/A; linearity check max err = $worst eV/A^2")
    println(io, "# PN character = $character, b_partial = $b_partial A")
    println(io, "committee,member,gamma_us_eV_per_A2,gamma_us_mJ_per_m2,C11_GPa,C12_GPa,C44_GPa,G_GPa,poisson,tau_P_GPa")
    foreach(r -> println(io, r), rows)
end
writedlm("$outdir/gsf_basis_delta.csv",
         vcat(permutedims(vcat("t", ["dB_$k" for k in 1:d])), hcat(ts, Δb)), ',')
println("outputs → $outdir/")
println("  gsf_basis_delta.csv holds Δb(t): row t dotted with any c gives A·γ(t),")
println("  i.e. the constraint row for pinning γ_us to a target value.")
ACEpotentials.Models.set_linear_parameters!(model, lin)
