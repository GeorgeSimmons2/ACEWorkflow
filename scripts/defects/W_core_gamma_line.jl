# W_core_gamma_line.jl
#
# {110}<111> gamma-line of BCC W and the Duesbery-Vitek core-degeneracy check,
# against W_20_4_5A_3. Working version of the W_core.jl draft (left untouched):
# same physics, but no ASE/PythonCall (not in Project.toml), the model actually
# loaded, and AtomsBase 0.5 constructors.
#
#     ΔDV = γ(b/3) - 2γ(b/6)       > 0  ⇒ non-degenerate core   (W_core.jl convention)
#
# Geometry. Everything is written in cubic crystal axes. The (-110) plane holds
# the in-plane lattice vectors
#
#     v1 = a/2 [111]   (= b, the slip direction)      v2 = a [001]
#
# and v3 = a [100] steps exactly one {110} plane (height a/√2). (v1, v2, v3) is a
# primitive BCC basis (det = a³/2), so a slab of `nplanes` {110} planes is one
# atom per plane at k·a[100], in the cell (v1, v2, nplanes·v3). This is the
# same lattice as the ASE cell in W_core.jl (nz = 4 there ⇔ nplanes = 8 here,
# 17.9 Å between faults), just the smallest cell that carries it.
#
# Faulting: the third cell vector is tilted by u·[111]/√3, so the periodic
# boundary between plane nplanes-1 and the image of plane 0 carries a rigid
# shift u. One fault per cell, area A = |v1 × v2| = a²/√2.
#
# Two gamma-lines are reported:
#   unrelaxed – rigid shift, as in the draft;
#   relaxed   – each plane may move along the fault normal only (in-plane
#               positions frozen), the usual convention for gamma-surfaces and
#               the one Duesbery-Vitek used. Fixed cell, plane 0 pinned.
#
# PARAMETERS. By default the model's own RLS lin_params. Pass a CSV of linear
# parameters as ARGS[1] to test another fit, e.g.
#     models/W_20_4_5A_3/positive_core_constrained_parameters.csv
# a_eq is relaxed for whichever parameters are loaded.
#
# OUTPUTS -> models/W_20_4_5A_3/results/W_core_gamma_line/
#   gamma_line_<label>.csv     u/b, γ_unrelaxed, γ_relaxed  (J/m²)
#   gamma_line_<label>.png
#   summary_<label>.csv        a_eq, γ(b/6), γ(b/3), ΔDV for both conventions
#
# Run:  julia --project scripts/defects/W_core_gamma_line.jl [params.csv] [nplanes]
#   e.g. julia --project scripts/defects/W_core_gamma_line.jl "" 12

using ACEWorkflow, ACEpotentials, AtomsBase, Unitful
using AtomsCalculators: potential_energy, forces
using LinearAlgebra, StaticArrays, Printf, DelimitedFiles, Optim, CairoMakie

@async while true; flush(stdout); sleep(5); end     # Julia block-buffers to files

element    = :W
params_csv = length(ARGS) >= 1 && !isempty(ARGS[1]) ? ARGS[1] : nothing
nplanes    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 12   # 12·a/√2 ≈ 27 Å ≫ r_cut = 5 Å
npts       = 41                                             # full γ-line, u ∈ [0, b]

result = load_model(element, 20, 4, 5, 3)
model  = result.model

label = "RLS"
if params_csv !== nothing
    θ = vec(readdlm(params_csv, ','))
    @assert length(θ) == length(result.lin_params) "params CSV has $(length(θ)) entries, model has $(length(result.lin_params))"
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    label = splitext(basename(params_csv))[1]
end

outdir = joinpath(result.dir, "results", "W_core_gamma_line")
mkpath(outdir)

a  = ACEWorkflow.relax_lattice_constant(model, element)
v1 = SVector(0.5, 0.5, 0.5) .* a          # b = a√3/2, along [111]
v2 = SVector(0.0, 0.0, 1.0) .* a
v3 = SVector(1.0, 0.0, 0.0) .* a          # one {110} plane
n̂  = SVector(-1.0, 1.0, 0.0) ./ sqrt(2)   # fault normal
x̂  = v1 ./ norm(v1)
b  = norm(v1)
A  = norm(cross(v1, v2))                  # Å², = a²/√2
eVÅ2_to_Jm2 = 16.0217663

@printf("model %s  params=%s\n", result.name, label)
@printf("a_eq = %.5f Å   b = %.5f Å   A = %.5f Å²   nplanes = %d (%.2f Å between faults)\n\n",
        a, b, A, nplanes, nplanes * a / sqrt(2))

"""Slab with fault shift `u` (Å) and per-plane normal displacements `s` (Å)."""
function slab(u, s)
    atoms = [AtomsBase.Atom(element, collect((k - 1) .* v3 .+ s[k] .* n̂) .* u"Å")
             for k in 1:nplanes]
    cell  = (collect(v1) .* u"Å", collect(v2) .* u"Å",
             collect(nplanes .* v3 .+ u .* x̂) .* u"Å")
    return periodic_system(atoms, cell)
end

energy(u, s) = ustrip(u"eV", potential_energy(slab(u, s), model))

"""Minimise E over normal displacements of planes 2..nplanes (plane 1 pinned)."""
function relaxed_energy(u)
    full(x) = vcat(0.0, x)
    f(x) = energy(u, full(x))
    function g!(G, x)
        F = forces(slab(u, full(x)), model)
        for k in 2:nplanes
            G[k-1] = -dot(ustrip.(u"eV/Å", F[k]), n̂)
        end
        return G
    end
    res = optimize(f, g!, zeros(nplanes - 1), LBFGS(),
                   Optim.Options(g_tol = 1e-5, iterations = 500))
    Optim.converged(res) || @warn "normal relaxation not converged at u/b = $(round(u/b, digits=3))"
    return Optim.minimum(res)
end

zero_s = zeros(nplanes)
E0     = energy(0.0, zero_s)
γ_unrel(u) = (energy(u, zero_s) - E0) / A * eVÅ2_to_Jm2
γ_rel(u)   = (relaxed_energy(u)  - E0) / A * eVÅ2_to_Jm2

# ── Duesbery-Vitek ───────────────────────────────────────────────────────────

γ6u, γ3u = γ_unrel(b/6), γ_unrel(b/3)
γ6r, γ3r = γ_rel(b/6),   γ_rel(b/3)
ΔDVu = γ3u - 2γ6u
ΔDVr = γ3r - 2γ6r
verdict(Δ) = Δ > 0 ? "non-degenerate" : "DEGENERATE"

@printf("             γ(b/6)     γ(b/3)     ΔDV = γ(b/3) - 2γ(b/6)   [J/m²]\n")
@printf("unrelaxed  %9.5f  %9.5f  %+10.5f   → %s\n", γ6u, γ3u, ΔDVu, verdict(ΔDVu))
@printf("relaxed    %9.5f  %9.5f  %+10.5f   → %s\n\n", γ6r, γ3r, ΔDVr, verdict(ΔDVr))

open(joinpath(outdir, "summary_$(label).csv"), "w") do io
    println(io, "params,a_eq,nplanes,relaxation,gamma_b6,gamma_b3,dDV,verdict")
    @printf(io, "%s,%.6f,%d,unrelaxed,%.8g,%.8g,%.8g,%s\n", label, a, nplanes, γ6u, γ3u, ΔDVu, verdict(ΔDVu))
    @printf(io, "%s,%.6f,%d,relaxed,%.8g,%.8g,%.8g,%s\n",   label, a, nplanes, γ6r, γ3r, ΔDVr, verdict(ΔDVr))
end

# ── Full γ-line ──────────────────────────────────────────────────────────────

ξ  = collect(range(0, 1, npts))
γu = [γ_unrel(x * b) for x in ξ]
γr = [γ_rel(x * b)   for x in ξ]
writedlm(joinpath(outdir, "gamma_line_$(label).csv"),
         vcat(["u_over_b" "gamma_unrelaxed_Jm2" "gamma_relaxed_Jm2"], hcat(ξ, γu, γr)), ',')

fig = Figure(size = (510, 340), fontsize = 13)
ax  = Axis(fig[1, 1]; xlabel = "u / b", ylabel = "γ  (J/m²)",
           title = @sprintf("W {110}<111>, %s   ΔDV = %+.3f J/m² (relaxed)", label, ΔDVr))
lines!(ax, ξ, γu; label = "unrelaxed", linestyle = :dash)
lines!(ax, ξ, γr; label = "relaxed ⊥")
scatter!(ax, [1/6, 1/3], [γ6r, γ3r]; color = :black, markersize = 8)
vlines!(ax, [1/6, 1/3]; color = (:gray, 0.4), linestyle = :dot)
axislegend(ax; position = :ct)
save(joinpath(outdir, "gamma_line_$(label).png"), fig; px_per_unit = 3)

println("wrote ", outdir)
