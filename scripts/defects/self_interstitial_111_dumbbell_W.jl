# self_interstitial_111_dumbbell_W.jl
#
# <111> dumbbell self-interstitial in BCC tungsten, against W_20_4_5A_3.
#
# The <111> dumbbell (equivalently the <111> crowdion, which it usually relaxes
# into) is the ground-state self-interstitial in BCC W. Construction:
#
#   1. Build an n x n x n cubic BCC supercell at the model's own relaxed a_eq.
#      A cubic BCC cell has 2 atoms, so the perfect cell has N = 2n^3 atoms.
#   2. Pick the lattice site nearest the cell centre, DELETE it, and insert two
#      atoms at r_site +/- (d/2) * u, with u = [111]/sqrt(3). The cell therefore
#      holds N+1 atoms: one extra atom = one self-interstitial.
#   3. Relax at FIXED CELL (constant volume) -- the formation energy below is
#      the constant-volume one, so the cell must not be allowed to breathe.
#
# The initial separation d is quoted as a fraction of the BCC nearest-neighbour
# distance d_nn = sqrt(3)/2 * a (the <111> chain spacing). The relaxed answer
# should not depend on it, so several starting fractions are relaxed and the
# lowest final energy is kept -- a cheap guard against the optimiser stalling on
# the saddle between the dumbbell and the crowdion.
#
# Formation energy at constant volume:
#
#     E_f = E_def(N+1) - (N+1)/N * E_perf(N)
#
# Literature (DFT, GGA) puts the W <111> self-interstitial at roughly 9.5-10.5 eV,
# so a fitted ACE model landing far outside that band is the interesting result,
# not a bug in this script.
#
# PARAMETERS. By default the model's own RLS lin_params are used. Pass a CSV of
# linear parameters as ARGS[1] to test a different fit -- e.g. the repulsive-core
# constrained vector, which is the one that matters if the dumbbell core probes
# pair distances below the training-data cutoff:
#
#     models/W_20_4_5A_3/positive_core_constrained_parameters.csv
#
# OUTPUTS -> models/W_20_4_5A_3/results/self_interstitial_111_dumbbell/
#   formation_energies.csv    one row per (supercell size, starting separation)
#   summary.csv               best relaxation per supercell size
#   dumbbell_n<N>_relaxed.extxyz   relaxed defect cells
#   formation_energy_vs_size.png   size convergence
#
# Run:  julia --project scripts/defects/self_interstitial_111_dumbbell_W.jl [params.csv] [sizes]
#   e.g. julia --project scripts/defects/self_interstitial_111_dumbbell_W.jl "" 3,4,5

using ACEWorkflow, ACEpotentials, AtomsBuilder, AtomsBase, Unitful, ExtXYZ
using AtomsCalculators: potential_energy
using GeometryOptimization
using LinearAlgebra, StaticArrays, Statistics, Printf, DelimitedFiles, CairoMakie

@async while true; flush(stdout); sleep(5); end     # Julia block-buffers to files

element    = :W
params_csv = length(ARGS) >= 1 && !isempty(ARGS[1]) ? ARGS[1] : nothing
sizes      = length(ARGS) >= 2 ? parse.(Int, split(ARGS[2], ',')) : [3, 4]
d_fracs    = [0.5, 0.6, 0.7]        # starting separations, in units of d_nn
maxiters   = 2000

result = load_model(element, 20, 4, 5, 3)
model  = result.model

param_label = "RLS"
if params_csv !== nothing
    θ = vec(readdlm(params_csv, ','))
    @assert length(θ) == length(result.lin_params) "params CSV has $(length(θ)) entries, model has $(length(result.lin_params))"
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    param_label = basename(params_csv)
end

outdir = joinpath(result.dir, "results", "self_interstitial_111_dumbbell")
mkpath(outdir)

a_eq = ACEWorkflow.relax_lattice_constant(model, element)
d_nn = sqrt(3) / 2 * a_eq                        # <111> nearest-neighbour spacing
@printf("model %s  params=%s\n", result.name, param_label)
@printf("a_eq = %.5f Å   d_nn(<111>) = %.5f Å\n\n", a_eq, d_nn)

# ── Geometry helpers ─────────────────────────────────────────────────────────

lattice_matrix_of(sys) = SMatrix{3,3,Float64}(ustrip.(reduce(hcat, sys.cell.cell_vectors)))

"""Minimum-image distance between two Cartesian points under lattice `L` (columns
are cell vectors). Triclinic-safe, though the cells here are cubic."""
function min_image_distance(r1, r2, L)
    Linv = inv(L)
    frac = Linv * (r1 .- r2)
    frac = frac .- round.(frac)
    return norm(L * frac)
end

atoms_of(sys) = [AtomsBase.Atom(sys[i].species, collect(ustrip.(sys[i].position)) .* u"Å", missing)
                 for i in 1:length(sys)]

"""
    dumbbell_111(element, a, n, d) -> (sys, idx_pair)

Perfect n×n×n cubic BCC supercell with the site nearest the cell centre replaced
by a <111> dumbbell of separation `d` (Å). Returns the (N+1)-atom system and the
indices of the two dumbbell atoms, which are appended last.
"""
function dumbbell_111(element::Symbol, a::Float64, n::Int, d::Float64)
    perfect = bulk(element; a=a*u"Å", cubic=true) * (n, n, n)
    L       = lattice_matrix_of(perfect)
    centre  = L * SVector(0.5, 0.5, 0.5)

    positions = [SVector{3,Float64}(ustrip.(perfect[i].position)) for i in 1:length(perfect)]
    site      = argmin([min_image_distance(p, centre, L) for p in positions])
    r_site    = positions[site]

    u   = SVector(1.0, 1.0, 1.0) ./ sqrt(3)
    keep = atoms_of(perfect)
    deleteat!(keep, site)                        # remove the host atom...
    push!(keep, AtomsBase.Atom(ChemicalSpecies(element), collect(r_site .- (d/2) .* u) .* u"Å", missing))
    push!(keep, AtomsBase.Atom(ChemicalSpecies(element), collect(r_site .+ (d/2) .* u) .* u"Å", missing))
    #                                            # ...and put two back: +1 atom

    sys = periodic_system(keep, perfect.cell.cell_vectors)
    return sys, (length(keep) - 1, length(keep))
end

energy_eV(sys, model) = ustrip(u"eV", potential_energy(sys, model))

# CSV field formatting: @printf needs a literal format string and a fixed arg
# count, neither of which suits these heterogeneous rows.
csv_field(x::Integer)       = string(x)
csv_field(x::Bool)          = string(x)
csv_field(x::AbstractFloat) = @sprintf("%.8g", x)
csv_field(x)                = string(x)
csv_row(r) = join(csv_field.(r), ",")

function write_extxyz(path, sys; info=Dict{String,Any}())
    pos = reduce(hcat, [collect(ustrip.(u"Å", sys[i].position)) for i in 1:length(sys)])
    L   = lattice_matrix_of(sys)
    frame = Dict{String,Any}(
        "N_atoms" => length(sys),
        "info"    => merge(Dict{String,Any}(
            "Lattice"    => join(string.(vec(L)), " "),
            "Properties" => "species:S:1:pos:R:3"), info),
        "arrays"  => Dict{String,Any}(
            "species" => [string(AtomsBase.atomic_symbol(sys, i)) for i in 1:length(sys)],
            "pos"     => pos),
    )
    ExtXYZ.write_frames(path, [frame])
end

# ── Sweep supercell size and starting separation ─────────────────────────────

rows    = Vector{Vector{Any}}()
summary = Vector{Vector{Any}}()

for n in sizes
    perfect  = bulk(element; a=a_eq*u"Å", cubic=true) * (n, n, n)
    N        = length(perfect)
    E_perf   = energy_eV(perfect, model)
    @printf("── n=%d: %d-atom perfect cell, E_perf = %.6f eV (%.6f eV/atom) ──\n",
            n, N, E_perf, E_perf / N)

    best = nothing
    for f in d_fracs
        d0  = f * d_nn
        sys, idx = dumbbell_111(element, a_eq, n, d0)
        @assert length(sys) == N + 1

        E_unrelaxed = energy_eV(sys, model)
        Ef_unrelaxed = E_unrelaxed - (N + 1) / N * E_perf

        res   = minimize_energy!(sys, model; variablecell=false, maxiters=maxiters)
        relaxed = res.system
        E_rel   = energy_eV(relaxed, model)
        Ef_rel  = E_rel - (N + 1) / N * E_perf

        L      = lattice_matrix_of(relaxed)
        r1     = SVector{3,Float64}(ustrip.(relaxed[idx[1]].position))
        r2     = SVector{3,Float64}(ustrip.(relaxed[idx[2]].position))
        d_rel  = min_image_distance(r1, r2, L)
        # Angle of the relaxed dumbbell axis to [111]: ~0° means it stayed on the
        # <111> line (dumbbell or crowdion); a large angle means it rotated.
        axis   = (r2 .- r1) ./ norm(r2 .- r1)
        cosang = abs(dot(axis, SVector(1.0, 1.0, 1.0) ./ sqrt(3)))
        angle  = rad2deg(acos(clamp(cosang, -1.0, 1.0)))

        # Smallest pair distance anywhere in the relaxed cell -- the core probes
        # short range, which is exactly where an unconstrained fit is unreliable.
        pos    = [SVector{3,Float64}(ustrip.(relaxed[i].position)) for i in 1:length(relaxed)]
        r_min  = minimum(min_image_distance(pos[i], pos[j], L)
                         for i in 1:length(pos) for j in i+1:length(pos))

        @printf("   d0/d_nn=%.2f  Ef_unrelaxed=%8.4f  Ef_relaxed=%8.4f eV  d_relaxed=%.4f Å (%.2f d_nn)  axis∠[111]=%5.2f°  r_min=%.4f Å\n",
                f, Ef_unrelaxed, Ef_rel, d_rel, d_rel / d_nn, angle, r_min)

        push!(rows, Any[n, N + 1, f, d0, Ef_unrelaxed, Ef_rel, d_rel, d_rel / d_nn,
                        angle, r_min, E_perf, E_rel])

        if best === nothing || Ef_rel < best.Ef
            best = (Ef=Ef_rel, f=f, sys=relaxed, d=d_rel, angle=angle, r_min=r_min)
        end
    end

    write_extxyz(joinpath(outdir, "dumbbell_n$(n)_relaxed.extxyz"), best.sys;
                 info=Dict{String,Any}("E_formation_eV" => best.Ef,
                                       "d_dumbbell_Ang" => best.d,
                                       "a_eq_Ang"       => a_eq,
                                       "params"         => param_label))
    push!(summary, Any[n, N + 1, best.Ef, best.f, best.d, best.d / d_nn, best.angle, best.r_min])
    @printf("   → best Ef = %.4f eV (from d0/d_nn = %.2f)\n\n", best.Ef, best.f)
end

# ── Persist ──────────────────────────────────────────────────────────────────

open(joinpath(outdir, "formation_energies.csv"), "w") do io
    println(io, "# model=$(result.name) params=$(param_label) a_eq=$(a_eq) d_nn=$(d_nn)")
    println(io, "n_cell,n_atoms,d0_over_dnn,d0_Ang,Ef_unrelaxed_eV,Ef_relaxed_eV," *
                "d_relaxed_Ang,d_relaxed_over_dnn,axis_angle_to_111_deg,r_min_Ang,E_perfect_eV,E_defect_eV")
    for r in rows
        println(io, csv_row(r))
    end
end

open(joinpath(outdir, "summary.csv"), "w") do io
    println(io, "# model=$(result.name) params=$(param_label) a_eq=$(a_eq)")
    println(io, "n_cell,n_atoms,Ef_eV,best_d0_over_dnn,d_relaxed_Ang,d_relaxed_over_dnn,axis_angle_to_111_deg,r_min_Ang")
    for r in summary
        println(io, csv_row(r))
    end
end

if length(summary) >= 2
    fig = Figure(size=(700, 450), fontsize=16)
    ax  = Axis(fig[1, 1];
               title  = "W <111> dumbbell — formation energy vs supercell size ($(param_label))",
               xlabel = "Atoms in perfect cell", ylabel = "E_f (eV)")
    scatterlines!(ax, [Float64(s[2] - 1) for s in summary], [Float64(s[3]) for s in summary];
                  color=:steelblue, markersize=12)
    hspan!(ax, 9.5, 10.5; color=(:green, 0.12))
    text!(ax, 0.05, 0.9; text="DFT band 9.5–10.5 eV", space=:relative, fontsize=13, color=:darkgreen)
    save(joinpath(outdir, "formation_energy_vs_size.png"), fig; px_per_unit=2)
end

@printf("outputs → %s/\n", outdir)
