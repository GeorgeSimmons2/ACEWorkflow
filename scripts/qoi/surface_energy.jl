using ACEpotentials
using GeometryOptimization
using LinearAlgebra
using Unitful
using AtomsBase
using AtomsCalculators: potential_energy
# ----------------------------
# Build FCC bulk (primitive)
# ----------------------------
function build_bulk_fcc(a=4.05)
    cell = a/2 * [
        0 1 1;
        1 0 1;
        1 1 0
    ]

    positions = [[0.0, 0.0, 0.0]]
    atoms = [Atom(:Al, cell * pos .* u"Å") for pos in positions]

    system = periodic_system(atoms, [cell[:, i] for i in 1:3] .* u"Å")
    return system
end

# ----------------------------
# Build FCC (111) slab
# ----------------------------
function build_fcc111_slab(a, layers, vacuum)
    a2d = a / sqrt(2)

    a1 = a2d * [1.0, 0.0, 0.0]
    a2 = a2d * [0.5, sqrt(3)/2, 0.0]

    d = a / sqrt(3)
    height = layers * d + vacuum
    a3 = [0.0, 0.0, height]

    shifts = [
        [0.0, 0.0],
        [1/3, 1/3],
        [2/3, 2/3]
    ]

    positions = Vector{Vector{Float64}}()

    for i in 0:layers-1
        shift = shifts[mod(i, 3) + 1]
        z = i * d

        pos = shift[1] * a1 + shift[2] * a2 + [0.0, 0.0, z]
        push!(positions, pos)
    end

    atoms = [Atom(:Al, pos .* u"Å") for pos in positions]
    system = periodic_system(atoms, [a1, a2, a3] .* u"Å")

    return system
end

# ----------------------------
# Surface energy calculation
# ----------------------------
function calculate_surface_energy(model; a=4.05, layers=8, vacuum=12.0)

    # ---- 1. Relax bulk ----
    sys_bulk = build_bulk_fcc(a)
    res_bulk = minimize_energy!(sys_bulk, model; variablecell=true)
    sys_bulk_opt = res_bulk.system

    e_bulk = ustrip(potential_energy(sys_bulk_opt, model))
    n_bulk = length(sys_bulk_opt)
    e_bulk_per_atom = e_bulk / n_bulk

    # ✅ Extract lattice constant from nearest-neighbour distance
    cell = AtomsBase.cell(sys_bulk_opt)

    # Extract lattice vectors via eachindex (works across versions)
    # latvecs = [ustrip.(cell[i]) for i in eachindex(cell)]

    # primitive FCC vector length = a / sqrt(2)
    a_opt = ustrip.(norm(cell.cell_vectors[1]) * 2)

    # ---- 2. Build slab ----
    sys_slab = build_fcc111_slab(a_opt, layers, vacuum)

    # ---- 3. Relax slab ----
    res_slab = minimize_energy!(sys_slab, model; variablecell=false)
    sys_slab_opt = res_slab.system

    e_slab = ustrip(potential_energy(sys_slab_opt, model))
    n_slab = length(sys_slab_opt)

    # ---- 4. Surface area (analytic) ----
    a2d = a_opt / sqrt(2)

    # Area of hex cell
    area = norm(cross(
        a2d * [1.0, 0.0, 0.0],
        a2d * [0.5, sqrt(3)/2, 0.0]
    ))

    # ---- 5. Surface energy ----
    gamma_ev_ang2 = (e_slab - n_slab * e_bulk_per_atom) / (2 * area)
    gamma_jm2 = gamma_ev_ang2 * 16.0218

    return (
        gamma_ev_ang2 = gamma_ev_ang2,
        gamma_jm2 = gamma_jm2,
        area = area,
        n_atoms = n_slab,
        e_bulk_per_atom = e_bulk_per_atom
    )
end

# ----------------------------
# Example usage
# ----------------------------
using DelimitedFiles

POPS_corrections = readdlm("small_high_entropy_ace_model/constrained_pops.csv", ',')
model, _ = ACEpotentials.load_model("small_high_entropy_ace_model/model.json")

n_pops = size(POPS_corrections, 1)
surface_energies = Vector{Float64}(undef, n_pops)

# Each thread needs its own copy of the model to avoid data races on set_linear_parameters!
thread_models = [deepcopy(model) for _ in 1:Threads.nthreads()]

Threads.@threads for i in 1:n_pops
    m = thread_models[Threads.threadid()]
    ACEpotentials.Models.set_linear_parameters!(m, POPS_corrections[i,:])

    results = calculate_surface_energy(m)
    surface_energies[i] = results.gamma_jm2

    if results.gamma_jm2 < 0.0
        println("[$i] Negative surface energy: $(results.gamma_ev_ang2) eV/Å²")
    else
        println("[$i] γ = $(results.gamma_jm2) J/m²")
    end
end

writedlm("small_high_entropy_ace_model/surface_energies.csv", surface_energies, ',')

using CairoMakie

fig = Figure()
ax = Axis(fig[1,1],
    xlabel = "Surface energy (J/m²)",
    ylabel = "Count",
    title  = "POPS Surface Energy Distribution")
hist!(ax, surface_energies; bins=50, color=(:steelblue, 0.7), strokecolor=:black, strokewidth=0.5)
save("small_high_entropy_ace_model/surface_energy_hist.png", fig)