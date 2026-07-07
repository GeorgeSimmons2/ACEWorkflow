using ACEpotentials, EmpiricalPotentials, ACEWorkflow, OSQP, SparseArrays
using ACEpotentials.Models: potential_energy_basis
using AtomsBase
using StaticArrays
using StatsBase
using Unitful, ExtXYZ
element = :W
totdeg  = 20
prior_param = 4
rcut = 5
v = 3
result = load_model(element, totdeg, prior_param, rcut, v)
model = result.model
W_zbl = ZBL(5.0*u"Å") # Tungsten ZBL potential
lattice_constants = LinRange(0.21, 2.18, 50)

function bulk_energy_basis(element::Symbol, lattice_constants::AbstractVector{Float64}, model)

    energy_bases    = []

    for a in lattice_constants
        ats = bulk(element, a=a*u"Å")
        E   = potential_energy_basis(ats, model)
        push!(energy_bases, ustrip.(E))
    end

    return energy_bases
end


function bulk_energy_curve(element::Symbol, lattice_constants::AbstractVector{Float64}, model)

    energies    = []

    for a in lattice_constants
        ats = bulk(element, a=a*u"Å")
        E = potential_energy(ats, model)
        push!(energies, ustrip(E))
    end

    return energies
end

using AtomsBuilder, LinearAlgebra
import AtomsCalculators: potential_energy

bulk_bases    = bulk_energy_basis(:W, lattice_constants, model)
bulk_energies = [dot(bulk_basis, result.lin_params) for bulk_basis in bulk_bases]
zbl_energies  = bulk_energy_curve(:W, lattice_constants, W_zbl)

function constrained_ridge_regression(X_train, Y_train, Gamma, constraint_matrix, constraint_bounds)
    H = (X_train' * X_train .+ (1.0 / (size(X_train, 1)) .* Gamma' * Gamma))
    b = - X_train' * Y_train
    model = OSQP.Model()
    OSQP.setup!(model; P=sparse(H), q=b, A=sparse(constraint_matrix / Gamma), l=constraint_bounds[1], u=constraint_bounds[2],
                max_iter=500_000, check_termination=1_000, verbose=true)
    results = OSQP.solve!(model)
    return Gamma \ results.x
end

constraint_matrix = Matrix(reduce(hcat, bulk_bases)')
bounds = (Vector{Float64}(zeros(length(bulk_bases))), Vector{Float64}(ones(length(bulk_bases)) .* Inf))

Ap = Diagonal(result.W) * result.A / result.P
Yw = result.W .* result.Y
Gamma = result.P
P = Gamma
# C = Gamma' * Gamma .* (1 / length(Yw)) .+ Ap' * Ap
# A      = C \ Ap'
# leverage = diag(Ap * A)
# constrained_errors = Yw .- (Ap * (P \ constrained_parameters))
# unconstrained_errors = Yw .- (Ap * (P \ result.lin_params))
# constrained_pointwise_corrections = ((P \ (A' .* (constrained_errors ./ leverage))') .+ constrained_parameters)'
# unconstrained_pointwise_corrections = (P \ (A' .* (unconstrained_errors ./ leverage))' .+ result.lin_params)'


# ace_positive_core_constrained_parameters = constrained_ridge_regression(Ap, Yw, Gamma, constraint_matrix, bounds)
ace_positive_core_constrained_parameters = vec(readdlm("$(result.dir)/positive_core_constrained_parameters.csv", ','))
ace_positive_core_model = deepcopy(model)
unc_model = deepcopy(model)
ACEpotentials.Models.set_linear_parameters!(ace_positive_core_model, ace_positive_core_constrained_parameters)
ACEpotentials.Models.set_linear_parameters!(unc_model, result.lin_params)

full_lattice_constants = vcat(collect(lattice_constants), collect(LinRange(maximum(lattice_constants), 2 * maximum(lattice_constants), 50)[2:end]))
full_bulk_bases    = bulk_energy_basis(:W, full_lattice_constants, model)
constrained_bulk_energies = [dot(bulk_basis, ace_positive_core_constrained_parameters) for bulk_basis in full_bulk_bases]
unconstrained_bulk_energies = [dot(bulk_basis, result.lin_params) for bulk_basis in full_bulk_bases]
zbl_energies  = bulk_energy_curve(:W, full_lattice_constants, W_zbl)
constrained_bulk_energies_add_zbl = constrained_bulk_energies .+ zbl_energies
unconstrained_bulk_energies_add_zbl = unconstrained_bulk_energies .+ zbl_energies

using LinearAlgebra, DelimitedFiles
using Unitful, ExtXYZ

atoms = ExtXYZ.load("data/W/df_W_train.extxyz")

using LinearAlgebra
using Unitful

function pair_distances(sys)

    N    = length(sys)
    dists = Float64[]

    # 3x3 matrix of lattice vectors (columns), per-config since each
    # training structure has its own (generally triclinic) cell
    L    = ustrip.(reduce(hcat, sys[:cell_vectors]))
    Linv = inv(L)

    for i in 1:N-1
        ri = ustrip.(position(sys, i))

        for j in i+1:N
            rj = ustrip.(position(sys, j))

            dr = ri .- rj

            # general (triclinic-safe) minimum image via fractional coordinates
            frac = Linv * dr
            frac = frac .- round.(frac)
            dr   = L * frac

            push!(dists, norm(dr))
        end
    end

    return dists
end

all_dists = reduce(vcat, [pair_distances(at) for at in atoms])

using Test
using LinearAlgebra

# -----------------------------
# simple fake system generator
# -----------------------------
struct FakeAtom
    position::Vector{Float64}
end

FakeSys = Vector{FakeAtom}

# -----------------------------
# reference naive distances
# -----------------------------
function naive_distances(sys)
    N = length(sys)
    d = Float64[]

    for i in 1:N-1
        for j in i+1:N
            push!(d, norm(sys[i].position .- sys[j].position))
        end
    end

    return d
end

# -----------------------------
# periodic minimum image (reference implementation)
# -----------------------------
function minimum_image_ref(r, L)
    return r .- L .* round.(r ./ L)
end

function periodic_distances_ref(sys, Lx, Ly, Lz)
    N = length(sys)
    d = Float64[]

    for i in 1:N-1
        for j in i+1:N
            dr = sys[i].position .- sys[j].position
            dr = minimum_image_ref(dr, [Lx, Ly, Lz])
            push!(d, norm(dr))
        end
    end

    return d
end

# -----------------------------
# test system
# -----------------------------
sys = FakeSys([
    FakeAtom([0.0, 0.0, 0.0]),
    FakeAtom([0.9, 0.0, 0.0]),
    FakeAtom([1.8, 0.0, 0.0])
])

Lx = 2.0
Ly = 2.0
Lz = 2.0

# -----------------------------
# TESTS
# -----------------------------
@testset "pair distances with PBC" begin

    d = periodic_distances_ref(sys, Lx, Ly, Lz)

    # 3 atoms → 3 pair distances
    @test length(d) == 3

    # symmetry check (order-independent)
    @test sort(d) ≈ sort(d)

    # known minimum-image result:
    # distances along x:
    # 0->0.9 = 0.9
    # 0->1.8 = min(1.8, 0.2) = 0.2  (wrap!)
    # 0.9->1.8 = 0.9
    expected = sort([0.9, 0.2, 0.9])

    @test sort(d) ≈ expected atol=1e-12

end

e_lo, e_hi = -14.0, 30.0   # eV — adjust to taste
using CairoMakie
using StatsBase
using CairoMakie

fig = Figure(
    size = (900, 1000),   # taller for stacked panels
    fontsize = 18         # base font size for everything
)

# -----------------------------
# TOP: EOS curves
# -----------------------------
ax1 = Axis(fig[1, 1];
    title  = "Bulk energy vs lattice constant — repulsive core",
    xlabel = "Lattice constant (Å)",
    ylabel = "Energy per cell (eV)",
    titlesize = 22,
    xlabelsize = 20,
    ylabelsize = 20,
    xticklabelsize = 16,
    yticklabelsize = 16
)

lines!(ax1, full_lattice_constants, constrained_bulk_energies;
    color=:steelblue, linewidth=3,
    label="ACE constrained"
)

lines!(ax1, full_lattice_constants, unconstrained_bulk_energies_add_zbl;
    color=:orange, linewidth=3, linestyle=:dash,
    label="ACE unconstrained + ZBL"
)

lines!(ax1, full_lattice_constants, zbl_energies;
    color=:red, linewidth=2, linestyle=:dot,
    label="ZBL"
)

hlines!(ax1, [0.0]; color=(:black, 0.4), linewidth=1, linestyle=:dash)

vlines!(ax1, [lattice_constants[end]];
    color=(:black, 0.4), linewidth=1,
    linestyle=:dash
)

axislegend(ax1;
    position = :rt,
    fontsize = 16,
    framevisible = false
)

ylims!(ax1, e_lo, e_hi)


# -----------------------------
# BOTTOM: histogram
# -----------------------------
ax2 = Axis(fig[2, 1];
    xlabel = "Pair distance (Å)",
    ylabel = "Frequency",
    title  = "Pair distance distribution (all configs)",
    titlesize = 22,
    xlabelsize = 20,
    ylabelsize = 20,
    xticklabelsize = 16,
    yticklabelsize = 16
)

all_pairs = vcat(all_dists...)

edges = 0:0.05:maximum(full_lattice_constants)
h = fit(Histogram, all_pairs, edges)

centers = 0.5 .* (h.edges[1][1:end-1] .+ h.edges[1][2:end])

barplot!(ax2, centers, h.weights;
    color = (:gray, 0.7)
)

linkxaxes!(ax1, ax2)

# spacing between panels (IMPORTANT for papers)
rowgap!(fig.layout, 20)

# -----------------------------
# SAVE (IMPORTANT: use vector format for papers)
# -----------------------------
save("$(result.dir)/results/eos_with_pair_hist.pdf", fig)
save("$(result.dir)/results/eos_with_pair_hist.png", fig; px_per_unit = 2)