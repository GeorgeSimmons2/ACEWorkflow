using ACEpotentials, EmpiricalPotentials, ACEWorkflow, OSQP, SparseArrays
using ACEpotentials.Models: potential_energy_basis

element = :W
totdeg  = 20
prior_param = 4
rcut = 5
v = 3
result = load_model(element, totdeg, prior_param, rcut, v)
model = result.model
W_zbl = ZBL(5.0*u"Å") # Tungsten ZBL potential
lattice_constants = LinRange(0.2, 2.18, 50)

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


ace_positive_core_constrained_parameters = constrained_ridge_regression(Ap, Yw, Gamma, constraint_matrix, bounds)
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

e_lo, e_hi = -14.0, 30.0   # eV — adjust to taste

fig = Figure(size=(800, 500))
ax  = Axis(fig[1,1];
           title   = "Bulk energy vs lattice constant — repulsive core",
           xlabel  = "Lattice constant (Å)",
           ylabel  = "Energy per cell (eV)")

lines!(ax, full_lattice_constants, constrained_bulk_energies_add_zbl;
       color=:steelblue, linewidth=2, label="ACE with constrained core (E => 0) + ZBL")
lines!(ax, full_lattice_constants, unconstrained_bulk_energies_add_zbl;
       color=:orange, linewidth=2, linestyle=:dash, label="ACE with unconstrained core + ZBL")
lines!(ax, full_lattice_constants, zbl_energies;
       color=:red, linewidth=1.5, linestyle=:dot, label="ZBL")

hlines!(ax, [0.0]; color=(:black, 0.3), linewidth=0.8, linestyle=:dash)
vlines!(ax, [lattice_constants[end]]; color=(:black, 0.3), linewidth=0.8,
        linestyle=:dash, label="constraint boundary")

ylims!(ax, e_lo, e_hi)
axislegend(ax; position=:rt)
save("$(result.dir)/results/constrain_ace_correction_core_repulsion.png", fig)