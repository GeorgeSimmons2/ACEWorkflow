using LinearAlgebra, Statistics, DelimitedFiles, Printf
using ACEpotentials, ForwardDiff, Unitful, ACEWorkflow, CairoMakie

# result     = load_model(:Al, 20, 5, 6.0, 3)
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)

# POPS = readdlm(joinpath(result.dir, "pops_corrections.csv"), ',', Float64)


quantile_corrections = pops_corrections

a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)

function lattice_basis(a_val)
    sys = ACEWorkflow.Elasticity.reference_system(:Al; a=a_val)
    ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
end

b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(
                     a_val -> ForwardDiff.derivative(lattice_basis, a_val), a_eq)
b_triple_prime = ForwardDiff.derivative(
                     a_val -> ForwardDiff.derivative(
                         a_val2 -> ForwardDiff.derivative(lattice_basis, a_val2),
                         a_val), a_eq)

K               = dot(lin_params, b_double_prime)   # θ_eq · b″
scalar_b_triple = dot(lin_params, b_triple_prime)   # θ_eq · b‴

δa = []

for i=1:size(quantile_corrections,1)
    δθ_1   = quantile_corrections[i,:]
    δa_1   = -dot(b_prime, δθ_1) / K
    bpp_dot_dθ = dot(b_double_prime, δθ_1)
    δa_2    = -(bpp_dot_dθ * δa_1 + 0.5 * scalar_b_triple * δa_1^2) / K
    δa_ift2 = δa_1 + δa_2
    push!(δa, δa_ift2)
end

fig = Figure()
ax  = Axis(fig[1,1], xlabel="Δa (Å)", ylabel="Counts")
hist!(ax, δa, normalization=:pdf, bins=30)
save("$(result.dir)/results/lattice_constant_distribution.png", fig)