using Statistics, DelimitedFiles, Unitful, AtomsBuilder, ACEWorkflow, LinearAlgebra
using AtomsCalculators: potential_energy
include("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/scripts/elasticity/lattice_constant_sensitivity.jl")
"""
    quantile_filter(corrections::Matrix, eta::Float64) -> Matrix

Given an N×d POPS corrections matrix, computes the [eta, 1-eta] quantile
bounds for each of the d parameters (columns) and returns a new matrix
containing only the rows where every parameter lies within its bounds.
"""
function quantile_filter(corrections::AbstractMatrix{Float64}, eta::Float64)
    N, d = size(corrections)

    lower = [quantile(corrections[:, j], eta)       for j in 1:d]
    upper = [quantile(corrections[:, j], 1.0 - eta) for j in 1:d]

    keep = Vector{Int}()
    for i in 1:N
        in_bounds = all(lower[j] <= corrections[i, j] <= upper[j] for j in 1:d)
        in_bounds && push!(keep, i)
    end

    return corrections[keep, :]
end
model_description = "Al_20_5_6A_3"
result = load_model(:Al, 20, 5, 6.0, 3)
lin_params = result.lin_params
model = result.model
P = result.P
A = result.A
Y = result.Y
W = result.W
Y = Y .* W
Ap= Diagonal(W) * A / P
POPS_corrections = corrections(Ap, Y, P; leverage_percentile=0.0)
writedlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/models/$model_description/pops_corrections.csv", POPS_corrections, ',')

POPS_corrections_quantile = quantile_filter(POPS_corrections, 0.05)
norms = []

for i=1:size(POPS_corrections_quantile, 1)
    push!(norms, norm(POPS_corrections_quantile[i,:]))
end

chosen_vec = POPS_corrections_quantile[argmax(norms),:] + lin_params

a_ref = relax_lattice_constant(model, :Al)
ACEpotentials.Models.set_linear_parameters!(model, chosen_vec)
a_sus = relax_lattice_constant(model, :Al)

dB = del_lattice_constant_design(ustrip(a_ref))

a_estimate = ustrip((potential_energy(bulk(:Al, a=a_sus), model) - potential_energy(bulk(:Al, a=a_ref), model)) / (dot(dB, chosen_vec))) + ustrip(a_ref)
println("a_estimate vs a_sus: $(ustrip(a_estimate)) vs $(ustrip(a_sus))")

del_2_lattice_constant_design(a) = ForwardDiff.derivative(
                              x -> del_lattice_constant_design(x),
                              a,
)