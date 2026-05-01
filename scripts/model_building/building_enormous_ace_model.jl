using ACEpotentials, ExtXYZ, DelimitedFiles, Statistics, LinearAlgebra, Random
totdeg = 17
order = 4
training_data = ExtXYZ.load("manual_df_train_Al.xyz")
println("Training data loaded")
model = ace1_model(elements=[:Al], totaldegree=totdeg, order=order, rcut=6.0)
println("Model constructed")
data = ACEpotentials.make_atoms_data(training_data, model;
				     energy_key=:dft_energy, 
				     force_key = :dft_forces, 
				     virial_key= :dft_virials, 
				     weights   = Dict("default"=>Dict("E"=>30., "F"=>1., "V"=>1.)))
P = ACEpotentials._make_prior(model, 4, nothing)

A, Y, W = ACEfit.assemble(data, model)
suffix = "_$(totdeg)_$(order)"
writedlm("./big_ACE/A$(suffix).csv", A, ',')
writedlm("./big_ACE/Y$(suffix).csv", Y, ',')
writedlm("./big_ACE/W$(suffix).csv", W, ',')
writedlm("./big_ACE/P$(suffix).csv", P, ',')


Y = Y[:,1]
W = W[:,1]
Ap= Diagonal(W) * A / P
Y = W .* Y
lin_params = P \ (Ap \ Y)
ACEpotentials.save_model(model, "./big_ACE/model$(suffix).json")
function corrections(X::AbstractMatrix{Float64}, Y::Vector{Float64}, Gamma::AbstractMatrix{Float64}; leverage_percentile::Float64 = 0.5, lambda::Float64 = 1.0 / size(X,1))
    C      = (Gamma' * Gamma .* lambda .+ X' * X)
    A      = C \ X'
    leverage = diag(X * A)
    coeffs = C \ (X' * Y)
    errors = Y .- (X * coeffs)
    leverage_threshold = quantile(leverage, leverage_percentile)
    mask = leverage .>= leverage_threshold
    pointwise_corrections = A[:,mask]'
    pointwise_corrections = pointwise_corrections .* (errors[mask] ./ leverage[mask])
    pointwise_corrections = Gamma \ pointwise_corrections'
    return pointwise_corrections'
end

function hypercube(pointwise_corrections::AbstractMatrix{Float64}; percentile_clipping::Float64 = 0.0)
    eig = eigen(Symmetric(pointwise_corrections' * pointwise_corrections))
    eigvals = eig.values
    eigvecs = eig.vectors

    mask = eigvals .> maximum(eigvals) * 1e-8
    eigvecs = eigvecs[:, mask]
    eigvals = eigvals[mask]

    projections = eigvecs
    projected = pointwise_corrections * projections

    lower = [quantile(projected[:, j], percentile_clipping / 100) for j in 1:size(projected, 2)]
    upper = [quantile(projected[:, j], 1.0 - percentile_clipping / 100) for j in 1:size(projected, 2)]

    bounds = vcat(lower', upper')

    return eigvecs, bounds
end

function sample_hypercube(eigvecs::AbstractMatrix{Float64}, bounds::AbstractMatrix{Float64}, coeffs::Vector{Float64}; number_of_committee_members::Int64 = 500)
    lower, upper = bounds[1, :], bounds[2, :]

    U = rand(Float64, (number_of_committee_members, size(lower, 1)))

    committee = eigvecs * (lower[:, :]' .+ (upper .- lower)[:,:]' .* U)'

    committee = coeffs[:,:] .+ committee
    
    return committee, U
end
pointwise_corrections = corrections(Ap, Y, P; leverage_percentile=0.0, lambda=1/size(Ap,1))
hypercube_eigs, hypercube_bounds = hypercube(pointwise_corrections; percentile_clipping=0.)
committee, dtheta = sample_hypercube(hypercube_eigs, hypercube_bounds, lin_params)
co_ps_vec = [committee[:,i] for i = 1:size(committee,2)]
ACEpotentials.Models.set_committee!(model, co_ps_vec)
writedlm("./big_ACE/pointwise_corrections$(suffix).csv", pointwise_corrections, ',')
test = ExtXYZ.load("high_entropy_pops/manual_df_test_Al.xyz")
test_E = []
test_E_predictions = []
test_co_E_predictions = []
ev_val = counter = 0
co_E_range = []
import Pkg; Pkg.add("AtomsCalculators")
using AtomsCalculators
for (i, at) in enumerate(test)
    try
        energy = at[:dft_energy]

        push!(test_E, energy)
        E, co_E= @committee AtomsCalculators.potential_energy(at, model)
        push!(test_E_predictions, ustrip(E) - energy)
        push!(test_co_E_predictions, ustrip.(co_E) .- ustrip(E))
        push!(co_E_range, abs(maximum(ustrip.(co_E)) - minimum(ustrip.(co_E))))
        if (ustrip(E) > minimum(ustrip.(co_E)) && ustrip(E) < maximum(ustrip.(co_E)))
            counter += 1
        else
            ev_val +=  1
            counter += 1
        end
    catch
    end
end
itr = deepcopy(test_E_predictions)
RMSE_ = sqrt.(sum(abs2.(itr .- mean(itr))) / (length(itr) - 1))
writedlm("./big_ACE/test_errors_$(RMSE_)_$(suffix).csv", test_E_predictions, ',')
writedlm("./big_ACE/pops_max_min_test_errors$(suffix).csv", test_E_predictions, ',')

