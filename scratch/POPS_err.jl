using Pkg; Pkg.add("CairoMakie")
using DelimitedFiles, CairoMakie, ExtXYZ, Unitful
using Statistics, LinearAlgebra, ACEpotentials
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

function sample_hypercube(eigvecs::AbstractMatrix{Float64}, bounds::AbstractMatrix{Float64}, coeffs::Vector{Float64}; number_of_committee_members::Int64 = 50)
    lower, upper = bounds[1, :], bounds[2, :]

    U = rand(Float64, (number_of_committee_members, size(lower, 1)))

    committee = eigvecs * (lower[:, :]' .+ (upper .- lower)[:,:]' .* U)'
    δθ        = committee * committee' ./ size(committee, 2)

    committee = coeffs[:,:] .+ committee
    
    return committee, δθ  
end

suffix = "_17_4"
A = readdlm("./big_ACE/A$(suffix).csv", ',')
Y = readdlm("./big_ACE/Y$(suffix).csv", ',')
W = readdlm("./big_ACE/W$(suffix).csv", ',')
P = readdlm("./big_ACE/P$(suffix).csv", ',')

Y = Y[:,1]
W = W[:,1]
Ap= Diagonal(W) * A / P
Y = W .* Y
lin_params = P \ (Ap \ Y)
model, _ = ACEpotentials.load_model("big_ACE/model$(suffix).json")
pointwise_corrections = corrections(Ap, Y, P; leverage_percentile=0.)
hypercube_eigs, hypercube_bounds = hypercube(pointwise_corrections; percentile_clipping=0.)
committee, dtheta = sample_hypercube(hypercube_eigs, hypercube_bounds, lin_params; number_of_committee_members = 2500)
co_ps_vec = [committee[:,i] for i = 1:size(committee,2)]
ACEpotentials.Models.set_committee!(model, co_ps_vec)
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
writedlm("big_ACE/co_E_range$(suffix).csv", co_E_range, ',')
fig = Figure(resolution = (800, 800))
ax  = Axis(fig[1, 1],
    xlabel = "Test errors / eV",
    ylabel = "Uncertainty estimates / eV"
)

xdata = abs.(test_E_predictions)
ydata = co_E_range ./ 2

scatter!(ax, xdata, ydata)

max_val = maximum(maximum.((xdata, ydata)))

# identity line
lines!(
    ax,
    [0, max_val], [0, max_val],
    linestyle = :dash,
    color = :black,
    label = "Identity"
)

# legend entry for EV (dummy line)
lines!(
    ax,
    [0, 0], [0, 0],   # degenerate line
    color = (:white, 0.0),   # fully transparent for Cairo
    label = "EV = $(ev_val)%"
)

padding = 0.05 * max_val
xlims!(ax, 0, max_val + padding)
ylims!(ax, 0, max_val + padding)

axislegend(ax, position = :rb)

save("./big_ACE/test_error_corner_plot$(suffix).png", fig)
fig
