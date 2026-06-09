using ACEWorkflow, AtomsBuilder, Unitful, CairoMakie, Printf, DelimitedFiles, Random, ACEpotentials, LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
#  Script
# ─────────────────────────────────────────────────────────────────────────────

Random.seed!(1234)

# ── Load model (adjust path as needed) ──────────────────────────────────────
result = load_model(:W, 20, 4, 6, 3)
model  = result.model

N_largest_leverage=39 

# # ── POPS ─────────────────────────────────────────────────────────────────────
X = Diagonal(result.W) * result.A / result.P
Yw = result.W .* result.Y
Gamma = result.P
lambda = 1.0 / size(X,1)
C      = (Gamma' * Gamma .* lambda .+ X' * X)
A      = C \ X'
leverage = diag(X * A)
pops_corrections = corrections(Ap, Yw, result.P; leverage_percentile=0.0)
hypercube_eigenvectors, hypercube_bounds = hypercube(pops_corrections)
pops_samples, dθ = sample_hypercube(hypercube_eigenvectors, hypercube_bounds, result.lin_params; number_of_committee_members=50)
pops_samples = [vec(pops_samples[:,i]) for i=1:size(pops_samples, 2)]

function _phonon_committee(model, coeffs_committee, result; N_per_seg=30, N_cell=3, file_prefix="")
    return phonon_committee(model, coeffs_committee, result, :Al; N_per_seg, N_cell, file_prefix)
end

normalised_leverage = leverage ./ sum(leverage)
cdf = cumsum(normalised_leverage)

# Draw M uniform samples
M = 50  # Number of points you want to subsample
u_samples = rand(M)

# Find indices for all samples simultaneously
sampled_indices = searchsortedfirst.(Ref(cdf), u_samples)
pops_samples = [vec(pops_corrections[i,:]) .+ result.lin_params for i in sampled_indices]


# pops_corrections = readdlm("$(result.dir)/pops_corrections.csv", ',')
pops_samples = [vec(pops_corrections[i,:]) .+ result.lin_params for i=1:size(pops_corrections,1)]
# N_largest_leverage_inds = partialsortperm(leverage, length(Yw)-N_largest_leverage:length(Yw))
# pops_samples[N_largest_leverage_inds]
# pops_samples = shuffle(pops_samples)[1:40]
x_vals_out, all_freqs, x_ticks_out, labels_out = _phonon_committee(model, pops_samples, result; N_per_seg=30, N_cell=3, file_prefix="leverage_weighted_delta_forest_sampling_")