using ACEWorkflow, AtomsBuilder, Unitful, CairoMakie, Printf, DelimitedFiles, Random, ACEpotentials, LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
#  Script
# ─────────────────────────────────────────────────────────────────────────────

Random.seed!(1234)

# ── Load model (adjust path as needed) ──────────────────────────────────────
result = load_model(:Al, 12, 4, 6, 3)
model  = result.model

# # ── POPS ─────────────────────────────────────────────────────────────────────
Ap = Diagonal(result.W) * result.A / result.P
Yw = result.W .* result.Y
pops_corrections = corrections(Ap, Yw, result.P; leverage_percentile=0.0)
hypercube_eigenvectors, hypercube_bounds = hypercube(pops_corrections)
pops_samples, dθ = sample_hypercube(hypercube_eigenvectors, hypercube_bounds, result.lin_params; number_of_committee_members=50)
pops_samples = [vec(pops_samples[:,i]) for i=1:size(pops_samples, 2)]

function _phonon_committee(model, coeffs_committee, result; N_per_seg=30, N_cell=3, file_prefix="")
    return phonon_committee(model, coeffs_committee, result, :Al; N_per_seg, N_cell, file_prefix)
end
pops_samples = readdlm("$(result.dir)/pca_multi_hypercube_samples.csv", ',')
pops_samples = [vec(pops_samples[i,:]) for i=1:size(pops_samples,1)]
x_vals_out, all_freqs, x_ticks_out, labels_out = _phonon_committee(model, pops_samples, result; N_per_seg=30, N_cell=3, file_prefix="kmeans_multi_hypercube_")