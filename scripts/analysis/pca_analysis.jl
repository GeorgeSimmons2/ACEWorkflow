using MultivariateStats, CairoMakie, Distributions, Statistics, LinearAlgebra

# function fit_gmm_robust(data, n_components)
# 	gmm = GMM(n_components, data)
# 	em!(gmm, data)
#
# 	if !(sum(gmm.w) > 0.0) || any(gmm.w .< 0.0)
# 		println("GMM fit produced invalid weights with :kmeans init; retrying with :split init")
# 		gmm = GMM(n_components, data; method=:split)
# 		em!(gmm, data)
# 	end
#
# 	if !(sum(gmm.w) > 0.0) || any(gmm.w .< 0.0)
# 		error("GMM fit failed: mixture weights remain invalid after retry")
# 	end
#
# 	gmm.w ./= sum(gmm.w)
# 	return gmm
# end

POPS_corrections = constrained_POPS_all
# pops_pca = fit(PCA, POPS_corrections'; maxoutdim=2)  # expects n_features × n_observations

# explained = principalvars(pops_pca)
# explained_ratio = explained ./ tvar(pops_pca)
# cumulative_ratio = cumsum(explained_ratio)

# println("Explained variance by retained PCs:")
# for (idx, (var, ratio, cum_ratio)) in enumerate(zip(explained, explained_ratio, cumulative_ratio))
# 	println("  PC$(idx): variance=$(round(var, sigdigits=6))  ratio=$(round(100 * ratio, digits=2))%  cumulative=$(round(100 * cum_ratio, digits=2))%")
# end

# # Project data onto the 2 principal components
# proj = transform(pops_pca, POPS_corrections')  # shape: 2 × n_observations

# # Fit a 2D Gaussian to the latent points
# μ = vec(mean(proj, dims=2))
# Σ = cov(proj')
# gauss = MvNormal(μ, Σ)

# # Fit a GMM in the same 2D latent space
# n_components = 2
# gmm = fit_gmm_robust(proj, n_components)
# println("GMM weights: ", round.(gmm.w, digits=4))

# # Sample from each fitted latent distribution
# n_samples = 1000
# samples_gauss = rand(gauss, n_samples)  # shape: 2 × n_samples
# samples_gmm = rand(gmm, n_samples)      # shape: 2 × n_samples

# latent_lin_params = transform(pops_pca, lin_params)

# # Visualize
# fig = Figure()
# ax = Axis(fig[1,1], xlabel="PC1", ylabel="PC2", title="POPS Corrections PCA: Original vs Gaussian vs GMM")
# scatter!(ax, proj[1,:], proj[2,:], markersize=6, alpha=0.7, color=:black, label="Original")
# scatter!(ax, samples_gauss[1,:], samples_gauss[2,:], markersize=4, alpha=0.35, color=:red, label="Naive Mv Gaussian")
# scatter!(ax, samples_gmm[1,:], samples_gmm[2,:], markersize=4, alpha=0.35, color=:dodgerblue, label="GMM")
# scatter!(ax, latent_lin_params[1], latent_lin_params[2], markersize=7, alpha=1.0, color=:orange, label="OLS")
# axislegend(ax)
# save("high_entropy_POPS/pca_analysis.png", fig)

# # Check bounds against the original POPS_corrections (features are columns)
# lb = minimum(POPS_corrections, dims=1)'  # lower bound per feature
# ub = maximum(POPS_corrections, dims=1)'  # upper bound per feature

# # Reconstruct samples into the full coefficient space (2D latent)
# samples_full_gauss = reconstruct(pops_pca, samples_gauss)  # shape: n_features × n_samples
# samples_full_gmm = reconstruct(pops_pca, samples_gmm)      # shape: n_features × n_samples

# # A sample is "outside" if any feature exceeds the bounds
# outside_gauss = [any(samples_full_gauss[:, i] .< lb .|| samples_full_gauss[:, i] .> ub) for i in 1:n_samples]
# outside_gmm = [any(samples_full_gmm[:, i] .< lb .|| samples_full_gmm[:, i] .> ub) for i in 1:n_samples]

# println("Naive Gaussian samples outside original bounds: $(sum(outside_gauss)) / $n_samples")
# println("GMM samples outside original bounds: $(sum(outside_gmm)) / $n_samples")

# 1D latent-space analysis (PC1 only)
pops_pca_1d = fit(PCA, POPS_corrections'; maxoutdim=1)
proj_1d = transform(pops_pca_1d, POPS_corrections')  # shape: 1 × n_observations

explained_1d = principalvars(pops_pca_1d)
explained_ratio_1d = explained_1d ./ tvar(pops_pca_1d)
println("\n1D latent analysis:")
println("  PC1 variance=$(round(explained_1d[1], sigdigits=6))  ratio=$(round(100 * explained_ratio_1d[1], digits=2))%")

μ1 = mean(proj_1d[1, :])
σ1 = std(proj_1d[1, :])
gauss_1d = Normal(μ1, σ1)
n_samples = 10
# n_components=1
# gmm_1d = fit_gmm_robust(proj_1d, n_components)
# println("1D GMM weights: ", round.(gmm_1d.w, digits=4))

samples_gauss_1d_vec = rand(gauss_1d, n_samples)
samples_gauss_1d = reshape(samples_gauss_1d_vec, 1, :)
# samples_gmm_1d = rand(gmm_1d, n_samples)

latent_lin_params_1d = transform(pops_pca_1d, lin_params)

fig1d = Figure()
ax1d = Axis(fig1d[1,1], xlabel="PC1", ylabel="Count", title="1D Latent (PC1): Original vs Gaussian")
hist!(ax1d, proj_1d[1, :], bins=40, color=(:black, 0.45), label="Original")
hist!(ax1d, samples_gauss_1d_vec, bins=40, color=(:red, 0.35), label="Naive 1D Gaussian")
# hist!(ax1d, vec(samples_gmm_1d), bins=40, color=(:dodgerblue, 0.35), label="1D GMM")
vlines!(ax1d, [latent_lin_params_1d[1]], color=:orange, linewidth=2, label="OLS")
axislegend(ax1d)
save("high_entropy_POPS/pca_analysis_1d.png", fig1d)

# Reconstruct 1D samples into full coefficient space and compare bounds
samples_full_gauss_1d = reconstruct(pops_pca_1d, samples_gauss_1d)
# samples_full_gmm_1d = reconstruct(pops_pca_1d, samples_gmm_1d)

lb = minimum(POPS_corrections, dims=1)'  # lower bound per feature
ub = maximum(POPS_corrections, dims=1)'  # upper bound per feature

outside_gauss_1d = [any(samples_full_gauss_1d[:, i] .< lb .|| samples_full_gauss_1d[:, i] .> ub) for i in 1:n_samples]
# outside_gmm_1d = [any(samples_full_gmm_1d[:, i] .< lb .|| samples_full_gmm_1d[:, i] .> ub) for i in 1:n_samples]

println("Naive 1D Gaussian samples outside original bounds: $(sum(outside_gauss_1d)) / $n_samples")
# println("1D GMM samples outside original bounds: $(sum(outside_gmm_1d)) / $n_samples")
