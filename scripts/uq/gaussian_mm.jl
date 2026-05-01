using GaussianMixtures
using DelimitedFiles, Statistics

pops_con = readdlm("small_high_entropy_ace_model/constrained_pops.csv", ',')
# pops_con: n_samples × n_params

μ = mean(pops_con, dims=1)        # 1 × n_params
σ = std(pops_con, dims=1) .+ 1e-8 # 1 × n_params

Xn = Float64.((pops_con .- μ) ./ σ)  # n_samples × n_params — correct orientation for GMM

K = 5  # number of mixture components (tune this!)
gmm = GMM(K, Xn; kind=:diag)

# Variance flooring can cause weights to drift off-sum-to-1; renormalise before sampling
gmm.w ./= sum(gmm.w)

# rand(gmm, n) returns n × n_params; de-normalise to original scale
samples = rand(gmm, 100) .* σ .+ μ  # 10 × n_params
