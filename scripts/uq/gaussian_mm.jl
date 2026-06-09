using ACEWorkflow, LinearAlgebra, Random, DelimitedFiles, Clustering, Statistics
using Distributed, JLD

addprocs(Threads.nthreads())
@everywhere using GaussianMixtures

Random.seed!(1234)

n_gaussians = 200

result = load_model(:Al, 20, 4, 6, 3)
model  = result.model
# Ap = Diagonal(result.W) * result.A / result.P
# Yw = result.W .* result.Y
# pops_corrections = corrections(Ap, Yw, result.P; leverage_percentile=0.0)
pops_corrections = readdlm("$(result.dir)/pops_corrections.csv", ',')
pops_corrections = Matrix(pops_corrections) * result.P

g = GMM(n_gaussians, pops_corrections, kind=:full, nIter=100)
p = GMMprior(g.d, 0.1, 1.0)
v = VGMM(g, p)
em!(v, pops_corrections)
g_fitted = GMM(v)

save("my_gmm.jld", "g_fitted", g_fitted)

gmm_pops = rand(g_fitted, 50)
gmm_pops = [(result.P \ vec(gmm_pops[i,:])) .+ result.lin_params for i=1:size(gmm_pops, 1)]
writedlm("$(result.dir)/$(n_gaussians)_gmm_pops_samples.csv", gmm_pops, ',')
