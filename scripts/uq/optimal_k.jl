using Clustering
using Statistics
using ACEWorkflow, LinearAlgebra, Random, DelimitedFiles, Clustering, Statistics
using GaussianMixtures
Random.seed!(1234)

result = load_model(:Al, 12, 4, 6, 3)
model  = result.model

k = 223 # maximum number of kmeans centres

# ── POPS ─────────────────────────────────────────────────────────────────────
Ap = Diagonal(result.W) * result.A / result.P
Yw = result.W .* result.Y
pops_corrections = corrections(Ap, Yw, result.P; leverage_percentile=0.0)
X = (pops_corrections * result.P)'

function total_hypercube_volume(X, assignments, K)
    d = size(X, 2)
    vol = 0.0

    for k in 1:K
        Xk = X[:, assignments .== k]   # correct slicing

        mins = minimum(Xk, dims=1)
        maxs = maximum(Xk, dims=1)

        vol += prod(maxs .- mins)
    end

    return vol
end
Ks = 2:80

wcss = Float64[]

for K in Ks
    R = kmeans(X, K; maxiter=10_000)
    push!(wcss, sum(R.costs))
end

fig = Figure(resolution = (800, 500))
ax = Axis(fig[1, 1],
    xlabel = "Number of clusters (K)",
    ylabel = "Within-cluster sum of squares (WCSS)",
    title = "K-means distortion curve (POPS → ACE space)"
)

lines!(ax, Ks, wcss, linewidth = 3)
scatter!(ax, Ks, wcss, markersize = 8)
save("$(result.dir)/results/k_means_costs.png", fig)