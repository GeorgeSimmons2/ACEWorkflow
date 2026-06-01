using ACEWorkflow, LinearAlgebra, Random, DelimitedFiles, Clustering, Statistics

Random.seed!(1234)

result = load_model(:Al, 12, 4, 6, 3)
model  = result.model

# ── POPS ─────────────────────────────────────────────────────────────────────
Ap = Diagonal(result.W) * result.A / result.P
Yw = result.W .* result.Y
pops_corrections = corrections(Ap, Yw, result.P; leverage_percentile=0.0)
R = kmeans(full_pops_corrections', 4; maxiter=10_000, display=:iter)

function multi_hypercube(pops_corrections::AbstractMatrix{Float64}, kmeans_clusters::KmeansResult{Matrix{Float64}, Float64, Int64}; percentile_clipping::Float64 = 0.0)
    clusters = [pops_corrections'[:, kmeans_clusters.assignments .== k] for k in 1:length(kmeans_clusters.counts)]
    cluster_bounds  = []
    cluster_eigvecs = []

    for pointwise_corrections in clusters
        eig = eigen(Symmetric(pointwise_corrections * pointwise_corrections'))
        eigvals = eig.values
        eigvecs = eig.vectors

        mask = eigvals .> maximum(eigvals) * 1e-8
        eigvecs = eigvecs[:, mask]
        eigvals = eigvals[mask]

        projections = eigvecs
        projected = pointwise_corrections' * projections

        lower = [quantile(projected[:, j], percentile_clipping / 100) for j in 1:size(projected, 2)]
        upper = [quantile(projected[:, j], 1.0 - percentile_clipping / 100) for j in 1:size(projected, 2)]

        bounds = vcat(lower', upper')
        push!(cluster_bounds, bounds)
        push!(cluster_eigvecs, eigvecs)
    end
    return cluster_eigvecs, cluster_bounds
end

function multi_hypercube_rejection(pops_corrections::AbstractMatrix{Float64}, kmeans_clusters::KmeansResult{Matrix{Float64}, Float64, Int64},
                                   coeffs::Vector{Float64}; percentile_clipping::Float64 = 0.0,
                                   number_of_committee_members::Int = 20)
    clusters = [pops_corrections'[:, kmeans_clusters.assignments .== k] for k in 1:length(kmeans_clusters.counts)]
    samples = []

    for pointwise_corrections in clusters
        eig = eigen(Symmetric(pointwise_corrections * pointwise_corrections'))
        eigvals = eig.values
        eigvecs = eig.vectors

        mask = eigvals .> maximum(eigvals) * 1e-8
        eigvecs = eigvecs[:, mask]

        projected = pointwise_corrections' * eigvecs

        lower_pca = [quantile(projected[:, j], percentile_clipping / 100) for j in 1:size(projected, 2)]
        upper_pca = [quantile(projected[:, j], 1.0 - percentile_clipping / 100) for j in 1:size(projected, 2)]

        lower_orig = vec(minimum(pointwise_corrections, dims=2))
        upper_orig = vec(maximum(pointwise_corrections, dims=2))

        perturbations = zeros(Float64, size(pointwise_corrections, 1), number_of_committee_members)
        count = 0
        while count < number_of_committee_members
            u = rand(Float64, length(lower_pca))
            δ = eigvecs * (lower_pca .+ (upper_pca .- lower_pca) .* u)
            if all(lower_orig .<= δ) && all(δ .<= upper_orig)
                count += 1
                perturbations[:, count] = δ
                if (mod(count, 10) == 0)
                    println("$(count) samples accepted")
                end
            end
        end

        committee_i = coeffs[:, :] .+ perturbations
        append!(samples, [vec(committee_i[:, j]) for j in 1:number_of_committee_members])
    end
    return samples
end

cluster_eigvecs, cluster_bounds = multi_hypercube(pops_corrections, R)

samples = []

for i = 1:length(R.counts)
    eigvecs_i, eigbounds_i = cluster_eigvecs[i], cluster_bounds[i]
    samples_i, _ = sample_hypercube(eigvecs_i, eigbounds_i, result.lin_params; number_of_committee_members=20)
    append!(samples, [vec(samples_i[:,j]) for j=1:size(samples_i, 2)])
end

writedlm("$(result.dir)/pca_multi_hypercube_samples.csv", samples, ',')