using ACEWorkflow, DelimitedFiles, LinearAlgebra, Distributions, Random

result = load_model(:Al, 20, 4, 6, 3)
model  = result.model

forest_mat = readdlm("$(result.dir)/pops_corrections.csv", ',')
lin_params = result.lin_params

empirical_cov  = forest_mat' * forest_mat ./ size(forest_mat, 2) +1e-20I
dist = MvNormal(lin_params, empirical_cov)

