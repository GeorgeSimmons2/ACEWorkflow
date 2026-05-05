import Pkg
Pkg.activate("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow")
using DelimitedFiles
using ACEWorkflow
using LinearAlgebra
result = load_model(:Al, 20, 4, 6, 4) 

A = result.A
Y = result.Y
W = result.W
P = result.P
lin_params = result.lin_params

Ap = Diagonal(W) * A / P
Yw = Y .* W

POPS_corrections = corrections(Ap, Y, P)
writedlm("$(result.dir)/pops_corrections.csv", POPS_corrections, ',')

eigvecs, bounds = hypercube(POPS_corrections)

samples, _ = sample_hypercube(eigvecs, bounds, lin_params; number_of_committee_members=20)

writedlm("$(result.dir)/pops_samples.csv", samples, ',')
