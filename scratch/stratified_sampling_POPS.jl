using ACEpotentials, ExtXYZ, DelimitedFiles, POPSRegression
suffix = "stratifiied_sampling"
Si_model = ace1_model(elements=[:Si], totaldegree=16, order=4, rcut=6.0)
A = readdlm("design_matrix_$(suffix).csv", ',')
Y = readdlm("Y_$suffix.csv", ',')
P = readdlm("P_$suffix.csv", ',')
W = readdlm("W_$suffix.csv", ',')
Y = Y[:,1]
W = W[:,1]
Ap= Diagonal(W) * A / P
Y = W .* Y
lin_params = P \ (Ap \ Y)
ACEpotentials.Models.set_linear_parameters!(Si_model, lin_params)
pointwise_corrections = corrections(Ap, Y, P; leverage_percentile=0.5, lambda=1/size(Ap,1))
hypercube_eigs, hypercube_bounds = hypercube(pointwise_corrections)
committee, dΘ = sample_hypercube(hypercube_eigs, hypercube_bounds, lin_params)
co_ps_vec = [committee[:,i] for i = 1:size(committee,2)]
ACEpotentials.Models.set_committee!(Si_model, co_ps_vec)