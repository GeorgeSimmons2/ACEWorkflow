using ACEpotentials, ExtXYZ, DelimitedFiles#, POPSRegression

training_configs = ExtXYZ.load("manual_df_train_Al.xyz")

suffix = "_14_4"
if !isfile("./A$(suffix).csv") || !isfile("./Y$(suffix).csv") || !isfile("./P$(suffix).csv") || !isfile("./W$(suffix).csv")
    
    model = ace1_model(elements=[:Al], totaldegree=14, order=4, rcut=6.0)

    data = ACEpotentials.make_atoms_data(training_configs, model; 
                            energy_key = :dft_energy, 
                            force_key  = :dft_forces, 
                            virial_key = :dft_virials, 
                            weights    = Dict("default"=>Dict("E"=>30.,"F"=>1.,"V"=>1.)))

    P = ACEpotentials._make_prior(model, 4, nothing)

    A, Y, W = ACEfit.assemble(data, model)

    writedlm("./A$(suffix).csv", A, ',')
    writedlm("./Y$(suffix).csv", Y, ',')
    writedlm("./P$(suffix).csv", P, ',')
    writedlm("./W$(suffix).csv", W, ',')
    ACEpotentials.save_model(model, "model$(suffix).json")
else
    model = ACEpotentials.load_model("high_entropy_POPS/model.json")
    A = readdlm("high_entropy_POPS/A.csv", ',')
    Y = readdlm("high_entropy_POPS/Y.csv", ',')
    P = readdlm("high_entropy_POPS/P.csv", ',')
    W = readdlm("high_entropy_POPS/W.csv", ',')
    Y = Y[:,1]
    W = W[:,1]
    Ap= Diagonal(W) * A / P
    Y = W .* Y
    lin_params = P \ (Ap \ Y)
    ACEpotentials.Models.set_linear_parameters!(model, lin_params)
    pointwise_corrections = corrections(Ap, Y, P)
    hypercube_eigs, hypercube_bounds = hypercube(pointwise_corrections)
    committee, dΘ = sample_hypercube(hypercube_eigs, hypercube_bounds, lin_params)
    co_ps_vec = [committee[i,:] for i = 1:size(committee,1)]
    ACEpotentials.Models.set_committee!(model, co_ps_vec)
end
model,_ = ACEpotentials.load_model("high_entropy_POPS/../model_14_4.json")
A = readdlm("high_entropy_POPS/../A_14_4.csv", ',')
Y = readdlm("high_entropy_POPS/../Y_14_4.csv", ',')
P = readdlm("high_entropy_POPS/../P_14_4.csv", ',')
W = readdlm("high_entropy_POPS/../W_14_4.csv", ',')
Y = Y[:,1]
W = W[:,1]
Ap= Diagonal(W) * A / P
Y = W .* Y
lin_params = P \ (Ap \ Y)
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
pointwise_corrections = POPSRegression.corrections(Ap, Y, P)
hypercube_eigs, hypercube_bounds = POPSRegression.hypercube(pointwise_corrections)
committee, dΘ = POPSRegression.sample_hypercube(hypercube_eigs, hypercube_bounds, lin_params)
co_ps_vec = [committee[i,:] for i = 1:size(committee,1)]
ACEpotentials.Models.set_committee!(model, co_ps_vec)