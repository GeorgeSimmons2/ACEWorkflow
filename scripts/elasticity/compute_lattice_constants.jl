using ACEpotentials, GeometryOptimization, AtomsBase, Unitful, DelimitedFiles, AtomsBuilder

if !(@isdefined POPS_delta)
    POPS_delta = readdlm("high_entropy_POPS/pointwise_corrections.csv", ',')
end

if !(@isdefined model)
    model, _ = ACEpotentials.load_model("high_entropy_POPS/model.json")
end

lattice_constants = zeros(1+size(POPS_delta,1))
system  = bulk(:Al; cubic=true)
results = minimize_energy!(system, model; variablecell=true)
optsystem = results.system
lattice_constants[1] = ustrip(optsystem.cell.cell_vectors[1][2] * 2)

for i=2:size(POPS_delta, 1)+1
    ACEpotentials.Models.set_linear_parameters!(model, POPS_delta[i-1,:])
    system  = bulk(:Al)
    local results = minimize_energy!(system, model; variablecell=true)
    optsystem = results.system
    local a = optsystem.cell.cell_vectors[1][2] * 2
    lattice_constants[i] = ustrip(a)
    println(i)
end

writedlm("high_entropy_POPS/lattice_constants.csv", lattice_constants, ',')