#   # ACEpotentials.jl Tutorial

#   ## Introduction
# 
#   The `ACEpotentials.jl` documentation
#   (https://acesuit.github.io/ACEpotentials.jl/) contains a number of short,
#   focused tutorials on key topics. This tutorial is longer and has a single
#   narrative. Many Julia commands are introduced by example.

#   ### Installing ACEpotentials
# 
#   `ACEpotentials` version 0.8 and above requires Julia 1.10. For detailed
#   installation instructions, see:
#   https://acesuit.github.io/ACEpotentials.jl/dev/gettingstarted/installation/.
# 
#   Warning: The following installation may take several minutes.

# add and load general packages used in this notebook.

using Pkg


## ACEpotentials installation:  
## If ACEpotentials has not been installed yet, uncomment the following lines
## using Pkg; Pkg.activate(".")
## Add the ACE registry, which stores the ACEpotential package information 
## Pkg.Registry.add(RegistrySpec(url="https://github.com/ACEsuit/ACEregistry"))
## Pkg.add("ACEpotentials")

#   We can check the status of the installed packages.

using Pkg; Pkg.activate(".")
Pkg.status()

#   Import all the packages that we will be using.

using ExtXYZ, Unitful

using ACEpotentials

#   ## Part 1: Basic dataset analysis
# 
#   `ACEpotentials` provides quick access to several example datasets, which can
#   be useful for testing. The following command lists these datasets. (We
#   expect to expand this list signifcantly; please feel free to suggest
#   additions.)

ACEpotentials.list_example_datasets()

#   We begin by loading the tiny silicon dataset.

Si_tiny_dataset, _, _ = ACEpotentials.example_dataset("Si_tiny");

#   These data were taken from a larger set published with:
#   > A. P. Bartók, J. Kermode, N. Bernstein, and G. Csányi, **Machine Learning a General-Purpose Interatomic Potential for Silicon**, Phys. Rev. X 8, 041048 (2018)
# 
#   To illustrate the procedure for loading extended xyz data from a file, we
#   download the larger dataset and load it.

if !isfile("Si_dataset.xyz")
    download("https://www.dropbox.com/scl/fi/z6lvcpx3djp775zenz032/Si-PRX-2018.xyz?rlkey=ja5e9z99c3ta1ugra5ayq5lcv&st=cs6g7vbu&dl=1",
         "Si_dataset.xyz");
end

Si_dataset = ExtXYZ.load("Si_dataset.xyz");

## The last command generates a warning referring to missing pbc in the 
## first structure in the dataset, the isolated atom. We can safely remove this. 
deleteat!(Si_dataset, 1);

#   Next, we assess the dataset sizes.

println("The tiny dataset has ", length(Si_tiny_dataset), " structures.")
println("The large dataset has ", length(Si_dataset), " structures.")

#   Next, we create arrays containing the config_type for each structure in the
#   datasets. Afterwards, we count the configurations of each type.

config_types_tiny = [at[:config_type] for at in Si_tiny_dataset]
config_types = [ at[:config_type] for at in Si_dataset]

function count_configs(config_types)
    config_counts = [sum(config_types .== ct) for ct in unique(config_types)]
    config_dict = Dict([ct=>cc for (ct,cc) in zip(unique(config_types), config_counts)])
end;

println("There are ", length(unique(config_types_tiny)), 
        " unique config_types in the tiny dataset:")
display(count_configs(config_types_tiny))

println("There are ", length(unique(config_types)), 
        " unique config_types in the full dataset:")
display(count_configs(config_types))

#   Two basic distributions which indicate how well the data fills space are the
#   radial and angular distribution functions. We begin with the radial
#   distribution function, plotting using the histogram function in Plots.jl.
#   For the RDF we add some vertical lines to indicate the distances and first,
#   second neighbours and so forth to confirm that the peaks are in the right
#   place.

r_cut = 6.0u"Å"
rnn = 2.35

rdf_tiny = ACEpotentials.get_rdf(Si_tiny_dataset, r_cut; rescale = true)
# plt_rdf_1 = histogram(rdf_tiny[(:Si, :Si)], bins=150, label = "rdf",
#                       title="Si_tiny_dataset", titlefontsize=10,
#                       xlabel = L"r[\AA]", ylabel = "RDF", yticks = [],
#                       xlims=(1.5,6), size=(400,200), left_margin = 2Plots.mm)
# vline!(rnn * [1.0, 1.633, 1.915, 2.3, 2.5], label = "r1, r2, ...", lw=3)

# rdf = ACEpotentials.get_rdf(Si_dataset, r_cut; rescale = true);
# plt_rdf_2 = histogram(rdf[(:Si, :Si)], bins=150, label = "rdf",
#                       title="Si_dataset", titlefontsize=10,
#                       xlabel = L"r[\AA]", ylabel = "RDF", yticks = [],
#                       xlims=(1.5,6), size=(400,200), left_margin = 2Plots.mm)
# vline!(rnn * [1.0, 1.633, 1.915, 2.3, 2.5], label = "r1, r2, ...", lw=3)

# plot(plt_rdf_1, plt_rdf_2, layout=(2,1), size=(400,400))

# #   The larger dataset clearly has a better-converged radial distribution
# #   function. (But also a much larger ratio between high and low distribution
# #   regions.)

# #   For the angular distribution function, we use a cutoff just above the
# #   nearest-neighbour distance so we can clearly see the equilibrium
# #   bond-angles. In this case, the vertical line indicates the equilibrium bond
# #   angle.

# r_cut_adf = 1.25 * rnn * u"Å"
# eq_angle = 1.91 # radians
# adf_tiny = ACEpotentials.get_adf(Si_tiny_dataset, r_cut_adf);
# plt_adf_1 = histogram(adf_tiny, bins=50, label = "adf", yticks = [], c = 3, 
#                     title = "Si_tiny_dataset", titlefontsize = 10,
#                     xlabel = L"\theta", ylabel = "ADF",
#                     xlims = (0, π), size=(400,200), left_margin = 2Plots.mm)
# vline!([ eq_angle,], label = "109.5˚", lw=3)

# adf = ACEpotentials.get_adf(Si_dataset, r_cut_adf);
# plt_adf_2 = histogram(adf, bins=50, label = "adf", yticks = [], c = 3, 
#                     title = "Si_dataset", titlefontsize = 10,
#                     xlabel = L"\theta", ylabel = "ADF",
#                     xlims = (0, π), size=(400,200), left_margin = 2Plots.mm)
# vline!([ eq_angle,], label = "109.5˚", lw=3)

# plot(plt_adf_1, plt_adf_2, layout=(2,1), size=(400,400))

# #   For later use, we define a function that extracts the energies stored in the
#   silicon datasets.

function extract_energies(dataset)
    energies = []
    for atoms in dataset
        for key in keys(atoms)
            if lowercase(String(key)) == "dft_energy"
                push!(energies, atoms[key] / length(atoms))
            end
        end
    end
    return energies
end;
    
Si_dataset_energies = extract_energies(Si_dataset)
;  # the ; is just to suppress the ouput

#   ## Part 2: ACE descriptors
# 
#   An ACE basis specifies a vector of invariant features of atomic environments
#   and can therefore be used as a general descriptor.
# 
#   Some important parameters include:
#   -  elements: list of chemical species, as symbols;
#   -  order: correlation/interaction order (body order - 1);
#   -  totaldegree: maximum total polynomial degree used for the basis;
#   -  rcut : cutoff radius (optional, defaults are provided).

model = ace1_model(elements = [:Si],
                   rcut = 5.5,
                   order = 3,        # body-order - 1
                   totaldegree = 8 );

#   As an example, we compute an averaged structural descriptor for each
#   configuration in the tiny dataset.

descriptors = []
for system in Si_tiny_dataset
    struct_descriptor = sum(site_descriptors(system, model)) / length(system)
    push!(descriptors, struct_descriptor)
end

#   Next, we extract and plot the principal components of the structural
#   descriptors. Note the segregation by configuration type.

descriptors = reduce(hcat, descriptors)  # convert to matrix
# M = fit(PCA, descriptors; maxoutdim=3, pratio=1)
# descriptors_trans = transform(M, descriptors)
# p = scatter(
#      descriptors_trans[1,:], descriptors_trans[2,:], descriptors_trans[3,:],
#      marker=:circle, linewidth=0, group=config_types_tiny, legend=:right)
# plot!(p, xlabel="PC1", ylabel="PC2", zlabel="PC3", camera=(20,10))

#   Finally, we repeat the procedure for the full dataset. Some clustering is
#   apparent, although the results are a bit harder to interpret.

descriptors = []
for system in Si_dataset
    struct_descriptor = sum(site_descriptors(system, model)) / length(system)
    push!(descriptors, struct_descriptor)
end

# descriptors = reduce(hcat, descriptors)  # convert to matrix
# M = fit(PCA, descriptors; maxoutdim=3, pratio=1)
# descriptors_trans = transform(M, descriptors)
# p = scatter(
#      descriptors_trans[1,:], descriptors_trans[2,:], descriptors_trans[3,:],
#      marker=:circle, linewidth=0, group=config_types, legend=:right)
# plot!(p, xlabel="PC1", ylabel="PC2", zlabel="PC3", camera=(10,10))

# #   ## Part 3: Basic model fitting
#
#   We begin by defining an (extremely simple) ACEModel.

model = ace1_model(
              elements = [:Si,],
              order = 3,
              totaldegree = 8,
              rcut = 5.0,
              Eref = Dict(:Si => -158.54496821))

## `ace1_model` specifies a linear model `model`; because it is linear 
## it is implicitly defined by a basis. In `ACEpotentials`, the size of 
## this basis (= number of parameters) can be checked as follows
@show length_basis(model);

#   Next, we fit determine the model parameters using the tiny dataset and ridge
#   regression via the QR solver.

solver = ACEfit.QR(lambda=1e-1)
data_keys = (energy_key = "dft_energy", force_key = "dft_force", virial_key = "dft_virial")
# acefit!(Si_tiny_dataset, model; solver=solver, data_keys...);

if !isfile("Si_dataset.xyz")
    download("https://www.dropbox.com/scl/fi/z6lvcpx3djp775zenz032/Si-PRX-2018.xyz?rlkey=ja5e9z99c3ta1ugra5ayq5lcv&st=cs6g7vbu&dl=1",
         "Si_dataset.xyz");
end

Si_dataset = ExtXYZ.load("Si_dataset.xyz");
deleteat!(Si_dataset, 1);
data_keys = (energy_key = "dft_energy", force_key = "dft_force", virial_key = "dft_virial")
smoothness = 4
prior = nothing
train = ACEpotentials.make_atoms_data(Si_dataset[end-1:end], model; 
                          energy_key = data_keys[1], 
                          force_key = data_keys[2], 
                          virial_key = data_keys[3],
                          weights = Dict("default"=>Dict("E"=>30.0, "F"=>1.0, "V"=>1.0)))
P = ACEpotentials._make_prior(model, smoothness, prior)
#A, Y, W = ACEfit.assemble(train, model)
solver=ACEfit.QR()
acefit!(Si_dataset, model; solver=solver, data_keys = data_keys)