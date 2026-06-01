using ACEWorkflow, Random, ExtXYZ
import AtomsCalculators: forces, potential_energy, @committee
Random.seed!(1234)
# ── Load model (adjust path as needed) ──────────────────────────────────────
result = load_model(:Al, 20, 4, 6, 3)
model  = result.model

# ── POPS ─────────────────────────────────────────────────────────────────────
Ap = Diagonal(result.W) * result.A / result.P
Yw = result.W .* result.Y
pops_corrections = corrections(Ap, Yw, result.P; leverage_percentile=0.0)
hypercube_eigenvectors, hypercube_bounds = hypercube(pops_corrections)
pops_samples, dθ = sample_hypercube(hypercube_eigenvectors, hypercube_bounds, result.lin_params; number_of_committee_members=50)
pops_samples = [vec(pops_samples[:,i]) for i=1:size(pops_samples, 2)]
ACEpotentials.Models.set_committee!(model, pops_samples)
corrections_maxima = maximum(pops_corrections, dims=1)' .+ result.lin_params
corrections_minima = minimum(pops_corrections, dims=1)' .+ result.lin_params

lb = vec(corrections_minima)
ub = vec(corrections_maxima)
outside_naive_ind = [(i, findall((s .< lb) .| (s .> ub))) for (i, s) in enumerate(pops_samples) if !all(lb .< s .< ub)]
testing_configs = ExtXYZ.load("../ace_archive/high_entropy_pops/manual_df_test_Al.xyz")

predicted_test_energies = []
test_energies = []
predicted_test_forces = []
test_forces = []
predicted_test_energies_uncertainties = ([], [])
predicted_test_forces_uncertainties = ([], [])

for config in testing_configs
    E, co_E = @committee potential_energy(config, model)
    append!(predicted_test_energies, ustrip(E))
    append!(predicted_test_energies_uncertainties[1], minimum(ustrip.(co_E)))
    append!(predicted_test_energies_uncertainties[2], maximum(ustrip.(co_E)))
    append!(test_energies, ustrip(config[:dft_energy]))
    
    # if (haskey(config[1], :dft_forces))
    #     config_forces = []
    #     for at in config
    #         append!(config_forces, at[:dft_forces])
    #     end
    #     F, co_F = @committee forces(config, model)
    #     append!(predicted_test_forces, ustrip.(F))

    #     for i in eachindex(co_F[1])
    #         fi = reduce(hcat, ustrip(co_F[k][i]) for k in eachindex(co_F))  # 3×50 matrix
    #         append!(predicted_test_forces_uncertainties[1], minimum(fi; dims=2))
    #         append!(predicted_test_forces_uncertainties[2], maximum(fi; dims=2))
    #     end
    # end
end
