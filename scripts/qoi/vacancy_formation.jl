using GeometryOptimization
import ACEpotentials.Models: potential_energy_basis 
sys_bulk = bulk(:Al, cubic=true)
results  = minimize_energy!(sys_bulk, model; variablecell=true)
sys_bulk_opt = results.system
sys_bulk_opt_copy = deepcopy(sys_bulk_opt)

sys_vac  = deleteat!(sys_bulk_opt_copy * (4,4,4), 1)

results  = minimize_energy!(sys_vac, model; variablecell=false)
sys_vac_opt  = results.system

E_def = potential_energy(sys_vac_opt, model)  - (length(sys_vac_opt) * potential_energy(sys_bulk_opt, model) / length(sys_bulk_opt))
E_def_design = potential_energy_basis(sys_vac_opt, model) .- (length(sys_vac_opt) .* potential_energy_basis(sys_bulk_opt, model) ./ length(sys_bulk_opt))

vac_Es = []
pca_samples = (samples_full .+ lin_params)'
for i=1:size(POPS_corrections, 1)
    pops_i = POPS_corrections[i,:] .+ lin_params
    vac_E  = dot(E_def_design, pops_i)
    push!(vac_Es, vac_E)
end

pca_vac_Es = []
for i=1:size(pca_samples, 1)
    pops_i = pca_samples[i,:]
    vac_E  = dot(E_def_design, pops_i)
    push!(pca_vac_Es, vac_E)
end


using CairoMakie

fig = Figure()
ax  = Axis(fig[1, 1];
           xlabel = "Vacancy formation energy (eV)",
           ylabel = "Density",
           title  = "POPS vacancy formation energy distribution")
hist!(ax, ustrip.(vac_Es); bins=50, normalization=:pdf, color=(:steelblue, 0.4), strokecolor=:black, strokewidth=0.5, label="POPS")
hist!(ax, ustrip.(pca_vac_Es); bins=50, normalization=:pdf, color=(:orange, 0.4), strokecolor=:black, strokewidth=0.5, label="PCA samples")
vlines!(ax, [ustrip(E_def)]; color=:red, linewidth=2, label="Mean model")
axislegend(ax)
save("small_high_entropy_ace_model/vacancy_formation_hist.png", fig)

