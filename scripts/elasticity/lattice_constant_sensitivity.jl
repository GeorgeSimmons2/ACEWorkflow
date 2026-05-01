using ForwardDiff
using Unitful
using AtomsBuilder
using ACEpotentials

function lattice_constant_design(a; model=model, element=:Al)
    sys = bulk(element; a=a*u"Å", T=typeof(a))
    return ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(sys, model))
end

del_lattice_constant_design(a; model=model, element=:Al) = ForwardDiff.derivative(
    x -> lattice_constant_design(x; model=model, element=element),
    a,
)

# a_pops = zeros(size(POPS_corrections))

# for i=1:length(a_pops)
#     a_pops[i] = dot(dB, POPS_corrections[i,:])
# end

# fig = Figure()
# ax  = Axis(fig[1,1], xlabel="δE/δa", ylabel="Counts")
# hist!(ax, a_pops, normalization=:pdf, bins=100)
# save("high_entropy_POPS/smoothness_5_delta_a.png", fig) 