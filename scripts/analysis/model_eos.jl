using CairoMakie
using AtomsBuilder
using ACEpotentials
using Unitful

function plot_eos(param_set, labels; element=:Al, a_min=2.0, a_max=7., n=1000)
    aa = range(a_min, a_max, length=n)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="a [Å]", ylabel="E [eV]")

    for (params, label) in zip(param_set, labels)
        ACEpotentials.Models.set_linear_parameters!(model, params)
        EE = [ustrip(ACEpotentials.potential_energy(bulk(element; a=a*u"Å"), model)) for a in aa]

        lines!(ax, aa, EE, label=label)
    end
    ylims!(ax, -10, 10.)
    axislegend(ax, position=:rt)

    return fig
end


param_set = lin_params_priors
labels = ["P: smoothness=$i" for i =2:6]

fig = plot_eos(param_set, labels; element=:Al)#, a_min=2.5, a_max=4.9, n=200)
save("high_entropy_POPS/eos_prior_sweep.png", fig)