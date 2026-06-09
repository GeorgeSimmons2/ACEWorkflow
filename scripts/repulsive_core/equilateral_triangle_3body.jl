using ACEWorkflow, ACEpotentials, AtomsBuilder, Unitful, CairoMakie
using AtomsCalculators: potential_energy

result = load_model(:W, 20, 4, 5, 3)
model = result.model

distances = LinRange(0.1, 4.5, 200)

function dimer_energy(element::Symbol, r::Float64, model)
    box = 1.01 * r * 2
    ats = AtomsBuilder._flexible_system(
        [[0.0, 0.0, 0.0], [r, 0.0, 0.0]] .* u"Å",
        [element, element],
        box .* [1. 0. 0.; 0. 1. 0.; 0. 0. 1.] .* u"Å",
        (false, false, false))
    return potential_energy(ats, model)
end

# Equilateral triangle with side length d:
#   atom 1 at (0, 0)
#   atom 2 at (d, 0)
#   atom 3 at (d/2, d*√3/2)
# All three pairs have identical distance d, so the 3-body term is:
#   V3 = E_triangle(d) - 3 * E_dimer(d)
#
# Unlike the linear trimer, the 60° angles are preserved at all d,
# so the angular basis functions remain fully active — any radial
# extrapolation failure at small d appears directly in V3.
function equilateral_triangle_3body_curve(element::Symbol,
                                          distances::AbstractVector{Float64},
                                          model)
    energies_3body = []

    for d in distances
        h = d * sqrt(3) / 2
        box = 1.01 * d * 1.5
        ats = AtomsBuilder._flexible_system(
            [[0.0, 0.0, 0.0], [d, 0.0, 0.0], [d/2, h, 0.0]] .* u"Å",
            [element, element, element],
            box .* [1. 0. 0.; 0. 1. 0.; 0. 0. 1.] .* u"Å",
            (false, false, false))
        E_tri = potential_energy(ats, model)

        # All three pairs are equal (distance d)
        E_pair = dimer_energy(element, d, model)

        push!(energies_3body, E_tri - 3 * E_pair)
    end

    return energies_3body
end

energies_3body = equilateral_triangle_3body_curve(:W, distances, model)
energies_3body_eV = ustrip.(energies_3body)

fig = Figure(size=(800, 500))
ax  = Axis(fig[1,1],
           title   = "Equilateral Triangle — Pure 3-body Interaction",
           xlabel  = "Side length (Å)",
           ylabel  = "3-body energy (eV)")
lines!(ax, distances, energies_3body_eV)
hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)

# Mark the ACE inner cutoff region as a shaded band to show where extrapolation begins
# (typical inner cutoff ~0.5–1.0 Å, adjust if needed)
vspan!(ax, 0.0, 1.0; color=(:red, 0.08))
text!(ax, 0.5, maximum(energies_3body_eV) * 0.9;
      text="extrapolation\nregion", align=(:center, :center), fontsize=11, color=:red)

save("$(result.dir)/results/equilateral_triangle_3body_curve.png", fig)
