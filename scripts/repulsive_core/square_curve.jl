using ACEWorkflow, ACEpotentials, AtomsBuilder, Unitful, CairoMakie
using AtomsCalculators: potential_energy

result = load_model(:W, 20, 4, 5, 3)
model = result.model

distances = LinRange(0.1, 4.5, 100)

function dimer_energy(element::Symbol, r::Float64, model)
    box = 1.01 * r * 2
    ats = AtomsBuilder._flexible_system(
        [[0.0, 0.0, 0.0], [r, 0.0, 0.0]] .* u"Å",
        [element, element],
        box .* [1. 0. 0.; 0. 1. 0.; 0. 0. 1.] .* u"Å",
        (false, false, false))
    return potential_energy(ats, model)
end

# Right-isosceles triangle with legs d and hypotenuse d√2:
#   atom 1 at (0,0),  atom 2 at (d,0),  atom 3 at (0,d)
# All 4 sub-triplets of the square have exactly this geometry.
function right_isosceles_triangle_energy(element::Symbol, d::Float64, model)
    box = 1.01 * d * sqrt(2) * 1.5
    ats = AtomsBuilder._flexible_system(
        [[0.0, 0.0, 0.0], [d, 0.0, 0.0], [0.0, d, 0.0]] .* u"Å",
        [element, element, element],
        box .* [1. 0. 0.; 0. 1. 0.; 0. 0. 1.] .* u"Å",
        (false, false, false))
    return potential_energy(ats, model)
end

function square_4body_curve(element::Symbol, distances::AbstractVector{Float64}, model)

    energies_4body = []

    for d in distances
        # Square: atoms at (0,0), (d,0), (d,d), (0,d)
        #   4 side-pairs at distance d
        #   2 diagonal-pairs at distance d√2
        #   4 sub-triplets, all right-isosceles triangles with legs d
        box = 1.01 * d * sqrt(2) * 1.5
        ats = AtomsBuilder._flexible_system(
            [[0.0, 0.0, 0.0], [d, 0.0, 0.0],
             [d,   d,   0.0], [0.0, d, 0.0]] .* u"Å",
            [element, element, element, element],
            box .* [1. 0. 0.; 0. 1. 0.; 0. 0. 1.] .* u"Å",
            (false, false, false))
        E_square = potential_energy(ats, model)

        # Inclusion-exclusion: V4 = E_square - Σ_triplets E_ijk + Σ_pairs E_ij
        E_tri  = right_isosceles_triangle_energy(element, d,          model)  # ×4
        E_side = dimer_energy(element,            d,                   model)  # ×4 side pairs
        E_diag = dimer_energy(element,            d * sqrt(2),         model)  # ×2 diagonal pairs

        V4 = E_square - 4*E_tri + 4*E_side + 2*E_diag
        push!(energies_4body, V4)
    end

    return energies_4body
end

square_energies = square_4body_curve(:W, distances, model)

fig = Figure()
ax  = Axis(fig[1,1], title="Square Pure 4-body Interaction", xlabel="Side length (Å)", ylabel="4-body energy (eV)")
lines!(ax, distances, ustrip.(square_energies))
hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
save("$(result.dir)/results/square_4body_curve.png", fig)
