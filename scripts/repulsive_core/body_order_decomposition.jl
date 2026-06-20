# body_order_decomposition.jl
#
# Diagnose the source of the attractive core by decomposing the ACE energy
# into pure 2-body, 3-body, and 4-body contributions.
#
# Geometry used at each distance r (r is the reference/shortest side):
#   2-body : dimer                — 2 atoms, separation r
#   3-body : scalene triangle     — sides r, 1.2r, 1.5r  (no two sides equal)
#   4-body : non-square rectangle — width r, height 1.5r  (planar, 4 corners)
#
# The 3-body and 4-body terms are isolated by the exact inclusion-exclusion:
#   E_3body = E_triangle  − Σ_{pairs} E_2body(r_ij)
#   E_4body = E_quad      − Σ_{pairs} E_2body(r_ij) − Σ_{triples} E_3body(triangle)
#
# For a BCC lattice the nearest-neighbour distance is r_NN = a * sqrt(3)/2,
# so a secondary x-axis shows the equivalent BCC lattice constant.

using ACEWorkflow, ACEpotentials, AtomsBuilder, Unitful, CairoMakie
using AtomsCalculators: potential_energy
using LinearAlgebra: norm

element = :W
result  = load_model(element, 20, 4, 5, 3)
model   = result.model

distances = LinRange(0.01, 4.0, 200)    # nearest-neighbour distance (Å)

# ── helper: build an isolated cluster in a large non-periodic box ────────────
function isolated_system(positions_Å, element::Symbol)
    n   = length(positions_Å)
    box = (1.01 * maximum(norm.(positions_Å)) + 5.0)
    AtomsBuilder._flexible_system(
        positions_Å .* u"Å",
        fill(element, n),
        box .* [1. 0. 0.; 0. 1. 0.; 0. 0. 1.] .* u"Å",
        (false, false, false))
end

function E(sys)
    ustrip(potential_energy(sys, model))   # eV
end

# ── 2-body ────────────────────────────────────────────────────────────────────
function e2_body(r)
    E(isolated_system([[0.,0.,0.], [r,0.,0.]], element))
end

# ── 3-body: general triangle placed by law of cosines ───────────────────────
# Atom 1 at origin, atom 2 at [r12, 0, 0], atom 3 computed from side lengths.
function e3_body_general(r12, r13, r23)
    x3 = (r12^2 + r13^2 - r23^2) / (2 * r12)
    y3 = sqrt(max(r13^2 - x3^2, 0.0))
    tri = isolated_system([[0., 0., 0.], [r12, 0., 0.], [x3, y3, 0.]], element)
    E(tri) - e2_body(r12) - e2_body(r13) - e2_body(r23)
end

# Scalene triangle: sides r, 1.2r, 1.5r  (triangle inequality satisfied)
function e3_body(r)
    e3_body_general(r, 1.2*r, 1.5*r)
end

# ── 4-body: non-square rectangle (width r, height 1.5r) ──────────────────────
# Four corners:  p1=(0,0), p2=(r,0), p3=(r,1.5r), p4=(0,1.5r)
# 6 pairs: 4 sides (r×2, 1.5r×2) + 2 diagonals (r*√3.25 each)
# 4 triangles: each is a right triangle with legs r and 1.5r
function e4_body(r)
    h  = 1.5 * r
    dg = sqrt(r^2 + h^2)   # diagonal = r*sqrt(3.25)
    p1 = [0., 0., 0.]
    p2 = [r,  0., 0.]
    p3 = [r,  h,  0.]
    p4 = [0., h,  0.]
    quad = isolated_system([p1, p2, p3, p4], element)

    # Subtract all 6 pairwise 2-body terms
    E2_sum = 2*e2_body(r) + 2*e2_body(h) + 2*e2_body(dg)

    # Subtract all 4 triangles' pure 3-body terms
    # (1,2,3): r12=r,  r13=dg, r23=h
    # (2,3,4): r23=h,  r24=dg, r34=r   — same shape
    # (3,4,1): r34=r,  r31=dg, r41=h   — same shape
    # (4,1,2): r41=h,  r42=dg, r12=r   — same shape
    E3_sum = 4 * e3_body_general(r, dg, h)

    E(quad) - E2_sum - E3_sum
end

# ── evaluate ─────────────────────────────────────────────────────────────────
@info "Computing 2-body curve …"
e2 = [e2_body(r) for r in distances]
@info "Computing 3-body curve …"
e3 = [e3_body(r) for r in distances]
@info "Computing 4-body curve …"
e4 = [e4_body(r) for r in distances]

# ── plot ──────────────────────────────────────────────────────────────────────
# Clip large values for visibility
clip = 30.0
e2c  = clamp.(e2, -clip, clip)
e3c  = clamp.(e3, -clip, clip)
e4c  = clamp.(e4, -clip, clip)

fig = Figure(size=(850, 520))
ax  = Axis(fig[1,1];
           title   = "Body-order decomposition — repulsive core ($(element))",
           xlabel  = "Cluster side length / NN distance (Å)",
           ylabel  = "Pure n-body energy (eV)")

hlines!(ax, [0.0]; color=(:black, 0.3), linewidth=0.8, linestyle=:dash)

lines!(ax, distances, e2c; color=:steelblue,  linewidth=2,   label="2-body (dimer)")
lines!(ax, distances, e3c; color=:darkorange,  linewidth=2,   label="3-body (scalene △, sides r:1.2r:1.5r)")
lines!(ax, distances, e4c; color=:crimson,     linewidth=2,   label="4-body (rectangle, width r × height 1.5r)")

# Mark the BCC nearest-neighbour distance at the constraint boundary (a ≈ 2.18 Å)
constraint_a    = 2.18          # BCC lattice constant at constraint boundary
constraint_r_nn = constraint_a * sqrt(3) / 2
vlines!(ax, [constraint_r_nn]; color=(:black, 0.4), linewidth=1.0,
        linestyle=:dash, label="BCC NN at constraint boundary")

ylims!(ax, -clip * 0.6, clip * 0.6)
axislegend(ax; position=:rt)

# Secondary x-axis: corresponding BCC lattice constant  (r_NN = a * √3/2  ⟹  a = r * 2/√3)
ax2 = Axis(fig[1,1];
           xaxisposition = :top,
           xlabel        = "Equivalent BCC lattice constant (Å)",
           xgridvisible  = false,
           ygridvisible  = false,
           yticksvisible = false,
           yticklabelsvisible = false)
hidedecorations!(ax2; label=false, ticklabels=false, ticks=false, grid=true)
ax2.limits = (distances[1] * 2/sqrt(3), distances[end] * 2/sqrt(3), -clip*0.6, clip*0.6)

save("$(result.dir)/results/body_order_decomposition.png", fig)
@info "Saved → $(result.dir)/results/body_order_decomposition.png"
