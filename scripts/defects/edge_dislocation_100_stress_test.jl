using ACEWorkflow, ACEpotentials, AtomsBuilder, Unitful, CairoMakie, ExtXYZ, AtomsBase
using AtomsCalculators: potential_energy
using GeometryOptimization
using LinearAlgebra, StaticArrays, Statistics, Printf

# ─────────────────────────────────────────────────────────────────────────────
#  <100> edge dislocation quadrupole — shear stress-test
#
#  Builds a periodic quadrupole of a[100] edge dislocations in BCC W (net
#  Burgers vector zero, fully periodic cell), relaxes the core with the ACE
#  potential, then ramps a simple shear strain (glide-driving, Schmid
#  geometry) with quasi-static energy minimisation at each step, looking for
#  the potential "collapsing in on itself" (unphysical short-range attractive
#  runaway) as the core is driven under increasing load.
# ─────────────────────────────────────────────────────────────────────────────

element = :W
result  = load_model(element, 20, 4, 5, 3)
model   = result.model

outdir = joinpath(result.dir, "results", "edge_dislocation_100_quadrupole")
mkpath(outdir)

# ── Equilibrium lattice constant & elastic constants ─────────────────────────

a_eq = ACEWorkflow.relax_lattice_constant(model, element)
elastic = ACEWorkflow.strain_hessian_GPa(model, element; a=a_eq)
C11, C12, C44 = elastic.C[1,1], elastic.C[1,2], elastic.C[4,4]
ν = C12 / (C11 + C12)   # isotropic approximation, used only to seed the elastic guess

@info "a_eq = $(round(a_eq, digits=4)) Å   C11=$(round(C11,digits=1)) C12=$(round(C12,digits=1)) C44=$(round(C44,digits=1)) GPa   ν=$(round(ν,digits=3))"

b    = a_eq                    # <100> Burgers vector magnitude (full BCC lattice translation)
d_nn = a_eq * sqrt(3) / 2       # BCC nearest-neighbour distance

# ── Supercell ─────────────────────────────────────────────────────────────────
# x=[100] (Burgers vector / glide direction), y=[010] (glide-plane normal),
# z=[001] (dislocation line, periodic). Coincides with the BCC conventional
# cubic cell axes, so bulk(cubic=true) is directly usable.

nx, ny, nz = 20, 20, 1   # set to (10,10,1) for a quick smoke test

sys_perfect = bulk(element, a=a_eq*u"Å", cubic=true) * (nx, ny, nz)
n_atoms = length(sys_perfect)
Lx, Ly, Lz = nx*a_eq, ny*a_eq, nz*a_eq
@info "Supercell: $n_atoms atoms, box = $(round(Lx,digits=2)) x $(round(Ly,digits=2)) x $(round(Lz,digits=2)) Å"

E_bulk_per_atom = ustrip(potential_energy(sys_perfect, model)) / n_atoms

# ── Isotropic edge-dislocation displacement field (Hirth & Lothe form) ───────

function edge_displacement(dx, dy, b, ν)
    r2 = dx^2 + dy^2
    ux = (b / (2π)) * (atan(dy, dx) + (dx*dy) / (2*(1-ν)*r2))
    uy = -(b / (2π)) * ((1-2ν)/(4*(1-ν)) * log(r2) + (dx^2 - dy^2) / (4*(1-ν)*r2))
    return ux, uy
end

min_image(d, L) = d - L * round(d / L)

# Checkerboard quadrupole: net Burgers vector zero in both x and y, so the
# cell stays exactly periodic with no atoms added or removed.
cores = [
    (Lx/4,  Ly/4,   +1.0),
    (3Lx/4, Ly/4,   -1.0),
    (Lx/4,  3Ly/4,  -1.0),
    (3Lx/4, 3Ly/4,  +1.0),
]

function quadrupole_displacement(x, y, cores, Lx, Ly, b, ν)
    ux_tot = 0.0
    uy_tot = 0.0
    for (xc, yc, s) in cores
        dx = min_image(x - xc, Lx)
        dy = min_image(y - yc, Ly)
        (dx == 0.0 && dy == 0.0) && continue
        ux, uy = edge_displacement(dx, dy, s*b, ν)
        ux_tot += ux
        uy_tot += uy
    end
    return ux_tot, uy_tot
end

function apply_displacement_field(sys, disp_fn)
    atoms_new = Vector{AtomsBase.Atom}(undef, length(sys))
    for i in 1:length(sys)
        at = sys[i]
        r  = ustrip.(at.position)
        ux, uy = disp_fn(r[1], r[2])
        rnew = (r[1] + ux, r[2] + uy, r[3])
        atoms_new[i] = AtomsBase.Atom(at.species, collect(rnew) .* u"Å", missing)
    end
    return periodic_system(atoms_new, sys.cell.cell_vectors)
end

# ── Build & relax the zero-strain dislocation quadrupole ────────────────────

sys_disloc0 = apply_displacement_field(sys_perfect,
                                        (x,y) -> quadrupole_displacement(x, y, cores, Lx, Ly, b, ν))

@info "Relaxing zero-strain dislocation quadrupole..."
res0 = minimize_energy!(sys_disloc0, model; variablecell=false, maxiters=2000)
res0.converged || @warn "Zero-strain relaxation did not converge!"
sys_disloc_relaxed = res0.system

E0_per_atom = ustrip(u"eV", res0.energy) / n_atoms
E_excess    = ustrip(u"eV", res0.energy) - n_atoms * E_bulk_per_atom
@info "Zero-strain quadrupole: E/atom = $(round(E0_per_atom,digits=4)) eV, excess over bulk = $(round(E_excess,digits=2)) eV (4 cores, line length $(round(Lz,digits=2)) Å)"

# ── General (triclinic-safe) pairwise diagnostics ────────────────────────────
# Same fractional-coordinate minimum image used in ZBL_core_ACE_correction.jl,
# needed here because the shear ramp below makes the cell non-orthogonal.

nn_cutoff      = 0.75 * d_nn
cluster_cutoff = 0.5  * d_nn
collapse_cutoff = 0.3 * d_nn

function pair_diagnostics(sys, cluster_cutoff, nn_cutoff)
    n = length(sys)
    L    = ustrip.(hcat(sys.cell.cell_vectors...))
    Linv = inv(L)
    pos  = [ustrip.(sys[i].position) for i in 1:n]

    min_d = Inf
    coord = zeros(Int, n)
    adj   = [Int[] for _ in 1:n]

    for i in 1:n-1, j in i+1:n
        dr   = pos[i] .- pos[j]
        frac = Linv * dr
        frac = frac .- round.(frac)
        dr   = L * frac
        r    = norm(dr)

        r < min_d && (min_d = r)
        if r < nn_cutoff
            coord[i] += 1; coord[j] += 1
        end
        if r < cluster_cutoff
            push!(adj[i], j); push!(adj[j], i)
        end
    end

    visited  = falses(n)
    max_comp = 0
    for start in 1:n
        visited[start] && continue
        queue = [start]; visited[start] = true; comp_size = 0
        while !isempty(queue)
            v = popfirst!(queue); comp_size += 1
            for nb in adj[v]
                visited[nb] && continue
                visited[nb] = true
                push!(queue, nb)
            end
        end
        comp_size > max_comp && (max_comp = comp_size)
    end

    return (min_pair_dist=min_d, max_coord=maximum(coord), max_cluster=max_comp)
end

# ── Affine shear-strain ramp (Schmid geometry, glide-driving) ───────────────
# Same shear matrix convention as scripts/repulsive_core/shear_cell_energies.jl.
# Applied incrementally, warm-started from the previous relaxed structure, so
# the core can genuinely glide as strain accumulates.

shear_matrix(δ) = [1.0 δ/2 0.0; δ/2 1.0 0.0; 0.0 0.0 1.0]

function apply_affine_strain(sys, ε)
    cell_vectors = sys.cell.cell_vectors
    new_cell     = Tuple(SVector{3}(ε * ustrip.(v)) .* u"Å" for v in cell_vectors)

    atoms_new = Vector{AtomsBase.Atom}(undef, length(sys))
    for i in 1:length(sys)
        at   = sys[i]
        r    = ustrip.(at.position)
        rnew = ε * r
        atoms_new[i] = AtomsBase.Atom(at.species, collect(rnew) .* u"Å", missing)
    end
    return periodic_system(atoms_new, new_cell)
end

n_steps         = 30
δ_step          = 0.005
max_force_thresh  = 1000.0   # eV/Å  — repulsive-wall blow-up
max_energy_thresh = 50.0     # eV/atom above bulk equilibrium, either direction

history = NamedTuple[]
current_sys = sys_disloc_relaxed

for step in 0:n_steps
    global current_sys
    δ_cum = step * δ_step

    if step > 0
        strained = apply_affine_strain(current_sys, shear_matrix(δ_step))
        res = minimize_energy!(strained, model; variablecell=false, maxiters=2000)
        res.converged || @warn "step $step (δ=$(round(δ_cum,digits=4))): did not converge"
        current_sys = res.system
        E = ustrip(u"eV", res.energy)
        max_F = maximum(norm.(ustrip.(res.forces)))
    else
        E = ustrip(u"eV", res0.energy)
        max_F = maximum(norm.(ustrip.(res0.forces)))
    end

    diag = pair_diagnostics(current_sys, cluster_cutoff, nn_cutoff)
    e_per_atom = E / n_atoms
    e_excess   = e_per_atom - E_bulk_per_atom

    collapsed = diag.min_pair_dist < collapse_cutoff ||
                max_F > max_force_thresh ||
                abs(e_excess) > max_energy_thresh ||
                isnan(E) || isinf(E)

    rec = (step=step, delta=δ_cum, energy=E, e_per_atom=e_per_atom, e_excess=e_excess,
           max_force=max_F, min_pair_dist=diag.min_pair_dist, max_coord=diag.max_coord,
           max_cluster=diag.max_cluster, collapsed=collapsed, converged=(step==0 ? res0.converged : true))
    push!(history, rec)

    ExtXYZ.save(joinpath(outdir, @sprintf("step_%03d_delta_%.4f.extxyz", step, δ_cum)), current_sys)

    status = collapsed ? "COLLAPSE" : "stable"
    @info @sprintf("step %2d  δ=%.4f  E/atom=%.4f eV  min_r=%.3f Å  max|F|=%.2f eV/Å  %s",
                    step, δ_cum, e_per_atom, diag.min_pair_dist, max_F, status)

    if collapsed
        @warn "Collapse detected at cumulative shear strain δ=$(round(δ_cum,digits=4)) (step $step) — stopping ramp."
        break
    end
end

# ── Summary table ─────────────────────────────────────────────────────────────

@info "\n── Edge dislocation shear ramp summary ──────────────────────────"
@info @sprintf("%5s  %8s  %12s  %10s  %10s  %8s", "step", "δ", "E/atom(eV)", "min_r(Å)", "max|F|", "status")
for r in history
    status = r.collapsed ? "COLLAPSE" : "stable"
    @info @sprintf("%5d  %8.4f  %12.4f  %10.3f  %10.2f  %8s",
                   r.step, r.delta, r.e_per_atom, r.min_pair_dist, r.max_force, status)
end

# ── Plots ─────────────────────────────────────────────────────────────────────

deltas       = [r.delta for r in history]
e_excess_vec = [r.e_excess for r in history]
min_r_vec    = [r.min_pair_dist for r in history]
cluster_vec  = [r.max_cluster for r in history]

fig = Figure(size=(900, 900), fontsize=16)

ax1 = Axis(fig[1,1]; title="Excess energy/atom vs shear strain", xlabel="Shear strain δ", ylabel="E_excess (eV/atom)")
lines!(ax1, deltas, e_excess_vec; color=:steelblue, linewidth=2)

ax2 = Axis(fig[2,1]; title="Minimum pairwise distance vs shear strain", xlabel="Shear strain δ", ylabel="min r (Å)")
lines!(ax2, deltas, min_r_vec; color=:tomato, linewidth=2)
hlines!(ax2, [collapse_cutoff]; color=:red, linestyle=:dash, label="collapse cutoff")
hlines!(ax2, [cluster_cutoff]; color=:orange, linestyle=:dot, label="danger cutoff")
axislegend(ax2)

ax3 = Axis(fig[3,1]; title="Largest connected cluster vs shear strain", xlabel="Shear strain δ", ylabel="Cluster size (atoms)")
lines!(ax3, deltas, Float64.(cluster_vec); color=:darkorange, linewidth=2)

save(joinpath(outdir, "shear_ramp_diagnostics.png"), fig)

@info "Done. Results in $outdir"
