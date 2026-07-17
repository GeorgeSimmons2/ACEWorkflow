using ACEWorkflow, ACEpotentials, AtomsBuilder, Unitful, CairoMakie, ExtXYZ, AtomsBase
using AtomsCalculators: potential_energy
using GeometryOptimization
using LinearAlgebra, StaticArrays, Statistics, Printf

# ─────────────────────────────────────────────────────────────────────────────
#  <100> edge dislocation quadrupole — shear stress-test, Al: unconstrained vs
#  C44-constrained model
#
#  Same construction/method as scripts/defects/edge_dislocation_100_stress_test.jl
#  (BCC W), applied here to FCC Al and run for TWO models loaded from the same
#  fit: the raw unconstrained model and the version re-fit with elastic-constant
#  (Born stability / C44 lower-bound) constraints via
#  scripts/elasticity/elastic_constraints.jl (see that directory's
#  C44_positive_constrained_params.csv). The applied shear here is a pure C44
#  deformation mode (xy shear in the cubic frame), so this test directly probes
#  whether the unconstrained model's weaker/less-controlled C44 lets the
#  dislocation core (or the surrounding lattice) go unstable at a lower applied
#  strain than the constrained model.
# ─────────────────────────────────────────────────────────────────────────────

element = :Al
result  = load_model(element, 12, 4, 6, 2; dataset_name="subset_10_percent")
model_unconstrained = result.model
model_constrained, _ = ACEpotentials.load_model("$(result.dir)/constrained_model.json")

outdir_base = joinpath(result.dir, "results", "edge_dislocation_100_quadrupole")
mkpath(outdir_base)

# ── Isotropic edge-dislocation displacement field (Hirth & Lothe form) ───────

function edge_displacement(dx, dy, b, ν)
    r2 = dx^2 + dy^2
    ux = (b / (2π)) * (atan(dy, dx) + (dx*dy) / (2*(1-ν)*r2))
    uy = -(b / (2π)) * ((1-2ν)/(4*(1-ν)) * log(r2) + (dx^2 - dy^2) / (4*(1-ν)*r2))
    return ux, uy
end

min_image(d, L) = d - L * round(d / L)

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

# ── Full pipeline for one model ──────────────────────────────────────────────

function run_dislocation_stress_test(model, label, outdir;
                                      nx=20, ny=20, nz=1,
                                      n_steps=30, δ_step=0.005,
                                      max_force_thresh=1000.0, max_energy_thresh=50.0)
    mkpath(outdir)

    a_eq = ACEWorkflow.relax_lattice_constant(model, element)
    elastic = ACEWorkflow.strain_hessian_GPa(model, element; a=a_eq)
    C11, C12, C44 = elastic.C[1,1], elastic.C[1,2], elastic.C[4,4]
    ν = C12 / (C11 + C12)

    @info "[$label] a_eq=$(round(a_eq,digits=4)) Å  C11=$(round(C11,digits=1)) C12=$(round(C12,digits=1)) C44=$(round(C44,digits=1)) GPa  ν=$(round(ν,digits=3))"

    b    = a_eq
    d_nn = a_eq / sqrt(2)          # FCC nearest-neighbour distance

    sys_perfect = bulk(element, a=a_eq*u"Å", cubic=true) * (nx, ny, nz)
    n_atoms = length(sys_perfect)
    Lx, Ly, Lz = nx*a_eq, ny*a_eq, nz*a_eq
    @info "[$label] Supercell: $n_atoms atoms, box = $(round(Lx,digits=2)) x $(round(Ly,digits=2)) x $(round(Lz,digits=2)) Å"

    E_bulk_per_atom = ustrip(potential_energy(sys_perfect, model)) / n_atoms

    cores = [
        (Lx/4,  Ly/4,   +1.0),
        (3Lx/4, Ly/4,   -1.0),
        (Lx/4,  3Ly/4,  -1.0),
        (3Lx/4, 3Ly/4,  +1.0),
    ]

    sys_disloc0 = apply_displacement_field(sys_perfect,
                                            (x,y) -> quadrupole_displacement(x, y, cores, Lx, Ly, b, ν))

    @info "[$label] Relaxing zero-strain dislocation quadrupole..."
    res0 = minimize_energy!(sys_disloc0, model; variablecell=false, maxiters=2000)
    res0.converged || @warn "[$label] Zero-strain relaxation did not converge!"
    sys_disloc_relaxed = res0.system

    E0 = ustrip(u"eV", res0.energy)
    E0_per_atom = E0 / n_atoms
    E_excess    = E0 - n_atoms * E_bulk_per_atom
    @info "[$label] Zero-strain quadrupole: E/atom=$(round(E0_per_atom,digits=4)) eV, excess=$(round(E_excess,digits=2)) eV"

    nn_cutoff       = 0.75 * d_nn
    cluster_cutoff  = 0.5  * d_nn
    collapse_cutoff = 0.3  * d_nn

    history = NamedTuple[]
    current_sys = sys_disloc_relaxed

    for step in 0:n_steps
        δ_cum = step * δ_step

        if step > 0
            strained = apply_affine_strain(current_sys, shear_matrix(δ_step))
            res = minimize_energy!(strained, model; variablecell=false, maxiters=2000)
            res.converged || @warn "[$label] step $step (δ=$(round(δ_cum,digits=4))): did not converge"
            current_sys = res.system
            E = ustrip(u"eV", res.energy)
            max_F = maximum(norm.(ustrip.(res.forces)))
        else
            E = E0
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
               max_cluster=diag.max_cluster, collapsed=collapsed)
        push!(history, rec)

        ExtXYZ.save(joinpath(outdir, @sprintf("step_%03d_delta_%.4f.extxyz", step, δ_cum)), current_sys)

        status = collapsed ? "COLLAPSE" : "stable"
        @info @sprintf("[%s] step %2d  δ=%.4f  E/atom=%.4f eV  min_r=%.3f Å  max|F|=%.2f eV/Å  %s",
                        label, step, δ_cum, e_per_atom, diag.min_pair_dist, max_F, status)

        if collapsed
            @warn "[$label] Collapse detected at δ=$(round(δ_cum,digits=4)) (step $step) — stopping ramp."
            break
        end
    end

    return (label=label, a_eq=a_eq, C11=C11, C12=C12, C44=C44, ν=ν,
            E_bulk_per_atom=E_bulk_per_atom, history=history)
end

# ── Run both models ───────────────────────────────────────────────────────────

nx, ny, nz = 20, 20, 1   # set to (10,10,1) for a quick smoke test
n_steps    = 30
δ_step     = 0.005

res_uncon = run_dislocation_stress_test(model_unconstrained, "unconstrained",
                                         joinpath(outdir_base, "unconstrained");
                                         nx=nx, ny=ny, nz=nz, n_steps=n_steps, δ_step=δ_step)
res_con   = run_dislocation_stress_test(model_constrained, "constrained",
                                         joinpath(outdir_base, "constrained");
                                         nx=nx, ny=ny, nz=nz, n_steps=n_steps, δ_step=δ_step)

# ── Summary table ─────────────────────────────────────────────────────────────

@info "\n── Al unconstrained vs C44-constrained: elastic constants ─────────"
for r in (res_uncon, res_con)
    @info @sprintf("%-14s  a_eq=%.4f Å  C11=%.1f  C12=%.1f  C44=%.1f GPa  ν=%.3f",
                   r.label, r.a_eq, r.C11, r.C12, r.C44, r.ν)
end

@info "\n── Shear ramp summary ───────────────────────────────────────────────"
for r in (res_uncon, res_con)
    @info "[$(r.label)]"
    for h in r.history
        status = h.collapsed ? "COLLAPSE" : "stable"
        @info @sprintf("  step %2d  δ=%.4f  E/atom=%.4f eV  min_r=%.3f Å  max|F|=%.2f eV/Å  %s",
                       h.step, h.delta, h.e_per_atom, h.min_pair_dist, h.max_force, status)
    end
end

# ── Comparison plots ──────────────────────────────────────────────────────────

fig = Figure(size=(900, 900), fontsize=16)

ax1 = Axis(fig[1,1]; title="Excess energy/atom vs shear strain (Al)", xlabel="Shear strain δ", ylabel="E_excess (eV/atom)")
lines!(ax1, [h.delta for h in res_uncon.history], [h.e_excess for h in res_uncon.history];
       color=:tomato, linewidth=2, label="unconstrained")
lines!(ax1, [h.delta for h in res_con.history], [h.e_excess for h in res_con.history];
       color=:steelblue, linewidth=2, label="C44-constrained")
axislegend(ax1)

ax2 = Axis(fig[2,1]; title="Minimum pairwise distance vs shear strain (Al)", xlabel="Shear strain δ", ylabel="min r (Å)")
lines!(ax2, [h.delta for h in res_uncon.history], [h.min_pair_dist for h in res_uncon.history];
       color=:tomato, linewidth=2, label="unconstrained")
lines!(ax2, [h.delta for h in res_con.history], [h.min_pair_dist for h in res_con.history];
       color=:steelblue, linewidth=2, label="C44-constrained")
axislegend(ax2)

ax3 = Axis(fig[3,1]; title="Largest connected cluster vs shear strain (Al)", xlabel="Shear strain δ", ylabel="Cluster size (atoms)")
lines!(ax3, [h.delta for h in res_uncon.history], Float64.([h.max_cluster for h in res_uncon.history]);
       color=:tomato, linewidth=2, label="unconstrained")
lines!(ax3, [h.delta for h in res_con.history], Float64.([h.max_cluster for h in res_con.history]);
       color=:steelblue, linewidth=2, label="C44-constrained")
axislegend(ax3)

save(joinpath(outdir_base, "shear_ramp_comparison.png"), fig)

@info "Done. Results in $outdir_base"
