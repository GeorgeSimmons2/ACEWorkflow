using ACEWorkflow, ACEpotentials, AtomsBuilder, Unitful, CairoMakie, ExtXYZ
using AtomsCalculators: potential_energy
using Molly, ACEWorkflow, Statistics, LinearAlgebra, DelimitedFiles

element = :W
# result = load_model(element, 12, 4, 6, 3)
# model = result.model
model, _ = ACEpotentials.load_model("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/models/W_20_4_5A_3/W_20_4_5A_3.json")
result   = (; dir="/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/models/W_20_4_5A_3")
if (constrained==true)
    ACEpotentials.Models.set_linear_parameters!(model, vec(readdlm("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/models/W_20_4_5A_3/repulsive_constrained_params.csv",',')))
    constrained_or_not = "con"
else
    constrained_or_not = "unc"
end
a_eq = ACEWorkflow.relax_lattice_constant(model, element)

# NVT note: volume is fixed at the 0 K equilibrium.  At 1200 K the lattice
# expands ~1-2%, so the simulation runs slightly over-compressed.  This is
# fine for a stability/convergence check; for thermodynamic properties run
# NPT first to equilibrate the volume at temperature.
sys = bulk(element, a=a_eq*u"Å", cubic=true) * (5,5,5)
sys_md = Molly.System(sys; force_units=u"eV/Å", energy_units=u"eV")
temp = 3600.0 * u"K"
dt =  0.5u"fs"
friction = 0.002*u"fs^-1"
n_steps   = 50_000
log_every = 50
dir_name  = "$(result.dir)/results/NPT_$(n_steps)_steps_$(ustrip(dt))_fs_dt_$(ustrip.(temp))_K_$(ustrip(friction))_per_fs_friction_$(constrained_or_not)_$(ustrip(sim_pressure))_GPa"
mkpath(dir_name)
sys_md = Molly.System(sys_md;
                      general_inters = (model,),
                      velocities = Molly.random_velocities(sys_md, temp),
                      loggers=(temp   = Molly.TemperatureLogger(log_every),
                               coords = Molly.CoordinatesLogger(log_every),
                               volume = Molly.VolumeLogger(log_every),
                               writer = Molly.TrajectoryWriter(20*log_every, "$(dir_name)/traj_log.xyz"; write_boundary=false),
                               energy = Molly.PotentialEnergyLogger(typeof(1.0u"eV"), log_every),))

simulator = Molly.Langevin(
   dt = dt,
   temperature=temp,
   friction=friction,
   coupling=Molly.MonteCarloBarostat(sim_pressure, temp, sys_md.boundary),)

Molly.simulate!(sys_md, simulator, n_steps)

@info("Temperature history:", sys_md.loggers.temp.history)
@info("Energy history:",      sys_md.loggers.energy.history)

# ── Save trajectory ───────────────────────────────────────────────────────────
# Write one ExtXYZ frame per logged step, including temperature and energy.
let
    traj_path = "$(dir_name)/md_trajectory.extxyz"
    energies  = sys_md.loggers.energy.history
    temps     = sys_md.loggers.temp.history
    species   = [string(Molly.atomic_symbol(sys_md, i)) for i in 1:length(sys_md)]
    n_atoms   = length(sys_md)
    volumes   = sys_md.loggers.volume.history
    box       = sys_md.boundary.side_lengths

    frames = Dict{String,Any}[]
    for (f, frame_coords) in enumerate(sys_md.loggers.coords.history)
        pos_matrix = reduce(hcat, [ustrip.(u"Å", c) for c in frame_coords])   # 3 × N, Å
        frame_dict = Dict{String,Any}(
            "N_atoms"  => n_atoms,
            "info"     => Dict{String,Any}(
                "Lattice"     => string(ustrip(u"Å", cbrt(volumes[f]))) * " 0.0 0.0 0.0 " *
                                 string(ustrip(u"Å", cbrt(volumes[f]))) * " 0.0 0.0 0.0 " *
                                 string(ustrip(u"Å", cbrt(volumes[f]))),
                "Properties"  => "species:S:1:pos:R:3",
                "energy"      => ustrip(energies[f]),
                "temperature" => ustrip(temps[f]),
                "step"        => (f - 1) * log_every,
            ),
            "arrays"   => Dict{String,Any}(
                "species" => species,
                "pos"     => pos_matrix,
            ),
        )
        push!(frames, frame_dict)
    end
    ExtXYZ.write_frames(traj_path, frames)
    @info "Trajectory saved to $traj_path ($(length(frames)) frames)"
end
# ── Analysis ──────────────────────────────────────────────────────────────────

coords_history = sys_md.loggers.coords.history   # Vector of Vector of coords
n_frames = length(coords_history)
n_atoms  = length(coords_history[1])
box_len  = ustrip(u"Å", sys_md.boundary.side_lengths[1])   # Å

# ── RDF ───────────────────────────────────────────────────────────────────────
# Compute g(r) averaged over all frames to check for unphysical atom overlap.
# A spike at r < ~2 Å (for Al) indicates atoms sitting on top of each other.

r_max  = box_len / 2
n_bins = 200
dr     = r_max / n_bins
r_edges = range(0.0, r_max; length=n_bins+1)
r_mids  = collect(r_edges[1:end-1] .+ dr/2)
rdf_counts = zeros(n_bins)

for frame in coords_history
    pos = [ustrip.(u"Å", c) for c in frame]   # Å
    for i in 1:n_atoms, j in i+1:n_atoms
        d = pos[i] - pos[j]
        # minimum image convention
        d = d .- box_len .* round.(d ./ box_len)
        r = norm(d)
        r < r_max || continue
        bin = floor(Int, r / dr) + 1
        bin <= n_bins && (rdf_counts[bin] += 2)   # count both i→j and j→i
    end
end

# Normalise to get g(r)
ρ = n_atoms / box_len^3
rdf = [rdf_counts[k] / (n_frames * n_atoms * 4π * r_mids[k]^2 * dr * ρ)
       for k in 1:n_bins]

fig_rdf = Figure(size=(700, 400))
ax_rdf  = Axis(fig_rdf[1,1];
               title  = "Radial Distribution Function — Al at 1200 K (NVT)",
               xlabel = "r (Å)", ylabel = "g(r)")
lines!(ax_rdf, r_mids, rdf; color=:steelblue)
vlines!(ax_rdf, [2.0]; color=(:red, 0.6), linestyle=:dash, linewidth=1.2)
text!(ax_rdf, 2.05, maximum(rdf)*0.9; text="overlap\nwarning", fontsize=11, color=:red)
save("$(dir_name)/md_rdf.png", fig_rdf)

# ── MSD ───────────────────────────────────────────────────────────────────────
# Mean squared displacement relative to frame 0.
# A plateau indicates a solid (diffusion frozen); linear growth indicates liquid.
# Convergence of the MD is confirmed once temperature and energy are stable AND
# the MSD is no longer drifting unexpectedly.

ref_pos = [ustrip.(u"Å", c) for c in coords_history[1]]
msd = Float64[]
for frame in coords_history
    pos = [ustrip.(u"Å", c) for c in frame]
    disp2 = 0.0
    for i in 1:n_atoms
        d = pos[i] - ref_pos[i]
        d = d .- box_len .* round.(d ./ box_len)
        disp2 += sum(abs2, d)
    end
    push!(msd, disp2 / n_atoms)
end

t_axis = (0:n_frames-1) .* (log_every * 1.0)   # femtoseconds

fig_msd = Figure(size=(700, 400))
ax_msd  = Axis(fig_msd[1,1];
               title  = "Mean Squared Displacement — Al at 1200 K (NVT)",
               xlabel = "Time (fs)", ylabel = "MSD (Å²)")
lines!(ax_msd, t_axis, msd; color=:tomato)
save("$(dir_name)/md_msd.png", fig_msd)

# ── Temperature & energy convergence ─────────────────────────────────────────
t_temps = (0:length(sys_md.loggers.temp.history)-1) .* (log_every * 1.0)
temps_K = ustrip.(sys_md.loggers.temp.history)
energies_eV = ustrip.(sys_md.loggers.energy.history)
volumes = ustrip.(sys_md.loggers.volume.history)

fig_conv = Figure(size=(700, 600))
ax_T = Axis(fig_conv[1,1]; title="Temperature", xlabel="Time (fs)", ylabel="T (K)")
ax_E = Axis(fig_conv[2,1]; title="Potential Energy", xlabel="Time (fs)", ylabel="E (eV)")
ax_V = Axis(fig_conv[3,1]; title="Volume", xlabel="Time (fs)", ylabel="V (nm^3)")
lines!(ax_T, t_temps, temps_K;    color=:steelblue)
hlines!(ax_T, [ustrip(temp)];     color=:black, linestyle=:dash, linewidth=0.8)
lines!(ax_E, t_temps, energies_eV; color=:tomato)
lines!(ax_V, t_temps, volumes; color=:green)
save("$(dir_name)/md_convergence.png", fig_conv)

# ── Cluster analysis ──────────────────────────────────────────────────────────
# Three complementary diagnostics:
#
#  1. Minimum pair distance per frame  — a dip below ~2 Å flags close encounters.
#  2. Coordination number distribution — atoms in a cluster have anomalously
#     high coordination; the distribution should be narrow around 12 for FCC Al.
#  3. Largest connected component      — build a graph where atoms are connected
#     if r < cluster_cutoff (set < first RDF peak); any component larger than
#     ~2 atoms indicates a genuine cluster.

# Cutoffs (Å) — adjust to your system
nn_cutoff      = 3.3   # first RDF minimum for Al; used for coordination number
cluster_cutoff = 2.2   # well below first peak (~2.8 Å); atoms closer than this
                       # are considered "overlapping / clustered"

min_pair_dist  = Float64[]
max_coord      = Int[]
largest_cluster_size = Int[]

for frame in coords_history
    pos = [ustrip.(u"Å", c) for c in frame]
    n   = length(pos)

    min_d    = Inf
    coord    = zeros(Int, n)
    adj      = [Int[] for _ in 1:n]   # adjacency list for cluster graph

    for i in 1:n, j in i+1:n
        d = pos[i] - pos[j]
        d = d .- box_len .* round.(d ./ box_len)
        r = norm(d)

        r < min_d && (min_d = r)

        if r < nn_cutoff
            coord[i] += 1
            coord[j] += 1
        end

        if r < cluster_cutoff
            push!(adj[i], j)
            push!(adj[j], i)
        end
    end

    # BFS to find connected components
    visited  = falses(n)
    max_comp = 0
    for start in 1:n
        visited[start] && continue
        # BFS
        queue = [start]
        visited[start] = true
        comp_size = 0
        while !isempty(queue)
            v = popfirst!(queue)
            comp_size += 1
            for nb in adj[v]
                visited[nb] && continue
                visited[nb] = true
                push!(queue, nb)
            end
        end
        comp_size > max_comp && (max_comp = comp_size)
    end

    push!(min_pair_dist,       min_d)
    push!(max_coord,           maximum(coord))
    push!(largest_cluster_size, max_comp)
end

# ── Plot cluster diagnostics ──────────────────────────────────────────────────
fig_cl = Figure(size=(800, 700))

ax1 = Axis(fig_cl[1,1];
           title="Minimum pair distance per frame",
           xlabel="Time (fs)", ylabel="Min r (Å)")
lines!(ax1, t_axis, min_pair_dist; color=:steelblue)
hlines!(ax1, [cluster_cutoff]; color=:red, linestyle=:dash, linewidth=0.8,
        label="cluster cutoff")
axislegend(ax1)

ax2 = Axis(fig_cl[2,1];
           title="Maximum coordination number per frame (cutoff=$(nn_cutoff) Å)",
           xlabel="Time (fs)", ylabel="Max coord. number")
lines!(ax2, t_axis, Float64.(max_coord); color=:tomato)

ax3 = Axis(fig_cl[3,1];
           title="Largest connected cluster (cutoff=$(cluster_cutoff) Å)",
           xlabel="Time (fs)", ylabel="Cluster size (atoms)")
lines!(ax3, t_axis, Float64.(largest_cluster_size); color=:darkorange)
hlines!(ax3, [2.0]; color=:black, linestyle=:dash, linewidth=0.8,
        label="isolated pair")
axislegend(ax3)

save("$(dir_name)/md_cluster_analysis.png", fig_cl)

# Print summary
@info "Cluster analysis summary:"
@info "  Min pair distance (all frames): $(minimum(min_pair_dist)) Å"
@info "  Max coordination number seen:   $(maximum(max_coord))"
@info "  Max cluster size seen:          $(maximum(largest_cluster_size)) atoms"
if minimum(min_pair_dist) < cluster_cutoff
    @warn "Atoms came closer than $(cluster_cutoff) Å — possible unphysical overlap!"
else
    @info "No atoms closer than $(cluster_cutoff) Å — no clustering detected."
end