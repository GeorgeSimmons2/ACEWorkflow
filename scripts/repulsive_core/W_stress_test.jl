using ACEWorkflow, ACEpotentials, AtomsBuilder, Unitful, CairoMakie, ExtXYZ
using AtomsCalculators: potential_energy, forces
using Molly, Statistics, LinearAlgebra

# ─────────────────────────────────────────────────────────────────────────────
#  Tungsten extreme-conditions stress test
#
#  W melting point ≈ 3695 K  (highest of all metals)
#  Tests:
#    1. Thermal stability at 3000 K, 5000 K, 7000 K (above melting)
#    2. Compressed volumes: 0.70, 0.80, 0.90, 1.0 × V_eq
#    3. Combined: compressed + high temperature (radiation-damage regime)
#    4. Max force and energy per atom tracked for blow-up detection
# ─────────────────────────────────────────────────────────────────────────────

element = :W
result  = load_model(element, 20, 4, 5, 3)
model   = result.model

outdir = joinpath(result.dir, "results", "stress_test")
mkpath(outdir)

a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@info "Equilibrium lattice constant: $(round(a_eq, digits=4)) Å"

# W BCC conventional cell: 2 atoms, so V_eq per atom = a_eq³ / 2
V_eq_per_atom = a_eq^3 / 2
@info "V_eq per atom: $(round(V_eq_per_atom, digits=4)) Å³"

# ─────────────────────────────────────────────────────────────────────────────
#  Parameters
# ─────────────────────────────────────────────────────────────────────────────

temperatures     = [3000.0, 5000.0, 7000.0]   # K — straddles melting point
volume_fractions = [0.70, 0.80, 0.90, 1.0]    # V / V_eq  (<1 = compression)
supercell        = (4, 4, 4)                   # 128 atoms
n_steps          = 2_000                       # steps per run
dt               = 0.5u"fs"                    # shorter timestep for stability
log_every        = 50
friction         = 0.5u"ps^-1"                 # strong coupling to thermostat

# Blow-up detection thresholds
max_force_threshold  = 1000.0   # eV/Å — unphysical if exceeded
max_energy_threshold = 100.0    # eV/atom above equilibrium

# ─────────────────────────────────────────────────────────────────────────────
#  Helper: run one MD and return summary statistics
# ─────────────────────────────────────────────────────────────────────────────

function run_stress_md(model, element, a, temp_K; n_steps, dt, log_every, friction,
                       label="")

    temp = temp_K * u"K"
    sys  = rattle!(bulk(element, a=a*u"Å", cubic=true) * supercell, 0.01)
    n_at = length(sys)

    sys_md = Molly.System(sys; force_units=u"eV/nm", energy_units=u"eV")
    sys_md = Molly.System(sys_md;
                          general_inters = (model,),
                          velocities     = Molly.random_velocities(sys_md, temp),
                          loggers = (
                              temp   = Molly.TemperatureLogger(log_every),
                              coords = Molly.CoordinatesLogger(log_every),
                              energy = Molly.PotentialEnergyLogger(typeof(1.0u"eV"), log_every),
                          ))

    simulator = Molly.Langevin(dt=dt, temperature=temp, friction=friction)

    t_elapsed = @elapsed Molly.simulate!(sys_md, simulator, n_steps; n_threads=1)

    temps_K     = ustrip.(sys_md.loggers.temp.history)
    energies_eV = ustrip.(sys_md.loggers.energy.history)
    e_per_atom  = energies_eV ./ n_at

    # Check max force on final frame
    final_coords = sys_md.loggers.coords.history[end]
    final_sys    = Molly.System(
        Molly.System(bulk(element, a=a*u"Å", cubic=true) * supercell;
                     force_units=u"eV/nm", energy_units=u"eV");
        general_inters = (model,),
        coords = final_coords,
        velocities = sys_md.velocities,
    )
    F = forces(final_sys, model)
    max_F = maximum(norm.(ustrip.(F)))

    blowup = max_F > max_force_threshold ||
             maximum(abs.(e_per_atom)) > max_energy_threshold

    return (
        label         = label,
        temp_K        = temp_K,
        a             = a,
        n_atoms       = n_at,
        mean_temp     = mean(temps_K),
        std_temp      = std(temps_K),
        mean_e_atom   = mean(e_per_atom),
        min_e_atom    = minimum(e_per_atom),
        max_e_atom    = maximum(e_per_atom),
        max_force     = max_F,
        blowup        = blowup,
        t_elapsed_s   = t_elapsed,
        temps_K       = temps_K,
        e_per_atom    = e_per_atom,
    )
end

# ─────────────────────────────────────────────────────────────────────────────
#  Run the stress matrix
# ─────────────────────────────────────────────────────────────────────────────

results = []
for vf in volume_fractions
    # Compressed lattice constant: a = a_eq × (V/V_eq)^(1/3)
    a = a_eq * vf^(1/3)
    for T in temperatures
        label = "V=$(round(Int, vf*100))%_T=$(round(Int,T))K"
        @info "Running: $label  (a = $(round(a, digits=3)) Å)"
        r = run_stress_md(model, element, a, T;
                          n_steps=n_steps, dt=dt, log_every=log_every,
                          friction=friction, label=label)
        push!(results, r)
        status = r.blowup ? "BLOW-UP" : "stable"
        @info "  → $status | ⟨E⟩ = $(round(r.mean_e_atom, digits=3)) eV/atom | max F = $(round(r.max_force, digits=1)) eV/Å | $(round(r.t_elapsed_s, digits=1)) s"
    end
end

# ─────────────────────────────────────────────────────────────────────────────
#  Summary table
# ─────────────────────────────────────────────────────────────────────────────

@info "\n── Stress test summary ──────────────────────────────────────────"
@info @sprintf("%-30s  %6s  %10s  %10s  %8s", "Run", "Status",
               "⟨E⟩/atom", "max|F|", "time(s)")
for r in results
    status = r.blowup ? "BLOW-UP ⚠" : "stable ✓"
    @info @sprintf("%-30s  %9s  %8.3f eV  %7.1f eV/Å  %6.1f s",
                   r.label, status, r.mean_e_atom, r.max_force, r.t_elapsed_s)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Plots
# ─────────────────────────────────────────────────────────────────────────────

# Colour map: one colour per temperature
temp_colors = Dict(3000.0 => :steelblue, 5000.0 => :darkorange, 7000.0 => :crimson)
vf_linestyles = Dict(0.70 => :dot, 0.80 => :dash, 0.90 => :dashdot, 1.0 => :solid)

t_axis = (0:length(results[1].temps_K)-1) .* (log_every * ustrip(u"fs", dt))

# Energy per atom vs time for all runs
fig_e = Figure(size=(1000, 500))
ax_e  = Axis(fig_e[1,1];
             title  = "W stress test — potential energy per atom",
             xlabel = "Time (fs)", ylabel = "E per atom (eV)")
for r in results
    lines!(ax_e, t_axis, r.e_per_atom;
           color     = temp_colors[r.temp_K],
           linestyle = vf_linestyles[r.a / a_eq |> vf -> round(vf, digits=2) |>
                           vf -> findmin(abs.(collect(volume_fractions) .- vf))[2] |>
                           i -> volume_fractions[i]],
           label     = r.label, linewidth=1.2)
end
axislegend(ax_e; position=:rb, nbanks=2, labelsize=9)
save(joinpath(outdir, "stress_energy.png"), fig_e)

# Temperature convergence
fig_t = Figure(size=(1000, 500))
ax_t  = Axis(fig_t[1,1];
             title  = "W stress test — thermostat temperature",
             xlabel = "Time (fs)", ylabel = "T (K)")
for r in results
    lines!(ax_t, t_axis, r.temps_K;
           color     = temp_colors[r.temp_K],
           linestyle = :solid,
           label     = r.label, linewidth=1.0)
end
axislegend(ax_t; position=:rb, nbanks=2, labelsize=9)
save(joinpath(outdir, "stress_temperature.png"), fig_t)

# Heatmap: mean energy per atom as function of (V/V_eq, T)
mean_e_matrix = [r.mean_e_atom
                 for vf in volume_fractions, T in temperatures
                 if any(r -> r.a ≈ a_eq*vf^(1/3) && r.temp_K ≈ T, results)]
mean_e_matrix = reshape([r.mean_e_atom for r in results],
                        length(volume_fractions), length(temperatures))

fig_hm = Figure(size=(600, 400))
ax_hm  = Axis(fig_hm[1,1];
              title   = "Mean energy per atom (eV) — W stress test",
              xlabel  = "Temperature (K)",
              ylabel  = "Volume fraction V/V_eq",
              xticks  = (1:length(temperatures), string.(round.(Int, temperatures))),
              yticks  = (1:length(volume_fractions), string.(volume_fractions)))
hm = heatmap!(ax_hm, mean_e_matrix; colormap=:RdYlBu_r)
Colorbar(fig_hm[1,2], hm; label="⟨E⟩/atom (eV)")
for (j, T) in enumerate(temperatures), (i, vf) in enumerate(volume_fractions)
    r = results[(i-1)*length(temperatures) + j]
    text!(ax_hm, j, i;
          text     = r.blowup ? "✗" : "✓",
          align    = (:center, :center),
          fontsize = 16,
          color    = r.blowup ? :red : :white)
end
save(joinpath(outdir, "stress_heatmap.png"), fig_hm)

@info "All plots saved to $outdir"
