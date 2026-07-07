using ExtXYZ
using LinearAlgebra
using Statistics
using CairoMakie
using StaticArrays
using LsqFit

const ANG2FS_TO_SI = 1e-5   # Å²/fs → m²/s

# -----------------------------
# Load frames
# -----------------------------
load_frames(path) = collect(ExtXYZ.read_frames(path))

# -----------------------------
# Equilibration removal
# -----------------------------
function discard_equilibration(frames; frac=0.1)
    start = Int(floor(length(frames) * frac)) + 1
    return frames[start:end]
end

# -----------------------------
# Extract positions (unused for vacancy tracking now, kept for consistency)
# -----------------------------
function get_positions(frames)
    r0 = frames[1]["arrays"]["pos"]
    natoms = size(r0, 2)
    nframes = length(frames)

    pos = zeros(Float64, nframes, natoms, 3)

    for t in 1:nframes
        r = frames[t]["arrays"]["pos"]
        for i in 1:natoms
            pos[t,i,1] = r[1,i]
            pos[t,i,2] = r[2,i]
            pos[t,i,3] = r[3,i]
        end
    end

    return pos
end

# -----------------------------
# Reference lattice (sites)
# -----------------------------
function reference_sites(frames)
    return frames[1]["arrays"]["pos"]
end

# -----------------------------
# Wigner–Seitz occupancy mapping
# -----------------------------
function find_vacancy_site(frame, ref)

    r = frame["arrays"]["pos"]
    natoms = size(r, 2)
    nsites = size(ref, 2)

    occupied = falses(nsites)

    # assign each atom to nearest lattice site
    for i in 1:natoms

        ri1 = r[1,i]
        ri2 = r[2,i]
        ri3 = r[3,i]

        best_j = 1
        best_d = Inf

        for j in 1:nsites
            dx = ri1 - ref[1,j]
            dy = ri2 - ref[2,j]
            dz = ri3 - ref[3,j]

            d = dx*dx + dy*dy + dz*dz

            if d < best_d
                best_d = d
                best_j = j
            end
        end

        occupied[best_j] = true
    end

    # vacancy = unoccupied site
    vac_idx = findfirst(!, occupied)

    if vac_idx === nothing
        # fallback: choose least confidently occupied site
        counts = zeros(Int, length(occupied))

        # approximate: re-evaluate distances per site
        r = frame["arrays"]["pos"]
        ref = ref

        for j in 1:length(occupied)
            min_d = Inf
            for i in 1:size(r,2)
                dx = r[1,i] - ref[1,j]
                dy = r[2,i] - ref[2,j]
                dz = r[3,i] - ref[3,j]
                d = dx*dx + dy*dy + dz*dz
                min_d = min(min_d, d)
            end
            counts[j] = min_d
        end

        vac_idx = argmax(counts)
    end

    return SVector(ref[1,vac_idx], ref[2,vac_idx], ref[3,vac_idx])
end

# -----------------------------
# Track vacancy trajectory
# -----------------------------
function track_vacancy(frames)

    ref = reference_sites(frames)

    n = length(frames)
    vac = zeros(Float64, n, 3)

    for t in 1:n
        v = find_vacancy_site(frames[t], ref)
        vac[t,1] = v[1]
        vac[t,2] = v[2]
        vac[t,3] = v[3]
    end

    return vac
end

# -----------------------------
# Vacancy MSD
# -----------------------------
function compute_msd(x)

    n = size(x,1)
    maxlag = div(n,2)

    msd = zeros(maxlag-1)

    for lag in 1:maxlag-1
        acc = 0.0
        count = 0

        for t in 1:(n-lag)
            dx = x[t+lag,1] - x[t,1]
            dy = x[t+lag,2] - x[t,2]
            dz = x[t+lag,3] - x[t,3]

            acc += dx^2 + dy^2 + dz^2
            count += 1
        end

        msd[lag] = acc / count
    end

    return msd
end

# -----------------------------
# Fit
# -----------------------------
linear(x,p) = p[1] .* x .+ p[2]

function fit_line(x,y)
    p = [1.0, 0.0]
    fit = LsqFit.curve_fit(linear, x, y, p)
    return fit.param[1], fit.param[2]
end

# -----------------------------
# Main diffusion
# -----------------------------
function vacancy_diffusion(path::String, dt::Float64)

    frames = load_frames(path)
    frames = discard_equilibration(frames, frac=0.2)

    vac = track_vacancy(frames)
    msd = compute_msd(vac)

    times = dt .* collect(1:length(msd))

    slope, intercept = fit_line(times, msd)

    D_Å2_fs = slope / 6.0
    D_SI = D_Å2_fs * ANG2FS_TO_SI

    return D_Å2_fs, D_SI, times, msd, slope, intercept
end

# -----------------------------
# Plot
# -----------------------------
function plot_msd(times, msd, slope, intercept, path)

    fig = Figure(size=(900,600))

    ax = Axis(fig[1,1],
        xlabel="Time (fs)",
        ylabel="Vacancy MSD (Å²)",
        title="Vacancy Diffusion (Wigner–Seitz tracking)"
    )

    scatter!(ax, times, msd, markersize=3)
    lines!(ax, times, msd, linewidth=2)

    fit = slope .* times .+ intercept
    lines!(ax, times, fit, linestyle=:dash, color=:red)

    out = splitext(basename(path))[1] * "_vacancy_msd.png"
    save(out, fig)

    println("Saved → $out")

    return fig
end

# -----------------------------
# RUN
# -----------------------------
traj     = "models/Al_12_4_6A_2_subset_10_percent/results/100000.0_fs_NVT/md_trajectory.extxyz"
con_traj = "models/Al_12_4_6A_2_subset_10_percent/results/constrained_100000.0_fs_NVT/md_trajectory.extxyz"

D, D_SI, t, msd, slope, intercept = vacancy_diffusion(traj, 1.0)

println("Vacancy diffusion:")
println("D = $D_SI m²/s   ( $D Å²/fs )")

plot_msd(t, msd, slope, intercept, traj)

D2, D2_SI, t2, msd2, s2, i2 = vacancy_diffusion(con_traj, 1.0)

println("Constrained vacancy diffusion:")
println("D = $D2_SI m²/s   ( $D2 Å²/fs )")

plot_msd(t2, msd2, s2, i2, con_traj)