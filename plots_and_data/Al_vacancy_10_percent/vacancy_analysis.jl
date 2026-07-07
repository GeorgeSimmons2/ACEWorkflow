using ExtXYZ
using LinearAlgebra
using Statistics
using LsqFit
using CairoMakie

const ANG2FS_TO_SI = 1e-5

# -----------------------------
# Result container
# -----------------------------
struct DiffusionResult
    D_Å2_fs::Float64
    D_m2_s::Float64
    times::Vector{Float64}
    msd::Vector{Float64}
    slope::Float64
    intercept::Float64
end

# -----------------------------
# Load frames
# -----------------------------
load_frames(path) = collect(ExtXYZ.read_frames(path))

# -----------------------------
# Extract positions (YOUR FORMAT)
# -----------------------------
function get_positions(frames)

    r0 = frames[1]["arrays"]["pos"]
    natoms = size(r0, 2)   # IMPORTANT: shape is (3, N) or (N, 3) depending → check below
    nframes = length(frames)

    # detect orientation
    if size(r0,1) == 3
        # (3, N) format
        pos = zeros(Float64, nframes, natoms, 3)

        for t in 1:nframes
            r = frames[t]["arrays"]["pos"]
            for i in 1:natoms
                pos[t,i,1] = r[1,i]
                pos[t,i,2] = r[2,i]
                pos[t,i,3] = r[3,i]
            end
        end

    else
        # (N, 3) format
        natoms = size(r0,1)
        pos = zeros(Float64, nframes, natoms, 3)

        for t in 1:nframes
            r = frames[t]["arrays"]["pos"]
            for i in 1:natoms
                pos[t,i,1] = r[i,1]
                pos[t,i,2] = r[i,2]
                pos[t,i,3] = r[i,3]
            end
        end
    end

    return pos
end

# -----------------------------
# MSD
# -----------------------------
function compute_msd(pos)

    nframes, natoms, _ = size(pos)
    maxlag = div(nframes, 2)

    msd = zeros(Float64, maxlag - 1)

    for lag in 1:maxlag-1
        acc = 0.0
        count = 0

        for t in 1:(nframes - lag)
            @inbounds for i in 1:natoms
                dx = pos[t+lag,i,1] - pos[t,i,1]
                dy = pos[t+lag,i,2] - pos[t,i,2]
                dz = pos[t+lag,i,3] - pos[t,i,3]
                acc += dx*dx + dy*dy + dz*dz
                count += 1
            end
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
    fit = curve_fit(linear, x, y, [1.0, 0.0])
    return fit.param[1], fit.param[2]
end

# -----------------------------
# Removing equilibration
# -----------------------------
function discard_equilibration(frames; frac=0.2)
    start = Int(floor(length(frames) * frac)) + 1
    return frames[start:end]
end

# -----------------------------
# Main diffusion
# -----------------------------
function compute_diffusion(path::String, dt::Float64)

    frames = load_frames(path)
    frames = discard_equilibration(frames, frac=0.1)
    pos = get_positions(frames)

    msd = compute_msd(pos)
    steps = [frames[t]["info"]["step"] for t in 1:length(frames)]
    times = Float64.(steps[1:length(msd)])
    slope, intercept = fit_line(times, msd)

    D_Å2_fs = slope / 6.0
    D_m2_s  = D_Å2_fs * ANG2FS_TO_SI

    return DiffusionResult(D_Å2_fs, D_m2_s, times, msd, slope, intercept)
end
function plot_msd(result::DiffusionResult; traj_path::String)

    fig = Figure(size = (900, 650))

    ax = Axis(
        fig[1, 1],
        xlabel = "Time (fs)",
        ylabel = "MSD (Å²)",
        title = "Mean Squared Displacement"
    )

    # --- raw MSD points
    scatter!(ax, result.times, result.msd, markersize = 3, color = :blue)

    # --- MSD line
    lines!(ax, result.times, result.msd, linewidth = 2, color = :blue, label = "MSD")

    # --- linear fit
    fit_line = result.slope .* result.times .+ result.intercept
    lines!(ax, result.times, fit_line, linestyle = :dash, linewidth = 2,
           color = :red, label = "Linear fit")

    # --- optional equilibration marker (first 20%)
    eq_idx = Int(round(0.1 * length(result.times)))
    vlines!(ax, [result.times[eq_idx]], linestyle = :dot, color = :gray,
            label = "end equilibration")

    axislegend(ax, position = :lt)

    # -----------------------------
    # automatic save path
    # -----------------------------
    base = splitext(basename(traj_path))[1]
    dir = dirname(traj_path)
    outpath = joinpath(dir, base * "_msd.png")

    save(outpath, fig)

    println("Saved MSD plot → $outpath")

    return fig
end
# -----------------------------
# Run
# -----------------------------
traj1 = "models/Al_12_4_6A_2_subset_10_percent/results/100000.0_fs_NVT/md_trajectory.extxyz"
traj2 = "models/Al_12_4_6A_2_subset_10_percent/results/constrained_100000.0_fs_NVT/md_trajectory.extxyz"

result = compute_diffusion(traj1, 1.0)
println("D = $(result.D_m2_s) m²/s  ( $(result.D_Å2_fs) Å²/fs )")
plot_msd(result; traj_path=traj1)

constrained_result = compute_diffusion(traj2, 1.0)
println("Constrained D = $(constrained_result.D_m2_s) m²/s  ( $(constrained_result.D_Å2_fs) Å²/fs )")
plot_msd(constrained_result; traj_path=traj2)