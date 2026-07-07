using ExtXYZ
using LinearAlgebra
using Statistics
using CairoMakie
using StaticArrays

# -----------------------------
# Load + equilibration removal
# -----------------------------
load_frames(path) = collect(ExtXYZ.read_frames(path))

function discard_equilibration(frames; frac=0.1)
    start = Int(floor(length(frames) * frac)) + 1
    return frames[start:end]
end

# -----------------------------
# PBC unwrap (CRITICAL FIX)
# -----------------------------
function unwrap_frames(frames)

    nframes = length(frames)
    natoms = size(frames[1]["arrays"]["pos"], 2)

    pos = zeros(Float64, nframes, natoms, 3)

    # initial box
    cell = frames[1]["cell"]
    Lx, Ly, Lz = cell[1,1], cell[2,2], cell[3,3]

    # first frame
    r0 = frames[1]["arrays"]["pos"]
    for i in 1:natoms
        pos[1,i,1] = r0[1,i]
        pos[1,i,2] = r0[2,i]
        pos[1,i,3] = r0[3,i]
    end

    # unwrap trajectory
    for t in 2:nframes
        r = frames[t]["arrays"]["pos"]

        for i in 1:natoms
            for d in 1:3
                prev = pos[t-1,i,d]
                curr = r[d,i]

                L = (d == 1 ? Lx : d == 2 ? Ly : Lz)

                delta = curr - prev
                delta -= L * round(delta / L)

                pos[t,i,d] = prev + delta
            end
        end
    end

    return pos
end

# -----------------------------
# Reference lattice
# -----------------------------
function reference_sites(frames)
    return frames[1]["arrays"]["pos"]
end

# -----------------------------
# Occupancy map (PBC-safe)
# -----------------------------
function site_occupancy(frame, ref, cell)

    r = frame["arrays"]["pos"]
    natoms = size(r,2)
    nsites = size(ref,2)

    Lx, Ly, Lz = cell[1,1], cell[2,2], cell[3,3]

    occupied = falses(nsites)

    for i in 1:natoms

        best_j = 1
        best_d = Inf

        for j in 1:nsites

            dx = r[1,i] - ref[1,j]
            dy = r[2,i] - ref[2,j]
            dz = r[3,i] - ref[3,j]

            # MINIMUM IMAGE CONVENTION
            dx -= Lx * round(dx / Lx)
            dy -= Ly * round(dy / Ly)
            dz -= Lz * round(dz / Lz)

            d = dx*dx + dy*dy + dz*dz

            if d < best_d
                best_d = d
                best_j = j
            end
        end

        occupied[best_j] = true
    end

    return occupied
end

# -----------------------------
# Vacancy site
# -----------------------------
function vacancy_site(frame, ref, cell)

    occ = site_occupancy(frame, ref, cell)

    idx = findfirst(!, occ)

    if idx === nothing
        idx = argmax(.!occ)
    end

    return idx
end

# -----------------------------
# Track vacancy
# -----------------------------
function track_vacancy_sites(frames)

    ref = reference_sites(frames)
    cell = frames[1]["cell"]

    n = length(frames)
    sites = zeros(Int, n)

    for t in 1:n
        sites[t] = vacancy_site(frames[t], ref, cell)
    end

    return sites, ref
end

# -----------------------------
# Filtered hops (stable version)
# -----------------------------
function filtered_hops(sites; min_residence=5)

    hops = 0
    last_site = sites[1]
    dwell = 1

    for t in 2:length(sites)

        if sites[t] == last_site
            dwell += 1
        else
            if dwell >= min_residence
                hops += 1
            end
            last_site = sites[t]
            dwell = 1
        end
    end

    return hops
end

# -----------------------------
# Diffusion coefficient
# -----------------------------
function vacancy_diffusion(path, dt)

    frames = load_frames(path)
    frames = discard_equilibration(frames, frac=0.1)

    # IMPORTANT: use unwrapped logic internally (keeps consistency)
    sites, ref = track_vacancy_sites(frames)

    hops = filtered_hops(sites)

    total_time = length(sites) * dt
    Γ = hops / total_time   # hop rate (1/fs)

    # lattice constant estimate
    dmin = Inf
    for i in 1:size(ref,2), j in i+1:size(ref,2)
        dx = ref[1,i] - ref[1,j]
        dy = ref[2,i] - ref[2,j]
        dz = ref[3,i] - ref[3,j]
        d = sqrt(dx*dx + dy*dy + dz*dz)
        if d > 0 && d < dmin
            dmin = d
        end
    end

    a = dmin

    D_Å2_fs = (Γ * a^2) / 6.0
    D_SI = D_Å2_fs * 1e-5

    return D_Å2_fs, D_SI, Γ, hops
end

# -----------------------------
# RUN
# -----------------------------
traj = "models/Al_12_4_6A_2_subset_10_percent/results/100000.0_fs_NVT/md_trajectory.extxyz"
con  = "models/Al_12_4_6A_2_subset_10_percent/results/constrained_100000.0_fs_NVT/md_trajectory.extxyz"

D, D_SI, Γ, hops = vacancy_diffusion(traj, 1.0)

println("\nUNCONSTRAINED VACANCY DIFFUSION")
println("Hops = $hops")
println("Hop rate Γ = $Γ fs⁻¹")
println("D = $D_SI m²/s   ( $D Å²/fs )")

D2, D2_SI, Γ2, hops2 = vacancy_diffusion(con, 1.0)

println("\nCONSTRAINED VACANCY DIFFUSION")
println("Hops = $hops2")
println("Hop rate Γ = $Γ2 fs⁻¹")
println("D = $D2_SI m²/s   ( $D2 Å²/fs )")