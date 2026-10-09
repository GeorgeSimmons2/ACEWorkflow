# ─────────────────────────────────────────────────────────────────────────────
# make_rdf_coordination.jl — the producer of rdf_<T>K.csv and coordination_<T>K.csv.
#
# fcc_compare_constrained_vs_naive.jl READS those two files; nothing in this repository
# wrote them.  In the original tree they were a side effect of two legacy FIGURE scripts
# (scripts/uq/fcc_stability_figure_* and fcc_instability_figure_*), so a pipeline rerun
# produced NPT trajectories with no RDF beside them and the FCC figure could not be drawn.
# That is the same shape of gap as the W positive-core θ: an input with no producer.
#
# The pair analysis below is lifted VERBATIM from fcc_stability_figure_Al_12_4_6A_2.jl
# (the `rdf_and_coord` function and the g(r) normalisation), with its constants:
#   n_equil = 10_000 steps, nn_cutoff = 3.3 Å, n_bins = 200.
# No physics or binning is changed — only the plotting is dropped.
#
# Each frame uses its OWN box, because the cell changes every frame under the barostat.
#
#   julia --project thermal_expansion_vs_experiment/make_rdf_coordination.jl <rundir> [T_K]
#
# Writes <rundir>/rdf_<T>K.csv and <rundir>/coordination_<T>K.csv, skipping either if it
# already exists (so it never overwrites the published files).  FORCE=1 to rewrite.
# ─────────────────────────────────────────────────────────────────────────────

using ExtXYZ, DelimitedFiles, Statistics, Printf

rundir = length(ARGS) >= 1 ? ARGS[1] : error("usage: make_rdf_coordination.jl <rundir> [T_K]")
T_K    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 300
FORCE  = get(ENV, "FORCE", "0") != "0"

n_equil, nn_cutoff, n_bins = 10_000, 3.3, 200

out_rdf = "$rundir/rdf_$(T_K)K.csv"
out_crd = "$rundir/coordination_$(T_K)K.csv"
if !FORCE && isfile(out_rdf) && isfile(out_crd)
    println("both files already exist, nothing to do (FORCE=1 to rewrite):\n  $out_rdf\n  $out_crd")
    exit(0)
end

traj = "$rundir/T$(T_K)K/md_trajectory.extxyz"
isfile(traj) || error("no trajectory at $traj")

frames = ExtXYZ.read_frames(traj)
steps  = [Int(f["info"]["step"]) for f in frames]
prod   = findall(>=(n_equil), steps)
@printf("%s T=%dK: %d frames, %d production (step ≥ %d)\n",
        basename(rundir), T_K, length(frames), length(prod), n_equil); flush(stdout)

boxes = [frames[f]["cell"][1,1] for f in prod]
poss  = [Matrix{Float64}(frames[f]["arrays"]["pos"]) for f in prod]

r_max = minimum(boxes)/2
dr    = r_max/n_bins
r_mid = collect(range(dr/2, r_max-dr/2; length=n_bins))

function rdf_and_coord(poss, boxes, n_bins, dr, r_max, nn_cutoff)
    n_at = size(poss[1], 2)
    pair_counts = zeros(n_bins); ρacc = 0.0; coord_all = Int[]
    for (f, L) in enumerate(boxes)
        p = poss[f]; ρacc += n_at/L^3
        z = zeros(Int, n_at)
        @inbounds for i in 1:n_at-1, j in i+1:n_at
            d1 = p[1,i]-p[1,j]; d2 = p[2,i]-p[2,j]; d3 = p[3,i]-p[3,j]
            d1 -= L*round(d1/L); d2 -= L*round(d2/L); d3 -= L*round(d3/L)
            r = sqrt(d1*d1 + d2*d2 + d3*d3)
            if r < nn_cutoff; z[i] += 1; z[j] += 1; end
            r < r_max || continue
            b = floor(Int, r/dr) + 1
            b <= n_bins && (pair_counts[b] += 2)
        end
        append!(coord_all, z)
    end
    return pair_counts, ρacc, coord_all
end

n_at = size(poss[1], 2)
t = @elapsed ((pair_counts, ρacc, coord_all) = rdf_and_coord(poss, boxes, n_bins, dr, r_max, nn_cutoff))
ρbar = ρacc/length(boxes)
g = [pair_counts[k]/(length(boxes)*n_at*4π*r_mid[k]^2*dr*ρbar) for k in 1:n_bins]
@printf("pair analysis %.1f s; %d bins to %.2f Å; ⟨coord⟩ = %.3f (FCC = 12)\n",
        t, n_bins, r_max, mean(coord_all))

zs    = collect(extrema(coord_all)[1]:extrema(coord_all)[2])
zfrac = [count(==(z), coord_all)/length(coord_all) for z in zs]

writedlm(out_rdf, vcat(["r_Ang" "g_r"], hcat(r_mid, g)), ',')
writedlm(out_crd, vcat(["Z" "fraction"], hcat(zs, zfrac)), ',')
println("→ $out_rdf\n→ $out_crd")
