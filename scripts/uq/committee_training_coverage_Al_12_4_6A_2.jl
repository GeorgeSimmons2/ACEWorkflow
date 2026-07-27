# committee_training_coverage_Al_12_4_6A_2.jl
#
# Coverage of a saved committee on the TRAINING data.
#
#   pred[i,k] = A[i,:] · θ_k      for every design-matrix row i and committee member k
#   lo[i], hi[i] = min/max over k
#   covered[i] = lo[i] < Y[i] < hi[i]
#
# Uses the RAW design matrix A and data vector Y (not the weighted/preconditioned
# Ap, Yw used for fitting) so the numbers are in physical units and directly
# comparable to Y.
#
# Run:  julia --project -t 8 scripts/uq/committee_training_coverage_Al_12_4_6A_2.jl
#       (optionally pass a committee CSV path as the first argument)

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))

element, dataset = :Al, ""
result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
A, Y, lin = result.A, result.Y, result.lin_params
M, K = size(A)

committee_csv = length(ARGS) >= 1 ? ARGS[1] :
    "$(result.dir)/results/bandpath_undotted_ncell4_densek/committee_rejection.csv"
Θ = readdlm(committee_csv, ',')                     # N × K, one member per row
size(Θ, 2) == K || error("committee has $(size(Θ,2)) columns, design matrix has $K")
N = size(Θ, 1)
@printf("committee : %s\n", committee_csv)
@printf("            %d members × %d params\n", N, K)
@printf("design    : %d rows × %d params\n\n", M, K)

# predictions: M × N  (one column per committee member)
pred = A * Θ'
lo   = vec(minimum(pred; dims=2))
hi   = vec(maximum(pred; dims=2))
pt   = A * lin                                      # the point (LSQ) model

covered = (lo .< Y) .& (Y .< hi)
width   = hi .- lo
cov     = 100 * count(covered) / M

@printf("── coverage ────────────────────────────────────────────\n")
@printf("  covered            : %d / %d  (%.2f%%)\n", count(covered), M, cov)
@printf("  below lo / above hi: %d / %d\n", count(Y .<= lo), count(Y .>= hi))
@printf("  interval width     : median %.4g, mean %.4g, max %.4g\n",
        median(width), mean(width), maximum(width))
@printf("  |Y - point model|  : median %.4g, mean %.4g\n",
        median(abs.(Y .- pt)), mean(abs.(Y .- pt)))
@printf("  point-model RMSE   : %.4g\n", sqrt(mean((Y .- pt).^2)))

# how badly do the misses miss?  distance outside the envelope, in interval widths
miss = .!covered
if any(miss)
    d = max.(lo[miss] .- Y[miss], Y[miss] .- hi[miss])        # > 0 outside
    rel = d ./ max.(width[miss], eps())
    @printf("\n  misses: %d\n", count(miss))
    @printf("    distance outside envelope : median %.4g, max %.4g\n", median(d), maximum(d))
    @printf("    as a multiple of the width: median %.3g, p90 %.3g, max %.3g\n",
            median(rel), quantile(rel, 0.9), maximum(rel))
end

# does the envelope at least bracket the point model?  (sanity: it should)
@printf("\n  point model inside envelope: %.2f%%\n", 100*count((lo .< pt) .& (pt .< hi))/M)

outdir = "$(result.dir)/results"; stem = "$outdir/training_coverage_$(basename(dirname(committee_csv)))"
writedlm("$stem.csv", vcat(["row" "Y" "point" "lo" "hi" "covered"],
                           hcat(1:M, Y, pt, lo, hi, covered)), ',')
@printf("\nper-row table → %s.csv\n", stem)
