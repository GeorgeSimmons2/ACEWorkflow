# lib_verify.jl — the comparison primitives behind verify/compare_all.jl.
#
# Reproduction here is not one thing.  Three mechanisms sit in this pipeline and they
# have genuinely different reproducibility, so a single "does it match?" would be a lie
# in one direction or the other:
#
#   :exact        No QP anywhere in the member path — a seeded draw, or a closed-form
#                 correction.  Must match to machine precision.  Anything else is a bug,
#                 not solver noise.  (Measured: the naive arm reproduces bit-exactly.)
#
#   :converged    ONE well-posed QP.  The iterate path can differ but the optimum does
#                 not.  (Measured: theta_mean over a full rerun, max|Δ| = 9.12e-3.)
#
#   :statistical  A cutting-plane cascade — up to ~1600 rows added one at a time.  Any
#                 solver difference is amplified into a different accepted vector and a
#                 different member ORDER.  (Measured: max|Δ| = 4.31, the softest member's
#                 index moved 18 → 22.)  Only ensemble statistics are meaningful here.
#                 NEVER compare member to member, and never quote an index.

using DelimitedFiles, Statistics, Printf, LinearAlgebra

# :exact and :converged compare θ directly, so their tolerances are in units of θ.
const TIER_TOL = Dict(:exact => 0.0, :converged => 1e-2, :statistical => 0.01)

# :statistical compares distributions, so its threshold is in units of SAMPLING NOISE —
# see check_ensemble.  4.5σ leaves room for the worst of ~91 correlated comparisons.
const Z_TOL = parse(Float64, get(ENV, "Z_TOL", "4.5"))

struct Check
    label::String
    tier::Symbol
    status::Symbol          # :ok | :fail | :missing
    detail::String
end

readvec(p) = vec(readdlm(p, ','))
readmat(p) = readdlm(p, ',')

_absent(label, tier, published, repro) = Check(label, tier, :missing,
    !isfile(repro) ? "not produced yet: $repro" : "no published reference: $published")

# ── element-wise comparison of two parameter vectors ─────────────────────────
function check_vector(label, tier, published, repro)
    (isfile(published) && isfile(repro)) || return _absent(label, tier, published, repro)
    a, b = readvec(published), readvec(repro)
    length(a) == length(b) || return Check(label, tier, :fail,
        @sprintf("length differs: published %d, reproduced %d", length(a), length(b)))
    Δ   = abs.(a .- b)
    tol = TIER_TOL[tier]
    ok  = tier === :exact ? maximum(Δ) == 0.0 : maximum(Δ) <= tol
    Check(label, tier, ok ? :ok : :fail,
          @sprintf("max|Δ| = %.3e  median|Δ| = %.3e  ‖Δ‖/‖θ‖ = %.3e  (tol %.1e)",
                   maximum(Δ), median(Δ), norm(Δ)/max(norm(a), eps()), tol))
end

# ── distribution comparison for an ensemble held as rows of a CSV ────────────
#
# Compares the DISTRIBUTION, not the members: per-coefficient mean and standard
# deviation across the ensemble.  This is the only honest comparison for anything that
# came out of a cutting-plane cascade.
#
# THE THRESHOLD IS IN UNITS OF SAMPLING NOISE, not in units of θ.  Two *correct*
# 30-member draws from the same distribution have means differing by about
# σ·sqrt(2/n) = 0.26σ, so a fixed fractional tolerance would reject them every time.
# The statistic is therefore a z-score:
#
#     z_mean = |μ_A − μ_B| / (σ̄ · sqrt(2/n))            ~ N(0,1) if the two agree
#     z_std  = |σ_A − σ_B| / σ̄ · sqrt(2(n−1))            likewise, to first order
#
# and the verdict is on the WORST coefficient of the 91 (or 684).  With that many
# comparisons the largest |z| of a matching pair runs to ~3, hence a 4.5 threshold.
# σ̄ is pooled so neither run privileges itself.
function check_ensemble(label, tier, published, repro)
    (isfile(published) && isfile(repro)) || return _absent(label, tier, published, repro)
    A, B = readmat(published), readmat(repro)
    size(A, 2) == size(B, 2) || return Check(label, tier, :fail,
        @sprintf("width differs: published %d, reproduced %d", size(A,2), size(B,2)))
    size(A, 1) == size(B, 1) || return Check(label, tier, :fail,
        @sprintf("MEMBER COUNT differs: published %d, reproduced %d — sample_hypercube depends on N, so these are not comparable", size(A,1), size(B,1)))
    n = size(A, 1)
    n >= 3 || return Check(label, tier, :fail, "only $n members — no meaningful statistics")
    μA, μB = vec(mean(A, dims=1)), vec(mean(B, dims=1))
    σA, σB = vec(std(A,  dims=1)), vec(std(B,  dims=1))
    σ̄  = max.((σA .+ σB) ./ 2, eps())
    zμ = maximum(abs.(μA .- μB) ./ (σ̄ .* sqrt(2/n)))
    zσ = maximum(abs.(σA .- σB) ./ σ̄ .* sqrt(2*(n-1)))
    Check(label, tier, max(zμ, zσ) <= Z_TOL ? :ok : :fail,
          @sprintf("n=%d  worst z(mean) = %.2f, worst z(std) = %.2f  (threshold %.1f σ of sampling noise)",
                   n, zμ, zσ, Z_TOL))
end

# ── summary statistics of a per-member diagnostic column (min ω, relaxed a, …) ─
#
# Order-independent by construction: the column is sorted first, so a permuted committee
# compares as equal — which is right, because the permutation carries no meaning.
# readdlm hands back an Any matrix when any column is non-numeric (member_diagnostics
# carries an OSQP status string), so the column is pulled out as Float64 explicitly.
function check_distribution(label, tier, published, repro; col::Int=2, unit="", skipstart::Int=0)
    (isfile(published) && isfile(repro)) || return _absent(label, tier, published, repro)
    grab(p) = sort(Float64.(readdlm(p, ','; skipstart=skipstart)[:, col]))
    a, b = grab(published), grab(repro)
    length(a) == length(b) || return Check(label, tier, :fail,
        @sprintf("member count differs: published %d, reproduced %d", length(a), length(b)))
    stats(v) = (minimum(v), median(v), maximum(v))
    (mnA, mdA, mxA), (mnB, mdB, mxB) = stats(a), stats(b)
    scale = max(maximum(abs, a), eps())
    d   = max(abs(mnA-mnB), abs(mdA-mdB), abs(mxA-mxB)) / scale
    tol = TIER_TOL[tier]
    Check(label, tier, d <= tol ? :ok : :fail,
          @sprintf("n=%d  min %+.4f→%+.4f  median %+.4f→%+.4f  max %+.4f→%+.4f %s  (rel %.4f, tol %.3f)",
                   length(a), mnA, mnB, mdA, mdB, mxA, mxB, unit, d, tol))
end

function report(checks::Vector{Check}, io::IO=stdout)
    sym = Dict(:ok => "PASS", :fail => "FAIL", :missing => "  — ")
    println(io, "| quantity | tier | verdict | detail |")
    println(io, "|---|---|---|---|")
    for c in checks
        println(io, "| $(c.label) | `$(c.tier)` | **$(sym[c.status])** | $(c.detail) |")
    end
    nf = count(c -> c.status === :fail, checks)
    nm = count(c -> c.status === :missing, checks)
    no = count(c -> c.status === :ok, checks)
    println(io)
    println(io, "$no passed, $nf failed, $nm not yet produced.")
    return nf
end
