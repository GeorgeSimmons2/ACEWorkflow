# aggregate_ensemble.jl — a(T) and α across the RLS model and both 20-member ensembles.
#
#   julia --project npt_al16/ensemble/aggregate_ensemble.jl
#
# Safe on partial results: reads whatever T<T>K/summary_row.csv files exist and says how
# many are missing, so it can be run while the arrays are still going.
#
# UNCERTAINTY IS THE RANGE.  Error bars are min–max across members, with the RLS (mean
# model) point drawn alongside — the question is whether the RLS prediction sits inside
# its own ensemble's range, and how wide each ensemble's range is.  Two consequences:
#
#   * A range grows with ensemble size, so 20-vs-20 is a fair comparison but these bars
#     cannot be compared with an ensemble of a different size.
#   * One transformed member would stretch a bar to cover a structural transformation
#     rather than thermal expansion.  So the bar covers members STILL FCC at that
#     temperature; transformed members are drawn as crosses and counted.  The table
#     reports the all-member range too, so nothing is hidden.
#
# α: OLS of a(T) over each member's FCC points, α = slope / fitted intercept — the same
# convention as thermal_expansion_vs_experiment/.  Two windows:
#   0–600 K    includes the relaxed T = 0 point
#   100–600 K  MD points only; insensitive to a jump between 0 K and the first MD point,
#              which is what a dynamically unstable member does (see npt_al16/aT_al16.png)
# A member needs ≥ 2 FCC points in a window to get an α there.
#
# Env: OUT (figure stem, default npt_al16/ensemble/ensemble_aT)  FIGW (pt, default 380)

using DelimitedFiles, Statistics, Printf, CairoMakie

ROOT   = @__DIR__
RUNS   = get(ENV, "RUNS", joinpath(ROOT, "runs"))
OUT    = get(ENV, "OUT", joinpath(ROOT, "ensemble_aT"))
FIGW   = parse(Float64, get(ENV, "FIGW", "380"))
TEMPS  = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
WILSON = joinpath(ROOT, "..", "..", "thermal_expansion_vs_experiment", "wilson_1941_aluminium.csv")
KX_TO_ANG = 1.00202
ENS = [("rls",           "RLS (mean model)", RGBf(0.0, 0.0, 0.0),       :diamond),
       ("unconstrained", "unconstrained",    RGBf(0.80, 0.15, 0.15),    :rect),
       ("constrained",   "constrained",      RGBf(0.0, 0.447, 0.698),   :circle)]

# ── load ─────────────────────────────────────────────────────────────────────
struct Member; k::Int; a0::Float64; T::Vector{Float64}; a::Vector{Float64}
               σ::Vector{Float64}; ω::Vector{Float64}; fcc::Vector{Bool}; end

function load(ens)
    d = joinpath(RUNS, ens); isdir(d) || return Member[]
    out = Member[]
    for md in sort(filter(x -> startswith(x, "member_"), readdir(d)))
        p = joinpath(d, md); f0 = joinpath(p, "a0.csv"); isfile(f0) || continue
        a0 = readdlm(f0, ','; skipstart=1)[1, 1]
        T, a, σ, ω, fcc = Float64[], Float64[], Float64[], Float64[], Bool[]
        for t in TEMPS
            f = joinpath(p, "T$(round(Int, t))K", "summary_row.csv"); isfile(f) || continue
            r = readdlm(f, ','; skipstart=1)
            push!(T, r[1]); push!(a, r[2]); push!(σ, r[3]); push!(ω, r[5]); push!(fcc, string(r[8]) == "true")
        end
        push!(out, Member(parse(Int, md[8:end]), a0, T, a, σ, ω, fcc))
    end
    out
end
data = Dict(e => load(e) for (e, _) in ENS)
n_rows = sum(length(m.T) for e in keys(data) for m in data[e]; init=0)
@printf("summary rows present: %d / %d\n", n_rows, 41 * length(TEMPS))

function linfit(x, y)
    x̄, ȳ = mean(x), mean(y); s = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
    ȳ - s * x̄, s
end
function alpha(m::Member, Tlo)
    keep = (m.T .>= Tlo) .& m.fcc
    T = m.T[keep]; a = m.a[keep]
    Tlo == 0 && (T = vcat(0.0, T); a = vcat(m.a0, a))
    length(T) >= 2 || return NaN
    c0, s = linfit(T, a); s / c0
end

# ── a(T) table ───────────────────────────────────────────────────────────────
println("\na(T) in Å.  Ensemble columns: [min, max] over members STILL FCC at that T, (n FCC / n run).")
@printf("%6s  %10s   %-30s   %-30s\n", "T (K)", "RLS", "unconstrained", "constrained")
for t in TEMPS
    rls = [m.a[i] for m in data["rls"] for i in eachindex(m.T) if m.T[i] == t]
    cells = map(("unconstrained", "constrained")) do e
        pts = [(m.a[i], m.fcc[i]) for m in data[e] for i in eachindex(m.T) if m.T[i] == t]
        isempty(pts) && return "—"
        ok = [a for (a, f) in pts if f]
        isempty(ok) ? @sprintf("none FCC (0/%d)", length(pts)) :
            @sprintf("[%.5f, %.5f] (%d/%d)", minimum(ok), maximum(ok), length(ok), length(pts))
    end
    @printf("%6.0f  %10s   %-30s   %-30s\n", t, isempty(rls) ? "—" : @sprintf("%.5f", rls[1]), cells...)
end

# ── α table ──────────────────────────────────────────────────────────────────
println("\nα (×10⁻⁵ K⁻¹), FCC points only.  Ensembles: [min, max] (members with an α), median")
αs = Dict()
for (e, label, _, _) in ENS, (Tlo, win) in ((0.0, "0–600 K"), (100.0, "100–600 K"))
    v = filter(!isnan, [alpha(m, Tlo) for m in data[e]]) .* 1e5
    αs[(e, Tlo)] = v
    isempty(v) && continue
    e == "rls" ? @printf("  %-17s %-10s %6.2f\n", label, win, v[1]) :
                 @printf("  %-17s %-10s [%5.2f, %5.2f]  median %5.2f  (n=%d)\n",
                         label, win, minimum(v), maximum(v), median(v), length(v))
end
for e in ("unconstrained", "constrained")
    bad = [m.k for m in data[e] if !all(m.fcc)]
    isempty(bad) || @printf("  %s members that left FCC at some T: %s\n", e, join(bad, ' '))
end
rls_α = get(αs, ("rls", 100.0), Float64[])
for e in ("unconstrained", "constrained")
    v = get(αs, (e, 100.0), Float64[])
    (isempty(v) || isempty(rls_α)) && continue
    @printf("  RLS α (100–600 K) inside the %s range? %s\n", e,
            minimum(v) <= rls_α[1] <= maximum(v) ? "yes" : "NO")
end

n_rows == 0 && (println("\nno data yet — figure skipped"); exit(0))

# ── figure: a(T) with range bars, RLS alongside ──────────────────────────────
W  = readdlm(WILSON, ','; header=true, comments=true, comment_char='#'); cw = vec(string.(W[2]))
Tw = Float64.(W[1][:, findfirst(==("T_C"), cw)]) .+ 273.15
aw = Float64.(W[1][:, findfirst(==("a_kX_obs"), cw)]) .* KX_TO_ANG
kw = Tw .<= 650

set_theme!(fontsize = 11)
fig = Figure(size = (FIGW, 0.8FIGW), figure_padding = (4, 10, 4, 4))
ax  = Axis(fig[1, 1]; xlabel = "Temperature (K)", ylabel = "Lattice constant a (Å)",
           xgridvisible = false, ygridvisible = false, xtickalign = 1, ytickalign = 1,
           xticks = 0:100:600)
offset = Dict("unconstrained" => -9.0, "constrained" => 9.0, "rls" => 0.0)
for (e, label, col, mk) in ENS
    e == "rls" && continue
    Tb, lo, hi, Tx, ax_ = Float64[], Float64[], Float64[], Float64[], Float64[]
    for t in TEMPS
        pts = [(m.a[i], m.fcc[i]) for m in data[e] for i in eachindex(m.T) if m.T[i] == t]
        ok = [a for (a, f) in pts if f]
        isempty(ok) || (push!(Tb, t + offset[e]); push!(lo, minimum(ok)); push!(hi, maximum(ok)))
        for (a, f) in pts; f || (push!(Tx, t + offset[e]); push!(ax_, a)); end
    end
    isempty(Tb) || rangebars!(ax, Tb, lo, hi; color = col, whiskerwidth = 6, linewidth = 1.6, label = "$label range")
    isempty(Tx) || scatter!(ax, Tx, ax_; color = col, marker = :xcross, markersize = 8)
end
for m in data["rls"]
    T = vcat(0.0, m.T); a = vcat(m.a0, m.a)
    lines!(ax, T, a; color = (:black, 0.6), linewidth = 1.0)
    scatter!(ax, T, a; color = :black, marker = :diamond, markersize = 7, label = "RLS (mean model)")
end
any(kw) && scatter!(ax, Tw[kw], aw[kw]; color = RGBf(0.45, 0.45, 0.45), marker = :utriangle,
                    markersize = 7, label = "experiment (Wilson 1941)")
axislegend(ax; position = :rb, framevisible = false, labelsize = 9, patchsize = (10, 8), rowgap = 0, unique = true)
save("$OUT.pdf", fig); save("$OUT.png", fig; px_per_unit = 4)
println("\nfigure → $OUT.{pdf,png}")

# ── figure: α per member, both windows ───────────────────────────────────────
fig2 = Figure(size = (FIGW, 0.7FIGW), figure_padding = (4, 10, 4, 4))
for (j, (Tlo, win)) in enumerate(((0.0, "0–600 K"), (100.0, "100–600 K")))
    a2 = Axis(fig2[1, j]; title = "α, $win", titlesize = 11, ylabel = j == 1 ? "α (10⁻⁵ K⁻¹)" : "",
              xticks = ([1, 2], ["uncon.", "con."]), xgridvisible = false, ygridvisible = false,
              xtickalign = 1, ytickalign = 1)
    for (x, e, col) in ((1, "unconstrained", RGBf(0.80,0.15,0.15)), (2, "constrained", RGBf(0.0,0.447,0.698)))
        v = get(αs, (e, Tlo), Float64[]); isempty(v) && continue
        scatter!(a2, fill(x, length(v)) .+ 0.12 .* (rand(length(v)) .- 0.5), v; color = (col, 0.6), markersize = 6)
        rangebars!(a2, [x + 0.28], [minimum(v)], [maximum(v)]; color = col, whiskerwidth = 6, linewidth = 1.6)
    end
    r = get(αs, ("rls", Tlo), Float64[])
    isempty(r) || hlines!(a2, r; color = :black, linestyle = :dash, linewidth = 1.0)
    xlims!(a2, 0.5, 2.6)
end
save("$(OUT)_alpha.pdf", fig2); save("$(OUT)_alpha.png", fig2; px_per_unit = 4)
println("figure → $(OUT)_alpha.{pdf,png}   (dashed = RLS)")
