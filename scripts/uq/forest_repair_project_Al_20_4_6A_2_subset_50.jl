# forest_repair_project_Al_20_4_6A_2_subset_50.jl
#
# Phonon-stable POPS committee via PROJECTION + cutting planes.  Two fixes over
# the pinned cutting-plane version:
#   (a) PROJECTION, not pinning.  Pinning a high-residual observation exactly
#       while demanding stability is infeasible (obs 33712 took 144 cuts and
#       stayed at −5 THz).  Instead each unstable member is projected onto the
#       nearest stable model in the fit metric:
#           min ½(θ̃−θ̃ᵢ)ᵀ C (θ̃−θ̃ᵢ)   s.t.  b′=0, Born, b″>0, cuts
#       (C = ApᵀAp + λPᵀP).  Always feasible; keeps the member close to θᵢ so
#       different members project to different stable points → dispersion (and
#       hence coverage) is preserved.
#   (b) DENSER detection mesh (N_cell=4 → 256 commensurate q) with a frequency
#       MARGIN.  The N=3 mesh was blind to off-mesh instabilities (obs 24270,
#       59231 were band-path-unstable with zero soft modes on the 108-q mesh).
#       A denser mesh + margin catches them and buffers the mesh↔band-path gap.
#
# Cut for a soft eigenvector v: row·θ = (v'D(θ)v)/‖v‖² built by FD of the basis
# energy along displacement v/√m (q=0, exact/commensurate).  Loop until the
# member clears the margin on the mesh.  Stable forest members pass untouched.
# Verified with full phonon bands + a test-set parity/coverage study.
#
# Outputs → models/Al_20_4_6A_2_subset_50_percent/results/forest_repair_project/
#
# Run:  julia --project [-t N] scripts/uq/forest_repair_project_Al_20_4_6A_2_subset_50.jl

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Random, Serialization
using SparseArrays, StaticArrays, Unitful, ForwardDiff, CairoMakie, OSQP, StatsBase
using ACEpotentials, ACEWorkflow
using ACEpotentials: potential_energy
using AtomsBuilder
using AtomsCalculatorsUtilities.SitePotentials: hessian
using ExtXYZ, AtomsCalculators
import AtomsCalculators: forces
import ACEWorkflow: phonon_committee

Random.seed!(1234)

element        = :Al
n_lev          = 5      # highest-leverage forest members
n_res          = 10     # highest-|residual| members (instability concentrates here)
n_rand         = 15     # random (typical) members
N_cell_detect  = 4      # denser folded mesh (256 commensurate q) for detection / cuts
N_cell_bands   = 3      # supercell for full-band verification
cut_margin_THz = 0.4    # cut any non-acoustic mode below this (margin buffers mesh↔path gap)
max_cuts       = 40     # safety cap on cutting-plane iterations per member
test_stride    = 10

# ── Model + forest ───────────────────────────────────────────────────────────
result     = load_model(element, 20, 4, 6, 2; dataset_name="subset_50_percent")
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
P          = result.P
Ap = Diagonal(result.W) * result.A / P
Yw = result.W .* result.Y
λ  = 1.0 / size(Ap, 1)
outdir = "$(result.dir)/results/forest_repair_project"; mkpath(outdir)
println("Model $(result.name): $n_params params.  Outputs → $outdir")

println("Leverage & residuals …")
C   = Symmetric(Ap' * Ap .+ λ .* (P' * P)); Cf = cholesky(C)
AtX = Cf \ Matrix(Ap'); θ̃ = Cf \ (Ap' * Yw)
leverage = vec(sum(Ap' .* AtX; dims=1)); residual = Yw .- Ap * θ̃
forest_member(i) = lin_params .+ (P \ (AtX[:, i] .* (residual[i] / leverage[i])))

# ── Born + a_eq base constraint rows ─────────────────────────────────────────
println("Building Born / a_eq rows …")
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
sys0 = ACEWorkflow.Elasticity.reference_system(element; a=a_eq)
L0   = ustrip.(ACEWorkflow.Elasticity.lattice_matrix(sys0.cell.cell_vectors))
eV_to_GPa = 160.2176621 / abs(det(L0))
H_basis = elastic_hessian_basis(model; element=element, a=a_eq)
c11_0 = reshape(H_basis, 36, n_params)[1, :]
c12_0 = reshape(H_basis, 36, n_params)[7, :]
c44_0 = reshape(H_basis, 36, n_params)[22, :]
born(θ) = (dot(c11_0, θ)*eV_to_GPa, dot(c12_0, θ)*eV_to_GPa, dot(c44_0, θ)*eV_to_GPa)
function lattice_basis(a_val)
    ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
        ACEWorkflow.Elasticity.reference_system(element; a=a_val), model))
end
b_prime        = ForwardDiff.derivative(lattice_basis, a_eq)
b_double_prime = ForwardDiff.derivative(a -> ForwardDiff.derivative(lattice_basis, a), a_eq)

base_rows  = vcat(c44_0', (c11_0 .- c12_0)', (c11_0 .+ 2 .* c12_0)', b_double_prime')
base_lower = [0.1, 1.0, 0.1, 1e-9]

# ── Folded-mesh stability check (member's own Hessian) ───────────────────────
sys_detect = bulk(element; a=a_eq*u"Å", cubic=true) * (N_cell_detect, N_cell_detect, N_cell_detect)
m_amu      = ustrip(sys_detect[1].mass)
ω2_cut     = (cut_margin_THz / FREQ_THz)^2

# returns (min_nonacoustic_freq_THz, soft_eigenvectors::Vector) where soft = modes with ω² < ω2_cut
function folded_soft_modes(θ)
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    H = Symmetric(ustrip.(hessian(sys_detect, model)) ./ m_amu)
    F = eigen(H)
    ω2 = F.values
    scale = maximum(abs.(ω2))
    acoustic_tol = 1e-6 * scale                  # |ω²| below this ⇒ acoustic translation
    soft = Int[]
    for k in eachindex(ω2)
        (abs(ω2[k]) <= acoustic_tol) && continue # skip the 3 acoustic zeros
        ω2[k] < ω2_cut && push!(soft, k)
    end
    nonac = [k for k in eachindex(ω2) if abs(ω2[k]) > acoustic_tol]
    fmin = isempty(nonac) ? 0.0 : (ω2m = minimum(ω2[nonac]); sign(ω2m)*sqrt(abs(ω2m))*FREQ_THz)
    return fmin, [F.vectors[:, k] for k in soft]
end

# curvature row for a real supercell displacement eigenvector v (q=0, commensurate)
pos_detect = nothing
function cut_row(v)
    mode = ComplexF64.(v)
    E(A) = apply_mode_design(model, sys_detect, mode, [0.0, 0.0, 0.0], A)   # displaces by A·v/√m
    dE(A) = ForwardDiff.derivative(E, A)
    h = 1e-5
    return (dE(h) .- dE(-h)) ./ (2h) ./ dot(v, v)      # row·θ = ω² of this mode
end

# ── Projection QP: nearest stable model to θᵢ in the fit metric C ─────────────
Cmat = sparse(Matrix(C))          # ApᵀAp + λPᵀP  (θ̃-space fit Hessian)
osqp = OSQP.Model()
function project(θi, rows, lowers)
    θ̃i = P * θi
    phys = sparse(vcat(b_prime', rows) / P)                 # b′ equality + inequalities
    l = vcat([0.0], lowers)
    u = vcat([0.0], fill(Inf, length(lowers)))
    OSQP.setup!(osqp; P=Cmat, q=Vector(-(Cmat * θ̃i)), A=phys, l=l, u=u,
                max_iter=4_000_000, check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    r = OSQP.solve!(osqp)
    return P \ r.x, r.info.status
end

# ── Committee draw ───────────────────────────────────────────────────────────
lev_idx = sortperm(leverage; rev=true)[1:n_lev]
res_idx = Int[]
for i in sortperm(abs.(residual); rev=true)
    i in lev_idx && continue; push!(res_idx, i); length(res_idx) == n_res && break
end
taken = Set(vcat(lev_idx, res_idx)); rand_idx = Int[]
while length(rand_idx) < n_rand
    i = rand(1:length(Yw)); (i in taken) && continue; push!(rand_idx, i); push!(taken, i)
end
selected = vcat(lev_idx, res_idx, rand_idx)
groups   = vcat(fill("leverage", n_lev), fill("residual", n_res), fill("random", n_rand))
println("Committee: $n_lev leverage + $n_res residual + $n_rand random = $(length(selected)) members\n")

# ── Cutting-plane repair per member ──────────────────────────────────────────
committee = Vector{Vector{Float64}}(undef, length(selected))
n_cuts    = zeros(Int, length(selected))
for (k, i) in enumerate(selected)
    θi = forest_member(i)
    θ  = θi
    rows = copy(base_rows); lowers = copy(base_lower)
    local fmin
    for it in 0:max_cuts
        fmin, soft = folded_soft_modes(θ)
        if isempty(soft)
            it > 0 && @printf("  obs %6d (%s): stable after %d iter, %d cut(s), min ω=%+.3f THz\n",
                              i, groups[k], it, n_cuts[k], fmin)
            break
        end
        it == max_cuts && (@warn "obs $i hit max_cuts, min ω=$(round(fmin;digits=3)) THz"; break)
        for v in soft
            rows = vcat(rows, cut_row(v)'); push!(lowers, ω2_cut)
        end
        n_cuts[k] += length(soft)
        θ, st = project(θi, rows, lowers)           # project ORIGINAL θᵢ onto growing stable set
        st == :Solved || @warn "obs $i projection QP status=$st at iter $it"
    end
    committee[k] = θ
end
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
@printf("\nRepaired (≥1 cut): %d / %d   [leverage %d, residual %d, random %d]\n",
        count(>(0), n_cuts), length(selected),
        count(>(0), n_cuts[groups.=="leverage"]), count(>(0), n_cuts[groups.=="residual"]),
        count(>(0), n_cuts[groups.=="random"]))
writedlm("$outdir/committee.csv", reduce(hcat, committee)', ',')

# ── Full-band verification ───────────────────────────────────────────────────
println("\n── Full-band verification ──────────────────────────────────")
x_vals, all_freqs, x_ticks, labels =
    phonon_committee(model, committee, result, element; N_cell=N_cell_bands, file_prefix="forest_repair_project/")
min_f  = [minimum(all_freqs[i + 1]) for i in 1:length(committee)]
n_imag = count(<(-0.05), min_f)
for (k, i) in enumerate(selected)
    C11, C12, C44 = born(committee[k])
    @printf("  %-9s obs %6d  cuts=%2d  ‖δθ‖=%.3f  C44=%6.1f  min ω=%+8.4f %s\n",
            groups[k], i, n_cuts[k], norm(committee[k] .- lin_params), C44, min_f[k], min_f[k] < -0.05 ? "✗" : "✓")
end
@printf("\n  → %d / %d committee members phonon-UNSTABLE (full band path)\n", n_imag, length(committee))

open("$outdir/members.csv", "w") do io
    println(io, "group,obs,n_cuts,dtheta_norm,C11_GPa,C12_GPa,C44_GPa,min_freq_THz")
    for (k, i) in enumerate(selected)
        C11, C12, C44 = born(committee[k])
        @printf(io, "%s,%d,%d,%.5f,%.3f,%.3f,%.3f,%.4f\n",
                groups[k], i, n_cuts[k], norm(committee[k] .- lin_params), C11, C12, C44, min_f[k])
    end
end

# ── Figure: kept (grey) / repaired (green) / unstable (red) + mean ───────────
fig = Figure(size=(900, 540))
ax  = Axis(fig[1, 1]; xlabel="Wave vector", ylabel="Frequency (THz)",
           title="$(result.name) — forest committee, projection repair (N=$N_cell_detect mesh)",
           xticks=(x_ticks, labels), xgridvisible=false)
for k in 1:length(committee)
    col = min_f[k] < -0.05 ? RGBAf(0.80,0.15,0.15,0.55) :
          n_cuts[k] > 0    ? RGBAf(0.10,0.55,0.20,0.55) : RGBAf(0.5,0.5,0.5,0.35)
    for b in 1:size(all_freqs[k+1], 1); lines!(ax, x_vals, all_freqs[k+1][b, :]; color=col, linewidth=1.0); end
end
for b in 1:size(all_freqs[1], 1); lines!(ax, x_vals, all_freqs[1][b, :]; color=RGBAf(0,0.3,0.7,0.95), linewidth=2.0); end
hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
vlines!(ax, x_ticks; color=(:black, 0.3), linewidth=0.8)
Legend(fig[1, 2],
       [LineElement(color=RGBAf(0.5,0.5,0.5,0.8),linewidth=2), LineElement(color=RGBAf(0.10,0.55,0.20,0.8),linewidth=2),
        LineElement(color=RGBAf(0.80,0.15,0.15,0.8),linewidth=2), LineElement(color=RGBAf(0,0.3,0.7,0.95),linewidth=2.5)],
       ["kept", "repaired (cuts)", "unstable", "mean"])
Label(fig[0, :], "$(count(>(0),n_cuts)) repaired, $n_imag/$(length(committee)) unstable after"; fontsize=13)
save("$outdir/bands.png", fig)

# ── Test-set parity + coverage study ─────────────────────────────────────────
println("\n── Test-set parity + coverage study ────────────────────────")
ACEpotentials.Models.set_committee!(model, committee)
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
testing_configs = ExtXYZ.load("data/Al/manual_df_test_Al.xyz")[1:test_stride:end]
pred_E=Float64[]; true_E=Float64[]; loE=Float64[]; hiE=Float64[]
pred_F=Float64[]; true_F=Float64[]; loF=Float64[]; hiF=Float64[]
for config in testing_configs
    E, co_E = @committee potential_energy(config, model)
    push!(pred_E, ustrip(E)); push!(loE, minimum(ustrip.(co_E))); push!(hiE, maximum(ustrip.(co_E)))
    push!(true_E, ustrip(config[:dft_energy]))
    if haskey(config[1], :dft_forces)
        F, co_F = @committee forces(config, model)
        append!(true_F, reduce(vcat, ustrip.([at[:dft_forces] for at in config])))
        append!(pred_F, reduce(vcat, ustrip.(F)))
        for i in eachindex(co_F[1])
            fi = reduce(hcat, ustrip(co_F[k][i]) for k in eachindex(co_F))
            append!(loF, vec(minimum(fi; dims=2))); append!(hiF, vec(maximum(fi; dims=2)))
        end
    end
end
function parity(t, p, lo, hi, xl, yl, path; col=:steelblue)
    rmse = sqrt(mean((p .- t).^2)); f = Figure(size=(600,600))
    ax = Axis(f[1,1]; xlabel=xl, ylabel=yl, title="RMSE = $(round(rmse,sigdigits=3))")
    errorbars!(ax, t, p, p.-lo, hi.-p; whiskerwidth=6, color=(col,0.5)); scatter!(ax, t, p; color=col, markersize=6)
    l = extrema([t;p]); lines!(ax, collect(l), collect(l); color=:black, linestyle=:dash); save(path, f); rmse
end
@printf("  energy RMSE = %.4g eV\n", parity(true_E, pred_E, loE, hiE, "DFT E (eV)", "ACE E (eV)", "$outdir/energy_parity.png"))
!isempty(true_F) && @printf("  force  RMSE = %.4g eV/Å\n",
    parity(true_F, pred_F, loF, hiF, "DFT F (eV/Å)", "ACE F (eV/Å)", "$outdir/force_parity.png"; col=:tomato))
function coverage(t, p, lo, hi; label, path, nbins=60)
    t,p,lo,hi = Float64.(t),Float64.(p),Float64.(lo),Float64.(hi)
    err = t.-p; mae = mean(abs.(err)); ne = err./mae
    be = vcat((t.-lo)./mae, (t.-hi)./mae); ev = mean((t.<lo).|(t.>hi)); bias = mean(err)/mae
    lim = maximum(abs.(vcat(ne,be))); ed = range(-lim, lim; length=nbins+1)
    d1=(h=fit(Histogram,ne,ed).weights; max.(h./(sum(h)*step(ed)),1e-3))
    d2=(h=fit(Histogram,be,ed).weights; max.(h./(sum(h)*step(ed)),1e-3))
    f=Figure(size=(560,520)); ax=Axis(f[2,1]; xlabel="Error / MAE", ylabel="density", yscale=log10,
        title="$label — coverage $(round((1-ev)*100,digits=1))% (EV $(round(ev*100,digits=1))%, bias $(round(bias*100,digits=0))%)")
    l1=stairs!(ax,ed[1:end-1],d1;step=:post,color=:black,linewidth=2); l2=stairs!(ax,ed[1:end-1],d2;step=:post,color=:orange,linewidth=2)
    ylims!(ax,1e-3,maximum(vcat(d1,d2))*3); Legend(f[1,1],[l1,l2],["test error","committee bound"];orientation=:horizontal,tellwidth=false)
    save(path,f); (1-ev)*100
end
@printf("  energy coverage = %.1f%%\n", coverage(true_E, pred_E, loE, hiE; label="Energy", path="$outdir/energy_error_vs_uncertainty.png"))
!isempty(true_F) && @printf("  force  coverage = %.1f%%\n",
    coverage(true_F, pred_F, loF, hiF; label="Force", path="$outdir/force_error_vs_uncertainty.png"))
ACEpotentials.Models.set_linear_parameters!(model, lin_params)

println("\n══ RESULT: $n_imag/$(length(committee)) unstable, $(count(>(0),n_cuts)) repaired ══")
println("All outputs → $outdir/")
