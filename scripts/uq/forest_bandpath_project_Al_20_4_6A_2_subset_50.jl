# forest_bandpath_project_Al_20_4_6A_2_subset_50.jl
#
# Phonon-stable POPS committee by constraining along the BAND PATH itself — the
# physically relevant slice, and exactly what phonon_committee verifies, so
# there is no mesh↔path gap.  The primitive dynamical matrix is LINEAR in θ:
#
#     D(q, θ) = Σ_k θ_k D_k(q)          D_k(q) = Bloch transform of H_k
#
# where H_k = ∂²(basis_k energy)/∂r² is the supercell Hessian of basis function
# k.  So the curvature along phonon mode e at ANY wavevector q is an EXACT
# linear row:  row·θ = eᴴ D(q,θ) e = Σ_k (eᴴ D_k(q) e) θ_k.  No commensurate-
# mesh restriction — every q on Γ→X→U→L→Γ→K (and between) is available.
#
# One-off: build H_k for all k (n_params supercell Hessians, serialized).  Then
# precompute B(q) = [vec(D_k(q))]_k at a dense sweep of band-path q; everything
# after (detection, cuts, projection) is cheap matrix-vector.
#
# Repair = PROJECTION onto the nearest stable model (no data pin, always
# feasible, preserves dispersion → coverage):
#     min ½(θ̃−θ̃ᵢ)ᵀ C (θ̃−θ̃ᵢ)  s.t.  b′=0, Born, b″>0, band-path cuts
# Cutting-plane: detect soft (q,e) on the dense path, add exact cut rows,
# re-project, until every path mode clears the margin.  Verified with full
# phonon bands + a test-set parity/coverage study.
#
# Outputs → models/Al_20_4_6A_2_subset_50_percent/results/forest_bandpath/
#
# Run:  julia --project [-t N] scripts/uq/forest_bandpath_project_Al_20_4_6A_2_subset_50.jl

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Random, Serialization
using SparseArrays, StaticArrays, Unitful, ForwardDiff, CairoMakie, OSQP, StatsBase
using ACEpotentials, ACEWorkflow
using ACEpotentials: potential_energy
using AtomsBuilder
using AtomsCalculatorsUtilities.SitePotentials: hessian
using ExtXYZ, AtomsCalculators
import AtomsCalculators: forces
import ACEWorkflow: phonon_committee, precompute_force_constants, dynamical_matrix_from_fc, fcc_band_path

Random.seed!(1234)

element        = :Al
n_lev          = 5
n_res          = 10
n_rand         = 15
N_cell_fc      = 3      # force-constant supercell (≥ 2×cutoff); also band verification cell
N_per_seg      = 20     # band-path q per segment for detection (dense)
cut_margin_THz = 0.15   # require every path mode ≥ this (buffers inter-sample gaps)
max_cuts       = 60
test_stride    = 10

result     = load_model(element, 20, 4, 6, 2; dataset_name="subset_50_percent")
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)
P          = result.P
Ap = Diagonal(result.W) * result.A / P
Yw = result.W .* result.Y
λ  = 1.0 / size(Ap, 1)
outdir = "$(result.dir)/results/forest_bandpath"; mkpath(outdir)
println("Model $(result.name): $n_params params.  Outputs → $outdir")

# ── Leverage / residual / forest corrections ─────────────────────────────────
C   = Symmetric(Ap' * Ap .+ λ .* (P' * P)); Cf = cholesky(C)
AtX = Cf \ Matrix(Ap'); θ̃ = Cf \ (Ap' * Yw)
leverage = vec(sum(Ap' .* AtX; dims=1)); residual = Yw .- Ap * θ̃
forest_member(i) = lin_params .+ (P \ (AtX[:, i] .* (residual[i] / leverage[i])))

# ── Born / a_eq rows ─────────────────────────────────────────────────────────
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

# ── Per-basis dynamical-matrix basis on the band path (one-off, serialized) ──
sys_prim  = bulk(element; a=a_eq*u"Å")
sys_super = bulk(element; a=a_eq*u"Å", cubic=true) * (N_cell_fc, N_cell_fc, N_cell_fc)
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
fc0 = precompute_force_constants(sys_prim, sys_super, model)          # geometry mappings
Np  = fc0.Np

hk_file = "$(result.dir)/results/bandpath_Hk_$(N_cell_fc)x$(N_cell_fc)x$(N_cell_fc).jls"
H_all = if isfile(hk_file)
    cached = deserialize(hk_file)
    abs(cached.a_eq - a_eq) < 1e-4 || error("cached H_k at a=$(cached.a_eq) ≠ $a_eq")
    println("Loaded cached per-basis Hessians: $hk_file")
    cached.H_all
else
    @printf("Building %d per-basis supercell Hessians in parallel (%d threads) …\n",
            n_params, Threads.nthreads())
    Hs = Vector{Matrix{Float64}}(undef, n_params); t0 = time()
    models_t = [deepcopy(model) for _ in 1:Threads.nthreads()]
    blas_old = BLAS.get_num_threads(); BLAS.set_num_threads(1)   # avoid oversubscription
    done = Threads.Atomic{Int}(0)
    Threads.@threads :static for k in 1:n_params
        m = models_t[Threads.threadid()]
        e_k = zeros(n_params); e_k[k] = 1.0
        ACEpotentials.Models.set_linear_parameters!(m, e_k)
        Hs[k] = ustrip.(hessian(sys_super, m))
        d = Threads.atomic_add!(done, 1) + 1
        d % 10 == 0 && @printf("\r  %d / %d  (%.1f min)      ", d, n_params, (time()-t0)/60)
    end
    BLAS.set_num_threads(blas_old)
    println("\r  done in $(round((time()-t0)/60;digits=1)) min.                    ")
    ACEpotentials.Models.set_linear_parameters!(model, lin_params)
    serialize(hk_file, (H_all=Hs, a_eq=a_eq, N_cell=N_cell_fc))
    println("Serialized → $hk_file")
    Hs
end

# dense band-path q, drop near-Γ points (acoustic zeros)
q_list, x_vals, x_ticks, labels, _ = fcc_band_path(fc0.L; N_per_seg=N_per_seg)
keep = [norm(q) > 5e-2 for q in q_list]
qpath = q_list[keep]
println("Band path: $(length(qpath)) q-points (dropped $(count(!, keep)) near-Γ)")

# B(q)[:, k] = vec(D_k(q))  — 9 × n_params (primitive 3×3, Np=1)
println("Precomputing per-basis D_k(q) along the path …")
Bq = Vector{Matrix{ComplexF64}}(undef, length(qpath))
for (iq, q) in enumerate(qpath)
    M = Matrix{ComplexF64}(undef, (3Np)^2, n_params)
    for k in 1:n_params
        M[:, k] = vec(dynamical_matrix_from_fc(merge(fc0, (H = H_all[k],)), q))
    end
    Bq[iq] = M
end
ω2_cut = (cut_margin_THz / FREQ_THz)^2

# exact band-path spectrum for θ; returns (min freq THz, [(iq, eigenvector) soft modes])
function bandpath_soft(θ)
    fmin = Inf; soft = Tuple{Int,Vector{ComplexF64}}[]
    for iq in eachindex(qpath)
        D = Hermitian(reshape(Bq[iq] * θ, 3Np, 3Np))
        F = eigen(D)
        for ν in 1:3Np
            f = sign(F.values[ν]) * sqrt(abs(F.values[ν])) * FREQ_THz
            f < fmin && (fmin = f)
            F.values[ν] < ω2_cut && push!(soft, (iq, F.vectors[:, ν]))
        end
    end
    return fmin, soft
end

# exact linear cut row for mode e at path point iq: row·θ = Re(eᴴ D(q,θ) e)
function cut_row(iq, e)
    c = vec(conj(e) * transpose(e))            # c_{(a,b)} = conj(e_a) e_b
    return real(vec(transpose(c) * Bq[iq]))
end

# ── Projection QP: nearest stable model to θᵢ in the fit metric C ─────────────
Cmat = sparse(Matrix(C)); osqp = OSQP.Model()
function project(θi, rows, lowers)
    θ̃i = P * θi
    phys = sparse(vcat(b_prime', rows) / P)
    l = vcat([0.0], lowers); u = vcat([0.0], fill(Inf, length(lowers)))
    OSQP.setup!(osqp; P=Cmat, q=Vector(-(Cmat * θ̃i)), A=phys, l=l, u=u,
                max_iter=4_000_000, check_termination=25, verbose=false, eps_abs=1e-6, eps_rel=1e-6)
    r = OSQP.solve!(osqp); return P \ r.x, r.info.status
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

# ── Cutting-plane projection along the band path ─────────────────────────────
committee = Vector{Vector{Float64}}(undef, length(selected)); n_cuts = zeros(Int, length(selected))
for (k, i) in enumerate(selected)
    θi = forest_member(i); θ = θi
    rows = copy(base_rows); lowers = copy(base_lower); local fmin
    for it in 0:max_cuts
        fmin, soft = bandpath_soft(θ)
        if isempty(soft)
            it > 0 && @printf("  obs %6d (%-8s): stable, %d cut(s), min ω=%+.3f THz\n", i, groups[k], n_cuts[k], fmin)
            break
        end
        it == max_cuts && (@warn "obs $i hit max_cuts, min ω=$(round(fmin;digits=3))"; break)
        for (iq, e) in soft; rows = vcat(rows, cut_row(iq, e)'); push!(lowers, ω2_cut); end
        n_cuts[k] += length(soft)
        θ, st = project(θi, rows, lowers)
        st == :Solved || @warn "obs $i projection status=$st at iter $it"
    end
    committee[k] = θ
end
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
@printf("\nRepaired (≥1 cut): %d / %d  [lev %d, res %d, rand %d]\n", count(>(0), n_cuts), length(selected),
        count(>(0), n_cuts[groups.=="leverage"]), count(>(0), n_cuts[groups.=="residual"]), count(>(0), n_cuts[groups.=="random"]))
writedlm("$outdir/committee.csv", reduce(hcat, committee)', ',')

# ── Full-band verification ───────────────────────────────────────────────────
println("\n── Full-band verification (phonon_committee) ───────────────")
x_v, all_freqs, x_t, labs = phonon_committee(model, committee, result, element; N_cell=N_cell_fc, file_prefix="forest_bandpath/")
min_f = [minimum(all_freqs[i+1]) for i in 1:length(committee)]; n_imag = count(<(-0.05), min_f)
for (k, i) in enumerate(selected)
    C11, C12, C44 = born(committee[k])
    @printf("  %-8s obs %6d cuts=%3d ‖δθ‖=%.3f C44=%6.1f minω=%+8.4f %s\n",
            groups[k], i, n_cuts[k], norm(committee[k].-lin_params), C44, min_f[k], min_f[k]<-0.05 ? "✗" : "✓")
end
@printf("\n  → %d / %d committee members phonon-UNSTABLE (full band path)\n", n_imag, length(committee))

open("$outdir/members.csv", "w") do io
    println(io, "group,obs,n_cuts,dtheta_norm,C11_GPa,C12_GPa,C44_GPa,min_freq_THz")
    for (k, i) in enumerate(selected)
        C11, C12, C44 = born(committee[k])
        @printf(io, "%s,%d,%d,%.5f,%.3f,%.3f,%.3f,%.4f\n", groups[k], i, n_cuts[k], norm(committee[k].-lin_params), C11, C12, C44, min_f[k])
    end
end

fig = Figure(size=(900,540)); ax = Axis(fig[1,1]; xlabel="Wave vector", ylabel="Frequency (THz)",
    title="$(result.name) — forest committee, band-path projection", xticks=(x_t, labs), xgridvisible=false)
for k in 1:length(committee)
    col = min_f[k]<-0.05 ? RGBAf(0.8,0.15,0.15,0.55) : n_cuts[k]>0 ? RGBAf(0.1,0.55,0.2,0.55) : RGBAf(0.5,0.5,0.5,0.35)
    for b in 1:size(all_freqs[k+1],1); lines!(ax, x_v, all_freqs[k+1][b,:]; color=col, linewidth=1.0); end
end
for b in 1:size(all_freqs[1],1); lines!(ax, x_v, all_freqs[1][b,:]; color=RGBAf(0,0.3,0.7,0.95), linewidth=2.0); end
hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8); vlines!(ax, x_t; color=(:black,0.3), linewidth=0.8)
Label(fig[0,:], "$(count(>(0),n_cuts)) repaired, $n_imag/$(length(committee)) unstable after"; fontsize=13)
save("$outdir/bands.png", fig)

# ── Test-set parity + coverage ───────────────────────────────────────────────
println("\n── Test-set parity + coverage ──────────────────────────────")
ACEpotentials.Models.set_committee!(model, committee); ACEpotentials.Models.set_linear_parameters!(model, lin_params)
tc = ExtXYZ.load("data/Al/manual_df_test_Al.xyz")[1:test_stride:end]
pE=Float64[];tE=Float64[];loE=Float64[];hiE=Float64[];pF=Float64[];tF=Float64[];loF=Float64[];hiF=Float64[]
for cfg in tc
    E, co = @committee potential_energy(cfg, model)
    push!(pE, ustrip(E)); push!(loE, minimum(ustrip.(co))); push!(hiE, maximum(ustrip.(co))); push!(tE, ustrip(cfg[:dft_energy]))
    if haskey(cfg[1], :dft_forces)
        F, coF = @committee forces(cfg, model)
        append!(tF, reduce(vcat, ustrip.([at[:dft_forces] for at in cfg]))); append!(pF, reduce(vcat, ustrip.(F)))
        for i in eachindex(coF[1]); fi = reduce(hcat, ustrip(coF[k][i]) for k in eachindex(coF)); append!(loF, vec(minimum(fi;dims=2))); append!(hiF, vec(maximum(fi;dims=2))); end
    end
end
function parity(t,p,lo,hi,xl,yl,path;col=:steelblue)
    rmse=sqrt(mean((p.-t).^2)); f=Figure(size=(600,600)); ax=Axis(f[1,1];xlabel=xl,ylabel=yl,title="RMSE = $(round(rmse,sigdigits=3))")
    errorbars!(ax,t,p,p.-lo,hi.-p;whiskerwidth=6,color=(col,0.5)); scatter!(ax,t,p;color=col,markersize=6)
    l=extrema([t;p]); lines!(ax,collect(l),collect(l);color=:black,linestyle=:dash); save(path,f); rmse
end
@printf("  energy RMSE = %.4g eV\n", parity(tE,pE,loE,hiE,"DFT E (eV)","ACE E (eV)","$outdir/energy_parity.png"))
!isempty(tF) && @printf("  force  RMSE = %.4g eV/Å\n", parity(tF,pF,loF,hiF,"DFT F","ACE F","$outdir/force_parity.png";col=:tomato))
function coverage(t,p,lo,hi;label,path,nbins=60)
    t,p,lo,hi=Float64.(t),Float64.(p),Float64.(lo),Float64.(hi); err=t.-p; mae=mean(abs.(err)); ne=err./mae
    be=vcat((t.-lo)./mae,(t.-hi)./mae); ev=mean((t.<lo).|(t.>hi)); bias=mean(err)/mae
    lim=maximum(abs.(vcat(ne,be))); ed=range(-lim,lim;length=nbins+1)
    d1=(h=fit(Histogram,ne,ed).weights;max.(h./(sum(h)*step(ed)),1e-3)); d2=(h=fit(Histogram,be,ed).weights;max.(h./(sum(h)*step(ed)),1e-3))
    f=Figure(size=(560,520)); ax=Axis(f[2,1];xlabel="Error / MAE",ylabel="density",yscale=log10,
        title="$label — coverage $(round((1-ev)*100,digits=1))% (EV $(round(ev*100,digits=1))%, bias $(round(bias*100,digits=0))%)")
    l1=stairs!(ax,ed[1:end-1],d1;step=:post,color=:black,linewidth=2); l2=stairs!(ax,ed[1:end-1],d2;step=:post,color=:orange,linewidth=2)
    ylims!(ax,1e-3,maximum(vcat(d1,d2))*3); Legend(f[1,1],[l1,l2],["test error","committee bound"];orientation=:horizontal,tellwidth=false); save(path,f); (1-ev)*100
end
@printf("  energy coverage = %.1f%%\n", coverage(tE,pE,loE,hiE;label="Energy",path="$outdir/energy_error_vs_uncertainty.png"))
!isempty(tF) && @printf("  force  coverage = %.1f%%\n", coverage(tF,pF,loF,hiF;label="Force",path="$outdir/force_error_vs_uncertainty.png"))
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
println("\n══ RESULT: $n_imag/$(length(committee)) unstable, $(count(>(0),n_cuts)) repaired ══\nAll outputs → $outdir/")
