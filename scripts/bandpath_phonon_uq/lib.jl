# lib.jl — shared machinery for the band-path phonon-UQ study.
#
# Everything is built on the UNDOTTED per-basis Hessian H_k = ∂²B_k/∂r² (which
# depends only on geometry), so at a fixed lattice constant it is built ONCE and
# every parameter vector's phonon check / band structure is Σ_k θ_k D_k(q) —
# a matvec plus a tiny eigen per q.  Validated vs the native dotted hessian to
# ~1e-13 in ../uq/undotted_hessian_test.jl.
#
# Loaded by the study scripts in this directory:  include("lib.jl")

using LinearAlgebra, DelimitedFiles, Statistics, Printf, Serialization
using StaticArrays, Unitful, ForwardDiff, CairoMakie, StatsBase
using ACEpotentials, ACEWorkflow
using ACEpotentials: potential_energy
import ACEpotentials.Models: evaluate_basis
using AtomsBuilder
using ExtXYZ, AtomsCalculators
import AtomsCalculators: forces
import AtomsCalculatorsUtilities.SitePotentials: PairList, get_neighbours, cutoff_radius
import ACEWorkflow: precompute_force_constants, dynamical_matrix_from_fc, fcc_band_path

# ── Undotted per-basis Hessian (parallel over atoms) ─────────────────────────
function _site_basis_hessian(model, Rs, Zs, z0, ps, st)
    nR = length(Rs)
    x0 = collect(Float64, reinterpret(Float64, Rs))
    tovec(x) = [SVector{3,eltype(x)}(x[3i-2], x[3i-1], x[3i]) for i in 1:nR]
    Bfun(x) = evaluate_basis(model, tovec(x), Zs, z0, ps, st)
    gradflat(x) = vec(ForwardDiff.jacobian(Bfun, x))
    Hflat = ForwardDiff.jacobian(gradflat, x0)
    return reshape(Hflat, length(Bfun(x0)), 3nR, 3nR)
end

function undotted_hessian(sys, V)
    nlist = PairList(sys, cutoff_radius(V)); Nat = length(sys); D = 3; ps = V.ps; st = V.st; m = V.model
    _, Rs0, Zs0, z00 = get_neighbours(sys, V, nlist, 1)
    N_basis = length(evaluate_basis(m, Rs0, Zs0, z00, ps, st))
    Ht = [zeros(D*Nat, D*Nat, N_basis) for _ in 1:Threads.nthreads()]
    Threads.@threads :static for i in 1:Nat
        H = Ht[Threads.threadid()]
        Js, Rs, Zs, z0 = get_neighbours(sys, V, nlist, i)
        Hi = _site_basis_hessian(m, Rs, Zs, z0, ps, st)
        Ji = (i-1)*D .+ (1:D)
        for (α1, j1) in enumerate(Js), (α2, j2) in enumerate(Js)
            A1 = (α1-1)*D .+ (1:D); A2 = (α2-1)*D .+ (1:D)
            J1 = (j1-1)*D .+ (1:D); J2 = (j2-1)*D .+ (1:D)
            @views for k in 1:N_basis
                blk = Hi[k, A1, A2]
                H[J1,J2,k] .+= blk; H[J1,Ji,k] .-= blk; H[Ji,J2,k] .-= blk; H[Ji,Ji,k] .+= blk
            end
        end
    end
    return sum(Ht)
end

# get (or build+cache) the undotted per-basis Hessian at lattice constant `a`
function undotted_Hbasis(result, model, element, a, N_cell; tag="")
    f = "$(result.dir)/results/undotted_Hbasis_$(tag)$(N_cell)x$(N_cell)x$(N_cell)_a$(round(a;digits=5)).jls"
    # back-compat: the committee run cached without the a-suffix at the reference a
    f0 = "$(result.dir)/results/undotted_Hbasis_$(N_cell)x$(N_cell)x$(N_cell).jls"
    for cand in (f, f0)
        if isfile(cand)
            c = deserialize(cand)
            if abs(c.a_eq - a) < 1e-4; println("  loaded H_basis cache $(basename(cand))"); return c.H_basis; end
        end
    end
    sys = bulk(element; a=a*u"Å", cubic=true) * (N_cell, N_cell, N_cell)
    @printf("  building undotted H_basis at a=%.5f (%d atoms, %d threads) …\n", a, length(sys), Threads.nthreads())
    t = @elapsed Hb = undotted_hessian(sys, model)
    @printf("    done in %.1f min\n", t/60)
    serialize(f, (H_basis=Hb, a_eq=a, N_cell=N_cell)); return Hb
end

# band-path per-basis D_k(q) over the FULL path (Γ INCLUDED — for plotting AND checks)
function bandpath_Dk(result, model, element, a, N_cell; N_per_seg=20, tag="")
    sys_prim = bulk(element; a=a*u"Å"); sys_super = bulk(element; a=a*u"Å", cubic=true) * (N_cell, N_cell, N_cell)
    ACEpotentials.Models.set_linear_parameters!(model, result.lin_params)
    fc0 = precompute_force_constants(sys_prim, sys_super, model); Np = fc0.Np; N3 = 3*length(sys_super)
    Hb = undotted_Hbasis(result, model, element, a, N_cell; tag=tag)
    n_params = length(result.lin_params)
    q_list, x_vals, x_ticks, labels, _ = fcc_band_path(fc0.L; N_per_seg=N_per_seg)
    Bq = Vector{Matrix{ComplexF64}}(undef, length(q_list))
    for (iq, q) in enumerate(q_list)
        M = Matrix{ComplexF64}(undef, (3Np)^2, n_params)
        for k in 1:n_params
            M[:, k] = vec(dynamical_matrix_from_fc(merge(fc0, (H = reshape(Hb[:, :, k], N3, N3),)), q))
        end
        Bq[iq] = M
    end
    return (; Bq, q_list, x_vals, x_ticks, labels, Np, qnorm = norm.(q_list))
end

# frequencies (THz), 3Np × n_q, over the full path
function bands(θ, bp)
    F = Matrix{Float64}(undef, 3bp.Np, length(bp.Bq))
    for iq in eachindex(bp.Bq)
        ev = eigvals(Hermitian(reshape(bp.Bq[iq]*θ, 3bp.Np, 3bp.Np)))
        F[:, iq] = sign.(ev) .* sqrt.(abs.(ev)) .* FREQ_THz
    end
    return F
end

# minimum non-acoustic frequency (skips q near Γ where acoustic → 0) — the stability metric
function min_freq_stable(θ, bp; qΓtol=5e-2)
    m = Inf
    for iq in eachindex(bp.Bq)
        bp.qnorm[iq] < qΓtol && continue
        ev = eigvals(Hermitian(reshape(bp.Bq[iq]*θ, 3bp.Np, 3bp.Np)))
        m = min(m, minimum(sign.(ev).*sqrt.(abs.(ev)).*FREQ_THz))
    end
    return m
end

# soft (iq, eigenvector) modes below margin, skipping near-Γ — for cutting-plane
function soft_modes(θ, bp, ω2_cut; qΓtol=5e-2)
    soft = Tuple{Int,Vector{ComplexF64}}[]
    for iq in eachindex(bp.Bq)
        bp.qnorm[iq] < qΓtol && continue
        Fe = eigen(Hermitian(reshape(bp.Bq[iq]*θ, 3bp.Np, 3bp.Np)))
        for ν in 1:3bp.Np
            Fe.values[ν] < ω2_cut && push!(soft, (iq, Fe.vectors[:, ν]))
        end
    end
    return soft
end
cut_row(iq, e, bp) = (c = vec(conj(e)*transpose(e)); real(vec(transpose(c) * bp.Bq[iq])))

# ── FIXED band-structure plot: full path, acoustic branches reach 0 at Γ ─────
function plot_committee_bands(members, θ_mean, bp, title, path; unstable_tol=-0.05)
    fig = Figure(size=(860, 520))
    ax = Axis(fig[1,1]; xlabel="Wave vector", ylabel="Frequency (THz)", title=title,
              xticks=(bp.x_ticks, bp.labels), xgridvisible=false)
    for θ in members
        F = bands(θ, bp); unstable = min_freq_stable(θ, bp) < unstable_tol
        col = unstable ? RGBAf(0.8,0.15,0.15,0.4) : RGBAf(0.45,0.45,0.45,0.35)
        for b in 1:3bp.Np; lines!(ax, bp.x_vals, F[b,:]; color=col, linewidth=1.0); end
    end
    Fm = bands(θ_mean, bp)
    for b in 1:3bp.Np; lines!(ax, bp.x_vals, Fm[b,:]; color=RGBAf(0,0.3,0.7,0.95), linewidth=2.0); end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    vlines!(ax, bp.x_ticks; color=(:black,0.3), linewidth=0.8)
    save(path, fig); return fig
end

# ── Test-set committee predictions (energy + forces with min/max bounds) ─────
function committee_predictions(model, committee, test_xyz; stride=10, point_params=mean(committee))
    ACEpotentials.Models.set_committee!(model, committee)
    ACEpotentials.Models.set_linear_parameters!(model, point_params)   # point prediction = committee mean
    tc = ExtXYZ.load(test_xyz)[1:stride:end]
    pE=Float64[];tE=Float64[];loE=Float64[];hiE=Float64[]
    pF=Float64[];tF=Float64[];loF=Float64[];hiF=Float64[]
    for cfg in tc
        E, co = @committee potential_energy(cfg, model)
        push!(pE, ustrip(E)); push!(loE, minimum(ustrip.(co))); push!(hiE, maximum(ustrip.(co))); push!(tE, ustrip(cfg[:dft_energy]))
        if haskey(cfg[1], :dft_forces)
            F, coF = @committee forces(cfg, model)
            append!(tF, reduce(vcat, ustrip.([at[:dft_forces] for at in cfg]))); append!(pF, reduce(vcat, ustrip.(F)))
            for i in eachindex(coF[1]); fi = reduce(hcat, ustrip(coF[k][i]) for k in eachindex(coF)); append!(loF, vec(minimum(fi;dims=2))); append!(hiF, vec(maximum(fi;dims=2))); end
        end
    end
    return (; pE, tE, loE, hiE, pF, tF, loF, hiF, n=length(tc))
end

# parity plot with committee error bars
function parity_plot(t, p, lo, hi, xl, yl, path; col=:steelblue)
    rmse = sqrt(mean((p .- t).^2)); f = Figure(size=(560,560))
    ax = Axis(f[1,1]; xlabel=xl, ylabel=yl, title="RMSE = $(round(rmse,sigdigits=3))")
    errorbars!(ax, t, p, p.-lo, hi.-p; whiskerwidth=5, color=(col,0.45)); scatter!(ax, t, p; color=col, markersize=5)
    l = extrema([t;p]); lines!(ax, collect(l), collect(l); color=:black, linestyle=:dash)
    save(path, f); return rmse
end

# error-vs-committee-bound calibration histogram → coverage, bias, error-violation
function calibration_hist(t, p, lo, hi; label, path, nbins=60)
    t,p,lo,hi = Float64.(t),Float64.(p),Float64.(lo),Float64.(hi)
    err = t.-p; mae = mean(abs.(err)); ne = err./mae
    be = vcat((t.-lo)./mae, (t.-hi)./mae); ev = mean((t.<lo).|(t.>hi)); bias = mean(err)/mae
    lim = maximum(abs.(vcat(ne,be))); ed = range(-lim,lim;length=nbins+1)
    d1=(h=fit(Histogram,ne,ed).weights; max.(h./(sum(h)*step(ed)),1e-3))
    d2=(h=fit(Histogram,be,ed).weights; max.(h./(sum(h)*step(ed)),1e-3))
    f=Figure(size=(600,540))
    ax=Axis(f[2,1]; xlabel="(true − pred) / MAE", ylabel="density", yscale=log10,
            title="$label — coverage $(round((1-ev)*100,digits=1))%  (bias $(round(bias*100,digits=0))% MAE)")
    l1=stairs!(ax,ed[1:end-1],d1;step=:post,color=:black,linewidth=2)
    l2=stairs!(ax,ed[1:end-1],d2;step=:post,color=:orange,linewidth=2)
    ylims!(ax,1e-3,maximum(vcat(d1,d2))*3)
    Legend(f[1,1],[l1,l2],["test error","committee bound"];orientation=:horizontal,tellwidth=false,framevisible=true)
    save(path,f); return (coverage=(1-ev)*100, bias=bias*100, rmse=sqrt(mean(err.^2)))
end
