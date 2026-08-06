# verify_claims_Al_12_4_6A_2.jl
#
# Independently recompute every headline claim from the raw artifacts on disk and
# print CLAIMED vs RECOMPUTED side by side, with PASS/FAIL.  Nothing here trusts a
# summary line; each number is rebuilt from the trajectory, the committee CSV or the
# model, and compared against what was reported.
#
# The `claimed` values are hard-coded below exactly as they were stated, so if a
# reported number was wrong this script says FAIL rather than quietly agreeing.
#
# What is verified:
#   1  provenance — which theta the NPT run used (byte identity against the committee)
#   2  a(T) recomputed from the trajectory cell volumes, not read from the summary
#   3  mean coordination and median NN recomputed from atomic positions
#   4  phonon min omega recomputed from theta at each a(T)
#   5  alpha refitted from the RECOMPUTED a(T), FCC points only
#   6  committee stability: min omega for all 30 members at all 6 volumes
#   7  the naive-worst comparison, and that it is genuinely unconstrained
#   8  cross-check: fresh native Hessian vs the cached undotted basis
#
# Run:  julia --project -t 8 scripts/uq/verify_claims_Al_12_4_6A_2.jl
#       (~5 min; all required Hessians are already cached)

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))

element = :Al; N_cell = 4; N_per_seg = [20,20,20,20,60]; qGtol = 5e-2
result = load_model(element, 12, 4, 6, 2; dataset_name="")
model, lin = result.model, result.lin_params
R = "$(result.dir)/results"
MV   = "$R/bandpath_undotted_multivolume"
NPT  = "$R/npt_multivolume_softest"
NAIV = "$R/npt_thermal_expansion_naive_worst_member"

np = Ref(0); nf = Ref(0)
function check(label, claimed, got; tol=5e-3)
    ok = abs(got - claimed) <= tol
    ok ? (np[] += 1) : (nf[] += 1)
    @printf("  %-44s claimed %11.5g  got %11.5g   %s\n",
            label, claimed, got, ok ? "PASS" : "*** FAIL ***")
end
function checkbool(label, got; expect=true)
    ok = got == expect
    ok ? (np[] += 1) : (nf[] += 1)
    @printf("  %-44s %-11s  %-11s   %s\n", label, string(expect), string(got), ok ? "PASS" : "*** FAIL ***")
end

# ── independent trajectory analysis (own code, not the MD script's) ──────────
function traj_stats(path; n_equil=10_000, rc=3.3)
    fr = ExtXYZ.read_frames(path)
    keep = [f for f in fr if Int(f["info"]["step"]) >= n_equil]
    Ls = [f["cell"][1,1] for f in keep]
    a  = mean(Ls)/4                                  # 4x4x4 supercell
    Zs = Float64[]; NNs = Float64[]
    for f in keep
        p = Matrix{Float64}(f["arrays"]["pos"]); L = f["cell"][1,1]; n = size(p,2)
        z = zeros(Int,n); nn = fill(Inf,n)
        @inbounds for i in 1:n-1, j in i+1:n
            d1=p[1,i]-p[1,j]; d2=p[2,i]-p[2,j]; d3=p[3,i]-p[3,j]
            d1-=L*round(d1/L); d2-=L*round(d2/L); d3-=L*round(d3/L)
            r=sqrt(d1*d1+d2*d2+d3*d3)
            r<rc && (z[i]+=1; z[j]+=1)
            r<nn[i] && (nn[i]=r); r<nn[j] && (nn[j]=r)
        end
        push!(Zs, mean(z)); push!(NNs, median(nn))
    end
    (; a, Z=mean(Zs), nn=median(NNs), nframes=length(keep))
end
native_minomega(θ, a) = begin
    ACEpotentials.Models.set_linear_parameters!(model, θ)
    sp, ss = bulk_prim_super(element; a=a, N_cell=N_cell)
    fc = precompute_force_constants(sp, ss, model)
    ql, _, _, _, _ = _band_path(AtomsBuilder.Chemistry.symmetry(element), fc.L; N_per_seg=N_per_seg)
    qn = norm.(ql); m = Inf
    for (iq,q) in enumerate(ql)
        qn[iq] < qGtol && continue
        ev = eigvals(Hermitian(dynamical_matrix_from_fc(fc, q)))
        m = min(m, minimum(sign.(ev).*sqrt.(abs.(ev)).*FREQ_THz))
    end
    m
end

println("="^96); println("1. PROVENANCE"); println("="^96)
θ_npt  = vec(readdlm("$NPT/theta_used.csv", ','))
Rrej   = readdlm("$MV/committee_rejection.csv", ',')
diffs  = [maximum(abs.(Rrej[k,:] .- θ_npt)) for k in 1:size(Rrej,1)]
k, dmin = argmin(diffs), minimum(diffs)
@printf("  theta_used matches committee_rejection row %d, max|diff| = %.3e\n", k, dmin)
checkbool("NPT theta is rejection member 18", k == 18)
checkbool("  ... and bit-identical", dmin == 0.0)

println(); println("="^96); println("2-4. NPT RUN: a(T), COORDINATION, PHONONS  (recomputed from trajectories)"); println("="^96)
claim = Dict(300=>(a=4.077861, Z=11.9900, w=0.3021),
             500=>(a=4.099550, Z=11.8471, w=0.2896),
             700=>(a=4.120435, Z=11.5121, w=0.2801),
             900=>(a=4.312109, Z= 9.1187, w=0.2078))
aT = Dict{Int,Float64}(); Zt = Dict{Int,Float64}()
for T in (300,500,700,900)
    s = traj_stats("$NPT/T$(T)K/md_trajectory.extxyz")
    w = native_minomega(θ_npt, s.a)
    aT[T] = s.a; Zt[T] = s.Z
    @printf("\n  T = %d K   (%d production frames)\n", T, s.nframes)
    check("  a(T) from cell volumes (Å)", claim[T].a, s.a; tol=2e-3)
    check("  mean coordination <Z>",      claim[T].Z, s.Z; tol=5e-2)
    check("  min non-acoustic ω (THz)",   claim[T].w, w;   tol=5e-2)
    checkbool("  FCC verdict (<Z> >= 11)", s.Z >= 11.0; expect = T != 900)
end

println(); println("="^96); println("5. THERMAL EXPANSION  (refitted from the RECOMPUTED a(T))"); println("="^96)
a0 = 4.044936
fccT = [T for T in (300,500,700,900) if Zt[T] >= 11.0]
X = hcat(ones(length(fccT)), Float64.(fccT))
α = (X \ [aT[T] for T in fccT])[2] / a0 * 1e6
@printf("  FCC temperatures used: %s\n", string(fccT))
check("  α on FCC points only (1e-6/K)", 26.31, α; tol=0.3)
αall = (hcat(ones(4), Float64[300,500,700,900]) \ [aT[T] for T in (300,500,700,900)])[2]/a0*1e6
@printf("  (for contrast, fitting ALL four points gives %.2f — the number the FCC filter avoids)\n", αall)
w900 = native_minomega(θ_npt, aT[900])
checkbool("  900 K lattice inside constrained range", aT[900] <= 1.10*a0)
checkbool("  900 K phonons POSITIVE (so not harmonic)", w900 > 0)
@printf("    a(900K)/a_eq = %.4f,  min ω there = %+.3f THz\n", aT[900]/a0, w900)

println(); println("="^96); println("6. COMMITTEE STABILITY  (all 30 members × 6 volumes)"); println("="^96)
vol = collect(1.00:0.02:1.10); a_eq = ACEWorkflow.relax_lattice_constant(model, element)
bps = [bandpath_Dk(result, model, element, a_eq*s, N_cell; N_per_seg=N_per_seg) for s in vol]
Rrep = readdlm("$MV/committee_repaired.csv", ',')
Mrej = [minimum(min_freq_stable(Rrej[k,:], bp) for bp in bps) for k in 1:size(Rrej,1)]
@printf("  rejection committee worst-over-volumes: min %+.4f, max %+.4f\n", minimum(Mrej), maximum(Mrej))
check("  softest rejection member (THz)", 0.1503, minimum(Mrej); tol=5e-3)
checkbool("  0 / 30 unstable at any volume", count(<(-0.05), Mrej) == 0)
per_vol = [minimum(min_freq_stable(Rrej[k,:], bps[v]) for k in 1:size(Rrej,1)) for v in eachindex(vol)]
@printf("  per-volume minima: %s\n", string(round.(per_vol; digits=3)))
for (v,c) in zip(1:6, [0.252,0.175,0.150,0.166,0.150,0.153])
    check(@sprintf("  volume %.0f%% minimum (THz)", 100vol[v]), c, per_vol[v]; tol=5e-3)
end

println(); println("="^96); println("7. NAIVE COMPARISON"); println("="^96)
θ_nai = vec(readdlm("$NAIV/theta_naive_worst.csv", ','))
sn = traj_stats("$NAIV/T300K/md_trajectory.extxyz")
wn = native_minomega(θ_nai, sn.a)
check("  naive a(300K) (Å)", 4.166635, sn.a; tol=2e-3)
check("  naive <Z> at 300 K", 9.19, sn.Z; tol=5e-2)
check("  naive min ω at a(300K) (THz)", -11.3676, wn; tol=5e-2)
checkbool("  naive LEFT FCC (<Z> < 11)", sn.Z < 11.0)
# is it genuinely unconstrained?
lb(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
            ACEWorkflow.Elasticity.reference_system(element; a=a), model))
ACEpotentials.Models.set_linear_parameters!(model, lin)
b1 = ForwardDiff.derivative(lb, a_eq)
@printf("  b′·θ  naive = %.3e   constrained = %.3e\n", dot(b1,θ_nai), dot(b1,θ_npt))
checkbool("  naive is UNCONSTRAINED (|b′·θ| > 1e-4)", abs(dot(b1,θ_nai)) > 1e-4)
checkbool("  constrained IS pinned  (|b′·θ| < 1e-6)", abs(dot(b1,θ_npt)) < 1e-6)

println(); println("="^96); println("8. CROSS-CHECK: fresh native Hessian vs cached undotted basis"); println("="^96)
bp300 = bandpath_Dk(result, model, element, aT[300], N_cell; N_per_seg=N_per_seg)
w_und = min_freq_stable(θ_npt, bp300); w_nat = native_minomega(θ_npt, aT[300])
@printf("  native %.6f THz   undotted %.6f THz   Δ = %.2e\n", w_nat, w_und, abs(w_nat-w_und))
checkbool("  two independent Hessian routes agree < 1e-6", abs(w_nat-w_und) < 1e-6)

println(); println("="^96)
@printf("  %d checks PASSED, %d FAILED\n", np[], nf[])
println(nf[] == 0 ? "  All reported numbers reproduce from the raw artifacts." :
                    "  *** DISCREPANCIES ABOVE — do not use the failing numbers. ***")
println("="^96)
ACEpotentials.Models.set_linear_parameters!(model, lin)
