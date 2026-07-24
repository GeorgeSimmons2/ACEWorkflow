# gruneisen.jl
#
# Grüneisen-parameter study of the constrained committee, using the undotted
# Hessian at three volumes.  Mode Grüneisen:
#
#     γ_qν = −dlnω_qν/dlnV = −(V0/ω0)·(ω₊ − ω₋)/(V₊ − V₋)
#
# from phonons at a₋ = a_eq(1−ε), a_eq, a₊ = a_eq(1+ε) (V ∝ a³).  Produces:
#   • mode γ along the band path  (committee mean ± 1σ)
#   • thermodynamic bulk γ(T)     (heat-capacity weighted; committee mean ± 1σ)
#
# The reference-volume undotted Hessian is reused from cache; the two strained
# volumes are built once (undotted, parallel) and cached.
#
# Run:  julia --project -t <N> scripts/bandpath_phonon_uq/gruneisen.jl

include(joinpath(@__DIR__, "lib.jl"))

element = :Al
dataset = "subset_50_percent"
N_cell  = 3
ε_vol   = 0.01                    # ±1% lattice-constant strain for the volume derivative
ω_cut   = 0.5                     # THz — mask modes below this (acoustic/near-Γ divergence)
T_range = 50.0:25.0:800.0
kB_eV   = 8.617333262e-5          # eV/K

result = load_model(element, 20, 4, 6, 2; dataset_name=dataset)
model  = result.model; lin_params = result.lin_params
cdir   = "$(result.dir)/results/bandpath_undotted"
outdir = "$(result.dir)/results/bandpath_phonon_uq"; mkpath(outdir)
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
a_m, a_p = a_eq*(1-ε_vol), a_eq*(1+ε_vol)
V(a) = a^3                        # volume convention cancels in γ; a³ is fine
@printf("Grüneisen at a = {%.5f, %.5f, %.5f} Å (±%.0f%%)\n", a_m, a_eq, a_p, 100ε_vol)

println("Band-path D_k(q) at three volumes (undotted, cached where possible) …")
bpm = bandpath_Dk(result, model, element, a_m,  N_cell; N_per_seg=20, tag="grun_m_")
bp0 = bandpath_Dk(result, model, element, a_eq, N_cell; N_per_seg=20)
bpp = bandpath_Dk(result, model, element, a_p,  N_cell; N_per_seg=20, tag="grun_p_")

committee = [readdlm("$cdir/committee_rejection.csv", ',')[i,:] for i in 1:30]
θ_mean = isfile("$cdir/theta_mean.csv") ? vec(readdlm("$cdir/theta_mean.csv", ',')) :
                                          vec(mean(readdlm("$cdir/committee_repaired.csv", ','); dims=1))

# naive POPS committee (raw forest members, same 5-leverage+10-residual+15-random draw) for comparison
import Random; Random.seed!(1234)
P=result.P; Ap=Diagonal(result.W)*result.A/P; Yw=result.W.*result.Y; λ=1/size(Ap,1)
Cf=cholesky(Symmetric(Ap'Ap.+λ.*(P'P))); AtX=Cf\Matrix(Ap'); θ̃v=Cf\(Ap'Yw)
lev=vec(sum(Ap'.*AtX;dims=1)); rres=Yw.-Ap*θ̃v; fmi(i)=lin_params.+(P\(AtX[:,i].*(rres[i]/lev[i])))
li=sortperm(lev;rev=true)[1:5]; ri=Int[]; for i in sortperm(abs.(rres);rev=true); i in li&&continue; push!(ri,i); length(ri)==10&&break; end
tk=Set(vcat(li,ri)); rd=Int[]; while length(rd)<15; i=Random.rand(1:length(Yw)); (i in tk)&&continue; push!(rd,i); push!(tk,i); end
naive = [fmi(i) for i in vcat(li,ri,rd)]

# mode Grüneisen via Hellmann–Feynman (branch-safe): γ = −(V0/2ω²)·⟨e|dD/dV|e⟩,
# with dD/dV from the ±volume dynamical matrices and e,ω² the V0 eigenpair.
# (Finite-differencing sorted frequencies mismatches near-degenerate branches.)
function mode_gamma(θ)
    nq = length(bp0.Bq); Np = bp0.Np
    γ = fill(NaN, 3Np, nq); ω0 = fill(NaN, 3Np, nq)
    dV = V(a_p) - V(a_m)
    for iq in 1:nq
        D0   = Hermitian(reshape(bp0.Bq[iq]*θ, 3Np, 3Np))
        dDdV = (reshape(bpp.Bq[iq]*θ, 3Np, 3Np) .- reshape(bpm.Bq[iq]*θ, 3Np, 3Np)) ./ dV
        F = eigen(D0)
        for ν in 1:3Np
            ω2 = F.values[ν]; ω0[ν,iq] = sign(ω2)*sqrt(abs(ω2))*FREQ_THz
            ω0[ν,iq] < ω_cut && continue               # mask acoustic/soft AND imaginary modes
            e = F.vectors[:, ν]
            γ[ν,iq] = -(V(a_eq)/(2ω2)) * real(e' * dDdV * e)
        end
    end
    return γ, ω0
end

# ── (1) mode γ along the band path: committee mean ± σ ───────────────────────
println("Computing mode Grüneisen along the band path …")
γ_all = [first(mode_gamma(θ)) for θ in committee]          # each 3Np × n_q
γ_mean_model, _ = mode_gamma(θ_mean)
# committee statistics per (ν, iq), ignoring NaN
γ_stack = cat(γ_all...; dims=3)                             # 3Np × n_q × N
γμ = [ (v = filter(!isnan, γ_stack[b,q,:]); isempty(v) ? NaN : mean(v)) for b in axes(γ_stack,1), q in axes(γ_stack,2)]
γσ = [ (v = filter(!isnan, γ_stack[b,q,:]); length(v)<2 ? NaN : std(v)) for b in axes(γ_stack,1), q in axes(γ_stack,2)]

let fig = Figure(size=(880,520)), Np = bp0.Np
    ax = Axis(fig[1,1]; xlabel="Wave vector", ylabel="Mode Grüneisen γ",
              title="$(result.name) — mode Grüneisen (committee mean ± 1σ)",
              xticks=(bp0.x_ticks, bp0.labels), xgridvisible=false)
    for b in 1:3Np
        x = bp0.x_vals; m = γμ[b,:]; s = γσ[b,:]; ok = .!isnan.(m)
        band!(ax, x[ok], (m.-s)[ok], (m.+s)[ok]; color=RGBAf(0.15,0.4,0.75,0.18))
        lines!(ax, x[ok], m[ok]; color=RGBAf(0.15,0.4,0.75,0.9), linewidth=1.3)
    end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.7)
    vlines!(ax, bp0.x_ticks; color=(:black,0.3), linewidth=0.8)
    save("$outdir/gruneisen_bandpath.png", fig)
end
@printf("  mode γ (masked ω>%.1f THz): committee-mean ∈ [%.2f, %.2f]\n",
        ω_cut, minimum(filter(!isnan, γμ)), maximum(filter(!isnan, γμ)))

# ── (2) thermodynamic bulk γ(T): heat-capacity weighted, committee mean ± σ ──
function bulk_gamma_T(θ, T)
    γ, ω0 = mode_gamma(θ)
    num = 0.0; den = 0.0
    for i in eachindex(ω0)
        isnan(γ[i]) && continue
        x = (THz_to_meV/1000 * ω0[i]) / (kB_eV * T)         # ħω/kT  (ħω[eV] = 4.136e-3·ν_THz)
        Cv = x^2 * exp(x) / (expm1(x))^2
        num += Cv*γ[i]; den += Cv
    end
    return den > 0 ? num/den : NaN
end
println("Thermodynamic bulk Grüneisen γ(T) …")
Ts = collect(T_range)
γT = [ [bulk_gamma_T(θ, T) for T in Ts] for θ in committee ]
γT_stack = reduce(hcat, γT)                                 # n_T × N
γTμ = vec(mean(γT_stack; dims=2)); γTσ = vec(std(γT_stack; dims=2))
γT_mean_model = [bulk_gamma_T(θ_mean, T) for T in Ts]

let fig = Figure(size=(620,430))
    ax = Axis(fig[1,1]; xlabel="Temperature (K)", ylabel="Bulk Grüneisen γ(T)",
              title="$(result.name) — thermodynamic Grüneisen (committee mean ± 1σ)")
    band!(ax, Ts, γTμ.-γTσ, γTμ.+γTσ; color=RGBAf(0.15,0.4,0.75,0.25))
    scatterlines!(ax, Ts, γTμ; color=RGBAf(0.15,0.4,0.75,0.95), linewidth=2.5, markersize=6, label="committee mean")
    lines!(ax, Ts, γT_mean_model; color=RGBAf(0.85,0.4,0.05,0.9), linewidth=2.0, linestyle=:dash, label="mean model")
    axislegend(ax; position=:rt)
    save("$outdir/gruneisen_vs_T.png", fig)
end
writedlm("$outdir/gruneisen_vs_T.csv", hcat(Ts, γTμ, γTσ, γT_mean_model), ',')
i300 = argmin(abs.(Ts.-300))
@printf("  constrained γ(300K) = %.3f ± %.3f   γ(%.0fK) = %.3f ± %.3f\n",
        γTμ[i300], γTσ[i300], Ts[end], γTμ[end], γTσ[end])

# ── (3) naive POPS comparison (imaginary modes excluded from the average) ────
println("Naive POPS γ(T) for comparison …")
γTn = reduce(hcat, [[bulk_gamma_T(θ, T) for T in Ts] for θ in naive])
nμ = [ (v=filter(isfinite, γTn[t,:]); isempty(v) ? NaN : mean(v)) for t in eachindex(Ts) ]
nσ = [ (v=filter(isfinite, γTn[t,:]); length(v)<2 ? NaN : std(v)) for t in eachindex(Ts) ]
let fig=Figure(size=(660,450))
    ax=Axis(fig[1,1]; xlabel="Temperature (K)", ylabel="Bulk Grüneisen γ(T)",
            title="$(result.name) — Grüneisen: constrained committee vs naive POPS")
    band!(ax, Ts, nμ.-nσ, nμ.+nσ; color=RGBAf(0.80,0.20,0.20,0.16))
    scatterlines!(ax, Ts, nμ; color=RGBAf(0.80,0.20,0.20,0.9), linewidth=2, markersize=5, label="naive POPS")
    band!(ax, Ts, γTμ.-γTσ, γTμ.+γTσ; color=RGBAf(0.15,0.4,0.75,0.28))
    scatterlines!(ax, Ts, γTμ; color=RGBAf(0.15,0.4,0.75,0.95), linewidth=2.5, markersize=6, label="constrained committee")
    axislegend(ax; position=:rt)
    save("$outdir/gruneisen_vs_T_comparison.png", fig)
end
@printf("  naive γ(300K)       = %.3f ± %.3f   (unphysical spread from residual imaginary modes)\n",
        nμ[i300], nσ[i300])
ACEpotentials.Models.set_linear_parameters!(model, lin_params)
println("\nSaved: gruneisen_bandpath.png, gruneisen_vs_T.png, gruneisen_vs_T_comparison.png, gruneisen_vs_T.csv → $outdir/")
