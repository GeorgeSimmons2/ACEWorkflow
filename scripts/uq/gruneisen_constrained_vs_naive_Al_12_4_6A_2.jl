# gruneisen_constrained_vs_naive_Al_12_4_6A_2.jl
#
# Mode and macroscopic Grüneisen parameters for the CONSTRAINED+REJECTION committee
# against the NAIVE POPS committee — i.e. does imposing a_eq + phonon positivity give
# physically sensible anharmonicity, where unconstrained POPS does not?
#
# Follows Togo & Tanaka (phonopy paper, main.pdf) exactly:
#
#   Eq (15)   γ_qj(V) = −(V / ω_qj) · ∂ω_qj/∂V
#   Eq (16)   γ_qj(V) = −(V / 2ω²_qj) · Σ_{αβ,jj'} e*^{αj}_qj (∂D^{αβ}_{jj'}/∂V) e^{βj'}_qj
#   macro     γ = Σ_qj γ_qj C_qj / C_V,   C_qj the mode heat capacity of Eq (9)
#
# Eq (16) is the one used: it is branch-safe, because it contracts ∂D/∂V with the
# eigenvector at the reference volume rather than differencing sorted frequencies,
# which mis-assigns modes wherever branches cross.  Eq (15) is ALSO computed, by finite
# difference of the sorted bands, purely as a cross-check — the two must agree away
# from crossings, and disagreement localises the crossings rather than indicating a bug.
# (phonopy itself is not installed here, so this internal pair is the available check.)
#
#   ∂D/∂V ≈ (D(a₊) − D(a₋)) / (V(a₊) − V(a₋)),   a∓ = a_eq(1∓ε),  V ∝ a³
#
# Committees compared (both from results/aeq_cheap_vs_expensive):
#   constrained : committee_A_cheap_phononreject.csv, centred on the constrained mean
#   naive       : committee_B_naive.csv,              centred on the RLS model
#
# Outputs → results/gruneisen_constrained_vs_naive/
#   gruneisen_mode_bandpath.pdf/png    mode γ along Γ–X–U–L–Γ–K, mean ± 1σ per committee
#   gruneisen_thermodynamic.pdf/png    macroscopic γ(T), mean ± 1σ per committee
#   gruneisen_crosscheck.pdf/png       Eq (16) vs Eq (15) finite difference
#   gruneisen_summary.csv              γ(300 K) per member, per committee
#
# Run:  julia --project -t 8 scripts/uq/gruneisen_constrained_vs_naive_Al_12_4_6A_2.jl

include(joinpath(@__DIR__, "..", "bandpath_phonon_uq", "lib.jl"))
@async while true; flush(stdout); sleep(5); end

element, dataset = :Al, ""
N_cell    = 4
N_per_seg = [20, 20, 20, 20, 60]
ε_vol     = 0.01                 # ±1% lattice constant → volume derivative
ω_cut     = 0.5                  # THz; mask acoustic/near-Γ where γ diverges
qΓtol     = 5e-2
T_range   = 50.0:25.0:900.0
kB_eV     = 8.617333262e-5       # eV/K

result = load_model(element, 12, 4, 6, 2; dataset_name=dataset)
model, lin = result.model, result.lin_params
RES    = "$(result.dir)/results"
SRC    = "$RES/aeq_cheap_vs_expensive"
outdir = "$RES/gruneisen_constrained_vs_naive"; mkpath(outdir)
θ_con  = vec(readdlm("$RES/bandpath_undotted_multivolume/theta_mean.csv", ','))

ACEpotentials.Models.set_linear_parameters!(model, lin)
a_eq = ACEWorkflow.relax_lattice_constant(model, element)
a_m, a_p = a_eq*(1-ε_vol), a_eq*(1+ε_vol)
V(a) = a^3                       # any volume convention cancels in γ
@printf("a_eq = %.5f Å;  volume derivative from a∓ = %.5f / %.5f (±%.0f%%)\n",
        a_eq, a_m, a_p, 100ε_vol); flush(stdout)

println("band-path D_k(q) at three volumes (cached where possible) …"); flush(stdout)
bpm = bandpath_Dk(result, model, element, a_m,  N_cell; N_per_seg=N_per_seg, tag="grun_m_")
bp0 = bandpath_Dk(result, model, element, a_eq, N_cell; N_per_seg=N_per_seg)
bpp = bandpath_Dk(result, model, element, a_p,  N_cell; N_per_seg=N_per_seg, tag="grun_p_")
nq, Np = length(bp0.Bq), bp0.Np
dV = V(a_p) - V(a_m)
@printf("  %d q-points, %d branches, dV = %.5f Å³\n\n", nq, 3Np, dV); flush(stdout)

# ── Eq (16): branch-safe Hellmann–Feynman mode γ, and Eq (15) as a cross-check ──
function mode_gamma(θ)
    γ  = fill(NaN, 3Np, nq); ω0 = fill(NaN, 3Np, nq); γfd = fill(NaN, 3Np, nq)
    for iq in 1:nq
        D0 = Hermitian(reshape(bp0.Bq[iq]*θ, 3Np, 3Np))
        Dm = reshape(bpm.Bq[iq]*θ, 3Np, 3Np)
        Dp = reshape(bpp.Bq[iq]*θ, 3Np, 3Np)
        dDdV = (Dp .- Dm) ./ dV
        F  = eigen(D0)
        # sorted frequencies at the three volumes, for the Eq (15) finite difference
        fm = sign.(eigvals(Hermitian(Dm))) .* sqrt.(abs.(eigvals(Hermitian(Dm)))) .* FREQ_THz
        fp = sign.(eigvals(Hermitian(Dp))) .* sqrt.(abs.(eigvals(Hermitian(Dp)))) .* FREQ_THz
        for ν in 1:3Np
            ω2 = F.values[ν]
            ω  = sign(ω2)*sqrt(abs(ω2))*FREQ_THz
            ω0[ν,iq] = ω
            (bp0.qnorm[iq] < qΓtol || ω < ω_cut) && continue
            e = F.vectors[:, ν]
            γ[ν,iq]   = -(V(a_eq)/(2ω2)) * real(e' * dDdV * e)          # Eq (16)
            γfd[ν,iq] = -(V(a_eq)/ω) * (fp[ν] - fm[ν]) / dV             # Eq (15)
        end
    end
    return γ, ω0, γfd
end

# ── macroscopic γ(T) = Σ γ_qj C_qj / Σ C_qj, C_qj from Eq (9) ────────────────
function gamma_thermo(γ, ω0, T)
    num = 0.0; den = 0.0
    for i in eachindex(γ)
        (isnan(γ[i]) || ω0[i] <= 0) && continue
        x = (ω0[i] * THz_to_meV / 1000) / (kB_eV * T)          # ħω / kBT
        x > 60 && continue                                      # frozen mode
        C = kB_eV * x^2 * exp(x) / (exp(x) - 1)^2               # Eq (9) mode heat capacity
        num += γ[i]*C; den += C
    end
    den == 0 ? NaN : num/den
end

committees = [
 ("constrained", "$SRC/committee_A_cheap_phononreject.csv", θ_con, RGBf(0.0,0.447,0.698)),
 ("naive",       "$SRC/committee_B_naive.csv",              lin,   RGBf(0.80,0.15,0.15)),
]

store = Dict{String,Any}(); rows = String[]
for (tag, csv, centre, col) in committees
    isfile(csv) || (@warn "missing $csv"; continue)
    Θ = readdlm(csv, ','); mem = [collect(Float64, Θ[k,:]) for k in 1:size(Θ,1)]
    @printf("── %s: %d members ──\n", tag, length(mem)); flush(stdout)
    G = [mode_gamma(θ) for θ in mem]
    γc, ωc, γfdc = mode_gamma(centre)
    γs = cat([g[1] for g in G]...; dims=3)
    μ  = [(v = filter(!isnan, γs[b,q,:]); isempty(v) ? NaN : mean(v)) for b in 1:3Np, q in 1:nq]
    σ  = [(v = filter(!isnan, γs[b,q,:]); length(v) < 2 ? NaN : std(v)) for b in 1:3Np, q in 1:nq]
    γT = [ [gamma_thermo(g[1], g[2], T) for T in T_range] for g in G ]
    γTc = [gamma_thermo(γc, ωc, T) for T in T_range]
    i300 = argmin(abs.(collect(T_range) .- 300.0))
    v300 = filter(!isnan, [g[i300] for g in γT])
    @printf("   γ(300 K): centre %.3f | members %.3f ± %.3f  (range %.3f – %.3f)\n",
            γTc[i300], mean(v300), (length(v300)>1 ? std(v300) : 0.0),
            minimum(v300), maximum(v300)); flush(stdout)
    ok = .!isnan.(γc) .& .!isnan.(γfdc)
    @printf("   Eq(16) vs Eq(15) on the centre: median |Δγ| = %.4f over %d modes\n\n",
            median(abs.(γc[ok] .- γfdc[ok])), count(ok)); flush(stdout)
    store[tag] = (; μ, σ, γT, γTc, γc, γfdc, col, ω=ωc, n=length(mem))
    for (k,g) in enumerate(γT)
        push!(rows, @sprintf("%s,%d,%.5f", tag, k, g[i300]))
    end
    push!(rows, @sprintf("%s,centre,%.5f", tag, γTc[i300]))
end

# ── plots ────────────────────────────────────────────────────────────────────
let fig = Figure(size=(560,340), figure_padding=(6,10,4,6))
    ax = Axis(fig[1,1]; xlabel="Wave vector", ylabel="mode Grüneisen γ",
              title="Mode Grüneisen along the band path (Eq. 16), committee mean ± 1σ",
              titlesize=10, xticks=(bp0.x_ticks, bp0.labels), xgridvisible=false,
              ygridvisible=false, xtickalign=1, ytickalign=1)
    for (tag, st) in store, b in 1:3Np
        m, s = st.μ[b,:], st.σ[b,:]
        good = .!isnan.(m)
        any(good) || continue
        band!(ax, bp0.x_vals[good], (m .- coalesce.(s, 0.0))[good], (m .+ coalesce.(s, 0.0))[good];
              color=(st.col, 0.18))
        lines!(ax, bp0.x_vals[good], m[good]; color=st.col, linewidth=1.2)
    end
    hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
    vlines!(ax, bp0.x_ticks; color=(:black,0.22), linewidth=0.6)
    elem = [LineElement(color=st.col) for (_, st) in store]
    Legend(fig[1,1], elem, collect(keys(store)); tellwidth=false, tellheight=false,
           halign=:left, valign=:top, margin=(8,8,8,8), framevisible=true, labelsize=8)
    save("$outdir/gruneisen_mode_bandpath.pdf", fig)
    save("$outdir/gruneisen_mode_bandpath.png", fig; px_per_unit=4)
end

let fig = Figure(size=(480,340), figure_padding=(6,10,4,6))
    ax = Axis(fig[1,1]; xlabel="Temperature (K)", ylabel="macroscopic Grüneisen γ",
              title="γ(T) = Σ γ_qj C_qj / C_V", titlesize=10,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1)
    Tv = collect(T_range)
    for (tag, st) in store
        M = reduce(hcat, st.γT)
        μ = [mean(filter(!isnan, M[i,:])) for i in axes(M,1)]
        s = [(v=filter(!isnan, M[i,:]); length(v)>1 ? std(v) : 0.0) for i in axes(M,1)]
        band!(ax, Tv, μ.-s, μ.+s; color=(st.col, 0.18))
        lines!(ax, Tv, μ; color=st.col, linewidth=1.6)
        lines!(ax, Tv, st.γTc; color=st.col, linewidth=1.0, linestyle=:dash)
    end
    hlines!(ax, [2.15]; color=(:black,0.55), linestyle=:dot, linewidth=1.0)
    text!(ax, 0.98, 0.06; text="experiment, Al: γ ≈ 2.1–2.2", space=:relative,
          align=(:right,:bottom), fontsize=8, color=(:black,0.7))
    elem = [LineElement(color=st.col) for (_, st) in store]
    Legend(fig[1,1], elem, collect(keys(store)); tellwidth=false, tellheight=false,
           halign=:left, valign=:top, margin=(8,8,8,8), framevisible=true, labelsize=8)
    save("$outdir/gruneisen_thermodynamic.pdf", fig)
    save("$outdir/gruneisen_thermodynamic.png", fig; px_per_unit=4)
end

let fig = Figure(size=(430,380), figure_padding=(6,10,4,6))
    ax = Axis(fig[1,1]; xlabel="γ from Eq. (16), Hellmann–Feynman",
              ylabel="γ from Eq. (15), finite difference",
              title="cross-check: branch-safe vs sorted-band derivative", titlesize=10,
              xgridvisible=false, ygridvisible=false, xtickalign=1, ytickalign=1, aspect=1)
    for (tag, st) in store
        ok = .!isnan.(st.γc) .& .!isnan.(st.γfdc)
        scatter!(ax, vec(st.γc[ok]), vec(st.γfdc[ok]); color=(st.col,0.5), markersize=4)
    end
    l = (-4.0, 6.0); lines!(ax, collect(l), collect(l); color=:black, linestyle=:dash, linewidth=0.9)
    xlims!(ax, l...); ylims!(ax, l...)
    save("$outdir/gruneisen_crosscheck.pdf", fig)
    save("$outdir/gruneisen_crosscheck.png", fig; px_per_unit=4)
end

open("$outdir/gruneisen_summary.csv", "w") do io
    println(io, "# mode gamma via phonopy Eq.(16); macroscopic gamma = sum(gamma*C)/sum(C), Eq.(9) heat capacity")
    println(io, "# a_eq = $a_eq, eps_vol = $ε_vol, N_cell = $N_cell, omega_cut = $ω_cut THz")
    println(io, "committee,member,gamma_300K")
    foreach(r -> println(io, r), rows)
end
println("\noutputs → $outdir/")
ACEpotentials.Models.set_linear_parameters!(model, lin)
