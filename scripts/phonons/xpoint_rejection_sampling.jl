# xpoint_rejection_sampling.jl
#
# Rejection sampling of the POPS hypercube conditioned on the FCC X-point
# phonon branch ordering:   ω²_L  >  ω²_T1  and  ω²_L  >  ω²_T2
#
# Each proposed sample θ_i is accepted only if, after relaxing the lattice
# constant and computing the AD curvature operator, the longitudinal mode
# remains above both transverse modes (no L–T crossing at X).
#
# Setup is shared with x_point_curvature.jl.
#
# Usage:
#   julia --project scripts/phonons/xpoint_rejection_sampling.jl

using LinearAlgebra, StaticArrays, Printf, ForwardDiff, Dates, DelimitedFiles, Random
using AtomsBuilder, AtomsCalculators, Unitful, AtomsBase
using AtomsCalculatorsUtilities.SitePotentials: hessian
using ACEWorkflow, ACEpotentials

Random.seed!(1234)

# ─────────────────────────────────────────────────────────────────────────────
#  Physical constants
# ─────────────────────────────────────────────────────────────────────────────
const _eV_J    = 1.602176634e-19
const _Å_m     = 1.0e-10
const _amu_kg  = 1.66053906660e-27
const FREQ_THz = sqrt(_eV_J / (_Å_m^2 * _amu_kg)) / (2π * 1e12)   # ≈ 15.633

# ─────────────────────────────────────────────────────────────────────────────
#  Load model + POPS setup
# ─────────────────────────────────────────────────────────────────────────────
println("Loading model Al_12_4_6A_3 …")
result = load_model(:Al, 12, 4, 6, 3)
model  = result.model
θ      = result.lin_params

pops_corrections     = readdlm("$(result.dir)/pops_corrections.csv", ',')
hypercube_eigenvectors, hypercube_bounds =
    hypercube(pops_corrections)   # from ACEWorkflow re-export of POPSRegression

lower_hc = hypercube_bounds[1, :]
upper_hc = hypercube_bounds[2, :]

@printf("  N_params   = %d\n", length(θ))
@printf("  Hypercube  : %d active directions\n", size(hypercube_eigenvectors, 2))

# ─────────────────────────────────────────────────────────────────────────────
#  Mean-model geometry  (for X-point eigenvector directions)
# ─────────────────────────────────────────────────────────────────────────────
println("\nRelaxing mean-model lattice constant …")
a0 = ACEWorkflow.relax_lattice_constant(model, :Al)
@printf("  a₀ = %.6f Å\n", a0)

sys_prim0 = bulk(:Al; a=a0 * u"Å")
L_prim0   = SMatrix{3,3,Float64}(
    ustrip.(hcat(sys_prim0.cell.cell_vectors...)))

# Reciprocal basis and X-point wavevector (Cartesian) from mean model
B_rec0 = 2π * inv(transpose(L_prim0))
q_X0   = B_rec0 * [0.0, 0.5, 0.5]

# Get mean-model X-point eigenvectors (used as fixed displacement directions;
# the Bloch phases at X are ±1 regardless of a, so vecs_X0 remains valid)
function get_xpoint_eigenvectors(model, a)
    sys_p  = bulk(:Al; a=a * u"Å")
    sys_s  = bulk(:Al; a=a * u"Å", cubic=true) * (3, 3, 3)
    H      = ustrip.(hessian(sys_s, model))
    Np     = length(sys_p);  Ns = length(sys_s)
    L_p    = SMatrix{3,3,Float64}(ustrip.(hcat(sys_p.cell.cell_vectors...)))
    L_s    = SMatrix{3,3,Float64}(ustrip.(hcat(sys_s.cell.cell_vectors...)))
    Linv_p = inv(L_p);  Linv_s = inv(L_s)
    fp = [Linv_p * SVector{3,Float64}(ustrip.(sys_p[i].position)) for i in 1:Np]
    fs = [Linv_p * SVector{3,Float64}(ustrip.(sys_s[k].position)) for k in 1:Ns]
    # s2p / p2s maps
    s2p = Vector{Int}(undef, Ns)
    for k in 1:Ns
        for i in 1:Np
            d = mod.(fs[k], 1.0) .- mod.(fp[i], 1.0); d = d .- round.(d)
            if norm(d) < 1e-6; s2p[k] = i-1; break; end
        end
    end
    p2s = Vector{Int}(undef, Np)
    for i in 1:Np
        cands = findall(s2p .== (i-1))
        r0    = SVector{3,Float64}(ustrip.(sys_p[i].position))
        p2s[i] = cands[argmin([norm(SVector{3,Float64}(ustrip.(sys_s[k].position)) .- r0) for k in cands])] - 1
    end
    masses = [ustrip(sys_p[i].mass) for i in 1:Np]
    B      = 2π * inv(transpose(L_p))
    q_X    = B * [0.0, 0.5, 0.5]
    Dq     = zeros(ComplexF64, 3Np, 3Np)
    for i in 0:Np-1, j in 0:Np-1
        dm = zeros(ComplexF64, 3, 3)
        for k in 0:Ns-1
            s2p[k+1] == j || continue
            R    = L_p * (fs[k+1] - fp[j+1])
            Rsf  = Linv_s * R; Rsf = Rsf .- round.(Rsf)
            Rmi  = L_s * Rsf
            eph  = exp(im * dot(q_X, Rmi))
            for α in 1:3, β in 1:3
                dm[α, β] += H[3*p2s[i+1]+α, 3k+β] * eph
            end
        end
        ms = sqrt(masses[i+1] * masses[j+1])
        Dq[3i+1:3i+3, 3j+1:3j+3] .= dm ./ ms
    end
    for i in 1:3Np, j in i:3Np
        Dq[i,j] = (Dq[i,j] + conj(Dq[j,i])) / 2; Dq[j,i] = conj(Dq[i,j])
    end
    F = eigen(Hermitian(Dq))
    return real.(F.values), F.vectors, q_X, L_p
end

println("Computing mean-model X-point eigenvectors …")
ω2_X0, vecs_X0, q_X0, L_prim0 = get_xpoint_eigenvectors(model, a0)
freqs_X0 = sign.(ω2_X0) .* sqrt.(abs.(ω2_X0)) .* FREQ_THz
@printf("  Mean-model X-point: ω = [%.4f, %.4f, %.4f] THz  (T1, T2, L)\n",
        freqs_X0...)

# ─────────────────────────────────────────────────────────────────────────────
#  AD curvature utilities  (same as x_point_curvature.jl)
# ─────────────────────────────────────────────────────────────────────────────
function make_displaced_design(sys0, r0, displacements, mdl)
    _cv   = sys0.cell.cell_vectors
    _nat  = length(sys0)
    _spec = [sys0[i].species for i in 1:_nat]
    function B(d)
        atoms = Vector{Atom}(undef, _nat)
        for i in 1:_nat
            rnew = r0[i] .+ d .* displacements[i]
            atoms[i] = Atom(_spec[i], collect(rnew .* u"Å"), missing)
        end
        sys_new = periodic_system(atoms, _cv)
        return ustrip.(ACEpotentials.Models.potential_energy_basis(sys_new, mdl))
    end
    return B
end

const h_fd        = 1e-4   # FD step for outer second derivative [Å√amu]
const N_prim_cells = 4     # 4-atom conventional FCC cell = 4 primitive cells

"""
    xpoint_omega2(θ_i, a_i, vecs_X, model)

Compute the three X-point ω² values [eV/Å²/amu] for parameter vector `θ_i`
at lattice constant `a_i` using the AD curvature operator.
`vecs_X` are the (real) 3D eigenvectors from the mean model (fixed directions).
Returns `(ω2_T1, ω2_T2, ω2_L)` in ascending eigenvalue order (same as eigen).
"""
function xpoint_omega2(θ_i, a_i, vecs_X, mdl)
    sys_conv_i = bulk(:Al; a=a_i * u"Å", cubic=true)
    m_Al       = ustrip(sys_conv_i[1].mass)
    r_conv_i   = [SVector{3,Float64}(ustrip.(sys_conv_i[j].position))
                  for j in 1:N_prim_cells]
    B_i  = 2π * inv(transpose(SMatrix{3,3,Float64}(
               ustrip.(hcat(bulk(:Al; a=a_i*u"Å").cell.cell_vectors...)))))
    q_X_i = B_i * [0.0, 0.5, 0.5]
    ω2 = Vector{Float64}(undef, 3)
    for μ in 1:3
        e_μ = vecs_X[:, μ]
        disps = SVector{3,Float64}[
            real.(e_μ .* exp(im * dot(q_X_i, r_conv_i[j]))) ./ sqrt(m_Al)
            for j in 1:N_prim_cells
        ]
        B_func = make_displaced_design(sys_conv_i, r_conv_i, disps, mdl)
        dB(d)  = ForwardDiff.derivative(B_func, d)
        C_μ    = (dB(+h_fd) .- dB(-h_fd)) ./ (2h_fd)
        ω2[μ]  = dot(C_μ, θ_i) / N_prim_cells
    end
    return ω2   # [T1, T2, L] (ascending by mean-model eigenvalue order)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Rejection sampling
# ─────────────────────────────────────────────────────────────────────────────
N_target = 50   # desired committee size

println("\n\n═══════════════════════════════════════════════════════════════════")
println("  X-point rejection sampling  (target N = $N_target)")
println("  Condition: ω²_L > ω²_T1  AND  ω²_L > ω²_T2")
println("═══════════════════════════════════════════════════════════════════\n")

accepted_samples = Vector{Vector{Float64}}()  # accepted θ vectors
accepted_a       = Vector{Float64}()          # corresponding lattice constants
accepted_ω2      = Vector{Vector{Float64}}()  # [T1, T2, L] ω² for each accepted

n_proposed  = 0
n_accepted  = 0

t_total = @elapsed while n_accepted < N_target
    # ── Draw one candidate from the hypercube ────────────────────────────────
    u   = rand(Float64, length(lower_hc))
    δ   = hypercube_eigenvectors * (lower_hc .+ (upper_hc .- lower_hc) .* u)
    θ_i = θ .+ δ
    n_proposed += 1

    # ── Relax lattice constant for this candidate ────────────────────────────
    ACEpotentials.Models.set_linear_parameters!(model, θ_i)
    a_i = ACEWorkflow.relax_lattice_constant(model, :Al)
    ACEpotentials.Models.set_linear_parameters!(model, θ)   # restore mean params

    # ── Compute X-point ω² via AD curvature ─────────────────────────────────
    ω2_i = xpoint_omega2(θ_i, a_i, vecs_X0, model)
    ω2_T1, ω2_T2, ω2_L = ω2_i

    # ── Acceptance criterion: L above both T modes ───────────────────────────
    accepted = ω2_L > ω2_T1 && ω2_L > ω2_T2

    rate_str = @sprintf("%.1f%%", 100 * n_accepted / n_proposed)
    status   = accepted ? "✓ ACCEPT" : "✗ reject"
    freqs_i  = sign.(ω2_i) .* sqrt.(abs.(ω2_i)) .* FREQ_THz
    @printf("  [%4d proposed | %3d accepted | %s] a=%.5f Å  ω=[%+.4f,%+.4f,%+.4f] THz  %s\n",
            n_proposed, n_accepted, rate_str, a_i,
            freqs_i[1], freqs_i[2], freqs_i[3], status)

    if accepted
        push!(accepted_samples, θ_i)
        push!(accepted_a,       a_i)
        push!(accepted_ω2,      ω2_i)
        n_accepted += 1
    end
end

@printf("\n  Sampling done in %.1f s\n", t_total)
@printf("  Acceptance rate: %d / %d  (%.1f%%)\n",
        n_accepted, n_proposed, 100 * n_accepted / n_proposed)

# ─────────────────────────────────────────────────────────────────────────────
#  Save results
# ─────────────────────────────────────────────────────────────────────────────
out_dir = joinpath(result.dir, "results")
mkpath(out_dir)

# Committee parameter corrections relative to θ (same format as pops_corrections.csv)
corrections_mat = reduce(hcat, s .- θ for s in accepted_samples)'   # N×P matrix
writedlm(joinpath(out_dir, "xpoint_stable_pops_corrections.csv"), corrections_mat, ',')
@printf("  Corrections saved  →  %s\n",
        joinpath(out_dir, "xpoint_stable_pops_corrections.csv"))

# Frequencies + lattice constants CSV
let freq_file = joinpath(out_dir, "xpoint_stable_pops_frequencies.csv")
    open(freq_file, "w") do io
        println(io, "# X-point phonon frequencies for rejection-sampled (L>T) POPS ensemble")
        println(io, "# Generated: ", Dates.now())
        println(io, "# member,a_Ang,freq_T1_THz,freq_T2_THz,freq_L_THz,omega2_T1,omega2_T2,omega2_L")
        for i in 1:n_accepted
            fi = sign.(accepted_ω2[i]) .* sqrt.(abs.(accepted_ω2[i])) .* FREQ_THz
            @printf(io, "%d,%.6f,%.6f,%.6f,%.6f,%.10f,%.10f,%.10f\n",
                    i, accepted_a[i], fi[1], fi[2], fi[3],
                    accepted_ω2[i][1], accepted_ω2[i][2], accepted_ω2[i][3])
        end
    end
    @printf("  Frequencies saved  →  %s\n", freq_file)
end

println("\n═══════════════════════════════════════════════════════════════════")
println("Done.")
