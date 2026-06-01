# check_X_point.jl
#
# For every member of the POPS forest (pops_corrections.csv + lin_params),
# build the primitive-cell dynamical matrix at the FCC X-point and check
# the 3 lowest non-zero modes.
#
# FCC fractional coordinates are scale-invariant, so the topology maps
# (frac_prim, frac_super, s2p_map, p2s_map, masses) are precomputed once.
# Per member: relax a_k, scale L matrices, compute Hessian, transform D(q).
# The Hessian loop is parallelised with Threads.@threads.
#
# Output:
#   xpt_freqs_THz.csv   — N × 3 matrix; columns = mode 1, 2, 3 (ascending)
#   xpt_stable.csv      — N × 1 Bool column (1 = all three modes >= 0)
#
# Usage (N threads):
#   julia --project -t auto check_X_point.jl

using LinearAlgebra, StaticArrays, Printf, DelimitedFiles, ForwardDiff
using AtomsBuilder, AtomsCalculators, Unitful, AtomsBase
using AtomsCalculatorsUtilities.SitePotentials: hessian
using ACEWorkflow, ACEpotentials

# ─────────────────────────────────────────────────────────────────────────────
#  Physical constant
# ─────────────────────────────────────────────────────────────────────────────
const _eV_J   = 1.602176634e-19
const _Å_m    = 1.0e-10
const _amu_kg = 1.66053906660e-27
const FREQ_THz = sqrt(_eV_J / (_Å_m^2 * _amu_kg)) / (2π * 1e12)   # ≈ 15.633

# ─────────────────────────────────────────────────────────────────────────────
#  Minimal dynamical-matrix machinery (copied from phonon_bands_ace.jl)
# ─────────────────────────────────────────────────────────────────────────────
function _lattice_mat(cell_vectors)
    a, b, c = cell_vectors
    SMatrix{3,3,eltype(a)}(hcat(a, b, c))
end

function precompute_force_constants(sys_prim, sys_super, model)
    H  = ustrip.(hessian(sys_super, model))
    Np = length(sys_prim)
    Ns = length(sys_super)

    L_prim  = SMatrix{3,3,Float64}(ustrip.(_lattice_mat(sys_prim.cell.cell_vectors)))
    L_super = SMatrix{3,3,Float64}(ustrip.(_lattice_mat(sys_super.cell.cell_vectors)))
    Linv_p  = inv(L_prim)
    Linv_s  = inv(L_super)

    frac_prim  = [Linv_p * SVector{3,Float64}(ustrip.(sys_prim[i].position))  for i in 1:Np]
    frac_super = [Linv_p * SVector{3,Float64}(ustrip.(sys_super[k].position)) for k in 1:Ns]

    s2p_map = Vector{Int}(undef, Ns)
    for k in 1:Ns
        matched = false
        for i in 1:Np
            d = mod.(frac_super[k], 1.0) .- mod.(frac_prim[i], 1.0)
            d = d .- round.(d)
            if norm(d) < 1e-6
                s2p_map[k] = i - 1
                matched = true
                break
            end
        end
        matched || error("Supercell atom $k did not match any primitive atom")
    end

    p2s_map = Vector{Int}(undef, Np)
    for i in 1:Np
        candidates = findall(s2p_map .== (i - 1))
        r_ref = SVector{3,Float64}(ustrip.(sys_prim[i].position))
        dists = [norm(SVector{3,Float64}(ustrip.(sys_super[k].position)) .- r_ref)
                 for k in candidates]
        p2s_map[i] = candidates[argmin(dists)] - 1
    end

    masses = [ustrip(sys_prim[i].mass) for i in 1:Np]
    return (; H, frac_prim, frac_super, masses,
              L=L_prim, L_super, Linv_super=Linv_s, Np, Ns, p2s_map, s2p_map)
end

function dynamical_matrix_from_fc(fc, q_cart::AbstractVector{<:Real})
    (; H, frac_prim, frac_super, masses, L, L_super, Linv_super, Np, Ns, p2s_map, s2p_map) = fc
    Dq = zeros(ComplexF64, 3Np, 3Np)
    for i in 0:Np-1, j in 0:Np-1
        dm = zeros(ComplexF64, 3, 3)
        for k in 0:Ns-1
            s2p_map[k+1] == j || continue
            R_cart  = L * (frac_super[k+1] - frac_prim[j+1])
            R_sfrac = Linv_super * R_cart
            R_sfrac = R_sfrac .- round.(R_sfrac)
            R_mi    = L_super * R_sfrac
            eph = exp(im * dot(q_cart, R_mi))
            for α in 1:3, β in 1:3
                dm[α, β] += H[3*p2s_map[i+1]+α, 3k+β] * eph
            end
        end
        ms = sqrt(masses[i+1] * masses[j+1])
        Dq[3i+1:3i+3, 3j+1:3j+3] .= dm ./ ms
    end
    for i in 1:3Np, j in i:3Np
        Dq[i,j] = (Dq[i,j] + conj(Dq[j,i])) / 2
        Dq[j,i] = conj(Dq[i,j])
    end
    return Dq
end

eigenvalues_to_freq_THz(ω2) = sign.(ω2) .* sqrt.(abs.(ω2)) .* FREQ_THz

function dq_freqs(Dq::Matrix{ComplexF64})
    ω2 = real.(eigen(Hermitian(Dq)).values)   # sorted ascending
    return eigenvalues_to_freq_THz(ω2)
end

# ─────────────────────────────────────────────────────────────────────────────
#  Load model and POPS corrections
# ─────────────────────────────────────────────────────────────────────────────
result = load_model(:Al, 12, 4, 6, 3)
model  = result.model
θ_mean = result.lin_params

println("Loading pops_corrections.csv …")
POPS_mat = readdlm(joinpath(result.dir, "pops_corrections.csv"), ',', Float64)
N        = size(POPS_mat, 1)
println("  $N members × $(size(POPS_mat, 2)) params")

# ─────────────────────────────────────────────────────────────────────────────
#  Geometry — precomputed once at the mean-model equilibrium lattice constant
# ─────────────────────────────────────────────────────────────────────────────
N_conv = 3   # 3×3×3 conventional supercell, L ≈ 12.15 Å > 2×cutoff = 12 Å

print("Relaxing mean-model lattice constant … ")
a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)
@printf("a_eq = %.6f Å\n", a_eq)
@printf("Supercell side = %.3f Å  (2×cutoff = %.1f Å)\n", N_conv*a_eq, 12.0)

sys_prim  = bulk(:Al; a=a_eq*u"Å")
sys_super = bulk(:Al; a=a_eq*u"Å", cubic=true) * (N_conv, N_conv, N_conv)

# FCC X-point in Cartesian (Å⁻¹): [0, 2π/a, 0]
# (primitive cell reciprocal lattice; matches fcc_band_path definition)
q_X = [0.0, 2π / a_eq, 0.0]

print("Precomputing mean-model force constants … ")
fc_mean = precompute_force_constants(sys_prim, sys_super, model)
println("done.  ($(length(sys_super)) supercell atoms)")

# Mean-model check
D_mean = dynamical_matrix_from_fc(fc_mean, q_X)
f_mean = dq_freqs(Matrix{ComplexF64}(Hermitian((D_mean + D_mean') / 2)))
@printf("Mean model X-pt freqs: [%.3f, %.3f, %.3f] THz\n",
        f_mean[1], f_mean[2], f_mean[3])

# ─────────────────────────────────────────────────────────────────────────────
#  Precompute Hessian basis  H_basis[:,:,n] = ∂²B_n/∂r²  at a_eq
#
#  ACE linearity:  H(θ,a) = Σ_n θ_n · H_basis[:,:,n](a)
#  so the per-member Hessian is a BLAS matrix-vector product — no per-member
#  hessian() call needed.
#
#  Launch with:  julia --project -t auto check_X_point.jl
# ─────────────────────────────────────────────────────────────────────────────
Nbasis = length(θ_mean)
Ns3    = 3 * length(sys_super)   # 3 × 108 = 324

# IFT first-order lattice correction: δa = −b′·δθ / K
#   b′[n]  = dE_n/da  at a_eq  (per-basis energy derivative)
#   K      = θ_mean · b″  (curvature of mean-model energy vs a)
println("\nComputing IFT lattice-correction ingredients …")
lattice_basis(a) = ustrip.(u"eV", ACEpotentials.Models.potential_energy_basis(
    ACEWorkflow.Elasticity.reference_system(:Al; a=a), model))
b_prime  = ForwardDiff.derivative(lattice_basis, a_eq)
b_double = ForwardDiff.derivative(
    a -> ForwardDiff.derivative(lattice_basis, a), a_eq)
K = dot(θ_mean, b_double)
@printf("  K = %.4e eV/Å²\n", K)

# Precompute H_basis[:,:,n]  (Nbasis Hessians, threaded over basis index)
@printf("Precomputing H_basis (%d Hessians on %d atoms, %d threads) …\n",
        Nbasis, length(sys_super), Threads.nthreads())
H_basis = zeros(Ns3, Ns3, Nbasis)
model_copies = [deepcopy(model) for _ in 1:Threads.nthreads()]

Threads.@threads for n in 1:Nbasis
    m  = model_copies[Threads.threadid()]
    en = zeros(Nbasis); en[n] = 1.0
    ACEpotentials.Models.set_linear_parameters!(m, en)
    H_basis[:, :, n] = ustrip.(hessian(sys_super, m))
    println(n)
end
println("  done.")

# dH/da at (a_eq, θ_mean): central FD, only 2 Hessian calls.
# Per-member correction is δa_k · dH_da_mean; error is O(δθ²) since
# dH_da(θ_k) − dH_da(θ_mean) = O(δθ), multiplied by δa_k = O(δθ).
println("Computing dH/da at mean geometry (2 Hessians) …")
ACEpotentials.Models.set_linear_parameters!(model, θ_mean)
ε_a     = 1e-4   # Å
H_plus  = ustrip.(hessian(bulk(:Al; a=(a_eq+ε_a)*u"Å", cubic=true) * (N_conv,N_conv,N_conv), model))
H_minus = ustrip.(hessian(bulk(:Al; a=(a_eq-ε_a)*u"Å", cubic=true) * (N_conv,N_conv,N_conv), model))
dH_da_mean = (H_plus .- H_minus) ./ (2ε_a)
println("  done.")

# Flatten H_basis to (Ns3²×Nbasis) for BLAS dgemv; prealloc work vector
HB     = reshape(H_basis, Ns3^2, Nbasis)   # view, no copy
dH_vec = vec(dH_da_mean)
h_k    = Vector{Float64}(undef, Ns3^2)     # reused every iteration

# L scale factors (FCC topology maps are scale-invariant)
L_prim_unit  = fc_mean.L        / a_eq
L_super_unit = fc_mean.L_super  / a_eq
Linv_s_unit  = fc_mean.Linv_super * a_eq

# ─────────────────────────────────────────────────────────────────────────────
#  Forest loop  — O(Nbasis × Ns3²) BLAS per member; no hessian() call
# ─────────────────────────────────────────────────────────────────────────────
freqs_out  = Matrix{Float64}(undef, N, 3)
stable_out = Vector{Bool}(undef, N)

println("\nChecking $N POPS members at the X-point …")
for k in 1:N
    k % max(1, N ÷ 20) == 0 && @printf("\r  %d / %d …", k, N)

    δθ   = POPS_mat[k, :]
    θ_k  = δθ .+ θ_mean
    δa_k = -dot(b_prime, δθ) / K
    a_k  = a_eq + δa_k

    # H_k = H_basis ⊗ θ_k  +  δa_k · dH_da_mean  (in-place, zero allocation)
    mul!(h_k, HB, θ_k)
    BLAS.axpy!(δa_k, dH_vec, h_k)
    H_k = reshape(h_k, Ns3, Ns3)   # view

    fc_k = merge(fc_mean, (;
        H          = H_k,
        L          = a_k * L_prim_unit,
        L_super    = a_k * L_super_unit,
        Linv_super = (1 / a_k) * Linv_s_unit,
    ))

    q_X_k = [0.0, 2π / a_k, 0.0]
    D     = dynamical_matrix_from_fc(fc_k, q_X_k)
    f     = dq_freqs(Matrix{ComplexF64}(Hermitian((D + D') / 2)))

    freqs_out[k, :]  = f[1:3]
    stable_out[k]    = all(f[1:3] .>= 0)
end
println("\r  Done. ($N members)                    ")

# ─────────────────────────────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────────────────────────────
n_stable   = count(stable_out)
n_unstable = N - n_stable

println("\n", repeat('─', 62))
@printf("  Stable   : %d / %d  (%.2f%%)\n", n_stable,   N, 100n_stable/N)
@printf("  Unstable : %d / %d  (%.2f%%)\n", n_unstable, N, 100n_unstable/N)
println(repeat('─', 62))

for j in 1:3
    col = freqs_out[:, j]
    @printf("  Mode %d  [%.3f, %.3f] THz  (mean %.3f THz)  %s\n",
            j, minimum(col), maximum(col), sum(col)/N,
            any(col .< 0) ? "← has imaginary members" : "")
end
println(repeat('─', 62))

if n_unstable > 0
    println("\n  Unstable member indices and their X-pt freqs (THz):")
    for k in findall(.!stable_out)
        f = freqs_out[k, :]
        @printf("    member %4d : [%+.3f, %+.3f, %+.3f] THz\n", k, f[1], f[2], f[3])
    end
end

# ─────────────────────────────────────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────────────────────────────────────
out_dir = joinpath(result.dir, "results")
mkpath(out_dir)

writedlm(joinpath(out_dir, "xpt_freqs_THz.csv"),  freqs_out,              ',')
writedlm(joinpath(out_dir, "xpt_stable.csv"),      Int.(stable_out),      ',')

println("\nSaved:")
println("  $(out_dir)/xpt_freqs_THz.csv  ($N × 3 matrix, THz)")
println("  $(out_dir)/xpt_stable.csv     ($N × 1 Bool column)")
