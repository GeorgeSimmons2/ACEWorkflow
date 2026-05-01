# test/strain_hessian_lattice_deriv.jl
#
# Compares the finite-difference (FD) and AD-based implementations of
# strain_hessian_lattice_constant_derivative.
#
# The two functions compute ∂H/∂a where H is the 6×6×n_params strain-Hessian
# basis operator.  They should agree to within O(h²) ≈ 1e-8.
#
# Uses the smallest pre-built model (Al_12_4_6A_3) to keep runtime short.

using Test
using LinearAlgebra
using StaticArrays
using ACEpotentials
using ACEWorkflow

result = load_model(:Al, 20, 5, 6.0, 3)
model  = result.model

@testset "strain_hessian_lattice_constant_derivative: FD vs AD" begin

    a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)

    # Build both derivative functions at equilibrium
    dH_fd = ACEWorkflow.Elasticity.strain_hessian_lattice_constant_derivative(model, :Al; a=a_eq)
    dH_ad = ACEWorkflow.Elasticity.strain_hessian_lattice_constant_derivative_ad(model, :Al; a=a_eq)

    # Evaluate both at the equilibrium lattice constant
    R_fd = dH_fd(a_eq)   # 6×6×n_params
    R_ad = dH_ad(a_eq)   # 6×6×n_params

    @testset "Shape agreement" begin
        @test size(R_fd) == size(R_ad)
    end

    @testset "Numerical agreement (rel. tol 1e-4)" begin
        # Relative error across the full tensor
        rel_err = norm(R_fd - R_ad) / (norm(R_fd) + 1e-30)
        @test rel_err < 1e-4
    end

    @testset "Slice-wise agreement on first 10 basis functions" begin
        # Check individual 6×6 slices to catch per-parameter discrepancies
        n_check = min(10, size(R_fd, 3))
        for k in 1:n_check
            slice_err = norm(R_fd[:, :, k] - R_ad[:, :, k]) /
                        (norm(R_fd[:, :, k]) + 1e-30)
            @test slice_err < 1e-4
        end
    end

    @testset "AD result varies with a (non-trivial)" begin
        # Sanity check: the derivative should not be identically zero
        @test norm(R_ad) > 1e-10
    end

end
