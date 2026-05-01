# test/create_model.jl
#
# Tests for the model creation and strain-energy pipeline extracted from
# scripts/model_building/create_model.jl.
#
# Uses the smallest pre-built model (Al_12_4_6A_3) to keep runtime short.

using Test
using StaticArrays
using Unitful
using AtomsBuilder: bulk
using ACEpotentials
using ACEWorkflow

result = load_model(:Al, 12, 4, 6.0, 3)
model  = result.model

@testset "create_model" begin

    a_eq = ACEWorkflow.relax_lattice_constant(model, :Al)   # Float64, unitless Å

    # ── Test 1: zero strain → strained_cell_energy == potential_energy(bulk) ──
    #
    # strained_cell_energy with ε = 0 must recover the undeformed cell energy.
    # Both sides go through reference_system, which attaches u"Å" as needed.
    @testset "Zero-strain energy identity" begin
        E_strained = ACEWorkflow.Elasticity.strained_cell_energy(
            @SVector([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
            model=model, element=:Al, a=a_eq
        )
        E_direct = ACEpotentials.potential_energy(
            bulk(:Al; a=a_eq * u"Å"), model
        )
        @test E_strained ≈ E_direct
    end

    # ── Test 2: relax_lattice_constant is in the expected physical range ───────
    @testset "Lattice constant range" begin
        @test 3.5 < a_eq < 4.5      # Al FCC: ~4.05 Å
    end

    # ── Test 3: strain_hessian_GPa satisfies Born stability criteria ──────────
    @testset "Born stability" begin
        res       = ACEWorkflow.strain_hessian_GPa(model, :Al; a=a_eq)
        C11, C12, C44 = res.C[1,1], res.C[1,2], res.C[4,4]
        @test C44        > 0    # shear stable
        @test C11 - C12  > 0    # tetragonal stability
        @test C11 + 2C12 > 0    # bulk modulus > 0
    end

end
