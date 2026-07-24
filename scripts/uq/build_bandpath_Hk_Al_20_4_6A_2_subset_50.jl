# build_bandpath_Hk_Al_20_4_6A_2_subset_50.jl
#
# Standalone PARALLEL builder for the per-basis supercell Hessians H_k that the
# band-path phonon constraints need (the one-off ~O(n_params) cost).  Each H_k
# is independent, so this is embarrassingly parallel over Julia threads.
#
# H_k = ∂²(basis_k energy)/∂r²  on the force-constant supercell, evaluated by
# setting the linear parameters to the k-th unit vector.  Serialized to the
# cache file that forest_bandpath_project_…jl loads (isfile → skips its build).
#
# Run on as many cores as you have:
#     julia --project -t $(nproc) scripts/uq/build_bandpath_Hk_Al_20_4_6A_2_subset_50.jl
# or under SLURM:
#     julia --project -t $SLURM_CPUS_PER_TASK scripts/uq/build_bandpath_Hk_Al_20_4_6A_2_subset_50.jl
#
# 314 Hessians / N threads: ~1 h serial → a few minutes on 32–64 cores.

using LinearAlgebra, Printf, Serialization, Unitful
using ACEpotentials, ACEWorkflow
using AtomsBuilder
using AtomsCalculatorsUtilities.SitePotentials: hessian

element   = :Al
N_cell_fc = 3      # force-constant supercell (≥ 2×cutoff) — MUST match the pipeline

result     = load_model(element, 20, 4, 6, 2; dataset_name="subset_50_percent")
model      = result.model
lin_params = result.lin_params
n_params   = length(lin_params)

nt = Threads.nthreads()
nt == 1 && @warn "running single-threaded — launch with `julia -t <ncores>` to parallelise"
@printf("Model %s: %d basis functions, %d threads\n", result.name, n_params, nt)

a_eq = ACEWorkflow.relax_lattice_constant(model, element)
@printf("a_eq = %.6f Å\n", a_eq)
sys_super = bulk(element; a=a_eq*u"Å", cubic=true) * (N_cell_fc, N_cell_fc, N_cell_fc)
@printf("Force-constant supercell: %d atoms → %d×%d Hessians each\n",
        length(sys_super), 3length(sys_super), 3length(sys_super))

hk_file = "$(result.dir)/results/bandpath_Hk_$(N_cell_fc)x$(N_cell_fc)x$(N_cell_fc).jls"
if isfile(hk_file)
    cached = deserialize(hk_file)
    if abs(cached.a_eq - a_eq) < 1e-4
        println("Cache already present and current: $hk_file — nothing to do.")
        exit(0)
    else
        @warn "cache at a=$(cached.a_eq) ≠ current a_eq=$a_eq — rebuilding"
    end
end

# ── Parallel build: independent H_k, per-thread model copies, BLAS single ────
H_all    = Vector{Matrix{Float64}}(undef, n_params)
models_t = [deepcopy(model) for _ in 1:nt]
blas_old = BLAS.get_num_threads(); BLAS.set_num_threads(1)
done = Threads.Atomic{Int}(0); t0 = time()
println("Building $n_params per-basis Hessians …")
Threads.@threads :static for k in 1:n_params
    m = models_t[Threads.threadid()]
    e_k = zeros(n_params); e_k[k] = 1.0
    ACEpotentials.Models.set_linear_parameters!(m, e_k)
    H_all[k] = ustrip.(hessian(sys_super, m))
    d = Threads.atomic_add!(done, 1) + 1
    d % 10 == 0 && @printf("\r  %d / %d  (%.1f min, ETA %.1f min)      ",
                           d, n_params, (time()-t0)/60, (time()-t0)/d*(n_params-d)/60)
end
BLAS.set_num_threads(blas_old)
@printf("\r  done: %d Hessians in %.1f min.                         \n", n_params, (time()-t0)/60)

serialize(hk_file, (H_all=H_all, a_eq=a_eq, N_cell=N_cell_fc))
println("Serialized → $hk_file")
println("Now run:  julia --project -t <N> scripts/uq/forest_bandpath_project_Al_20_4_6A_2_subset_50.jl")
