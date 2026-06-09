addprocs(Threads.nthreads())
@everywhere using ACEfit
using ACEWorkflow

result = build_model(:W, 20, 4,
                     5.0, 3;
                     training_xyz = "data/W/df_W_train.extxyz",
                     energy_key = :energy,
                     force_key  = :forces,
                     virial_key = :dft_virials)

                     