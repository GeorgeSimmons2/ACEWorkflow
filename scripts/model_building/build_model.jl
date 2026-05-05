using Distributed
addprocs(Sys.CPU_THREADS)   # or however many cores you want

@everywhere begin
    import Pkg
    Pkg.activate("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow")
end

@everywhere using ACEWorkflow

model_name, model_dir, _, _ = build_model(:Al, 20, 4, 6.0, 3; training_xyz="/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/scratch/manual_df_train_Al.xyz")
model_name, model_dir, _, _ = build_model(:Al, 20, 4, 6.0, 4; training_xyz="/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/scratch/manual_df_train_Al.xyz")