using Distributed
addprocs(Sys.CPU_THREADS)   # or however many cores you want

@everywhere begin
    import Pkg
    Pkg.activate("/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow")
end

@everywhere using ACEWorkflow

for i = 12:20
    percent = 5
    model_name, model_dir, _, _ = build_model(:Al, i, 4, 6.0, 2; training_xyz="/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/data/Al/manual_df_train_Al.extxyz", dataset_name = "")
end

# for i = 12:20
#     percent = 20
#     model_name, model_dir, _, _ = build_model(:Al, i, 4, 6.0, 3; training_xyz="/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/data/Al/subset_$(percent)_percent.extxyz", dataset_name = "subset_$(percent)_percent")
# end

# for i = 12:20
#     percent = 50
#     model_name, model_dir, _, _ = build_model(:Al, i, 4, 6.0, 3; training_xyz="/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/data/Al/subset_$(percent)_percent.extxyz", dataset_name = "subset_$(percent)_percent")
# end
    # model_name, model_dir, _, _ = build_model(:Al, 20, 4, 6.0, 4; training_xyz="/storage/astro2/phupfb/PhD/acestuff/ACEWorkflow/scratch/manual_df_train_Al.xyz")