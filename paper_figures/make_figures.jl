# paper_figures/make_figures.jl
#
# Run the canonical script behind one or more paper figures.  This is a thin driver:
# it does not reimplement anything, it shells out to the script named in manifest.jl
# so the figure a reviewer builds is made by exactly the code that made ours.
#
# Run from the repository root:
#   julia --project paper_figures/make_figures.jl            # every figure
#   julia --project paper_figures/make_figures.jl <id> ...   # selected figures
#   DRY_RUN=1 julia --project paper_figures/make_figures.jl  # print commands only
#
# Inputs are checked first; a figure whose inputs are incomplete is SKIPPED with an
# explanation rather than started and left to fail somewhere in the middle.

include(joinpath(@__DIR__, "manifest.jl"))
using Printf

dry = get(ENV, "DRY_RUN", "0") != "0"
sel = isempty(ARGS) ? FIGURES : [figure_by_id(a) for a in ARGS]

ran = String[]; skipped = Tuple{String,String}[]; failed = String[]

for f in sel
    println("\n", "═"^78); println(f.id); println("═"^78)

    missing_inputs = [p for (p, _, _) in f.inputs if !isfile(joinpath(REPO, p))]
    if !isempty(missing_inputs)
        msg = "$(length(missing_inputs)) missing input(s), first: $(first(missing_inputs))"
        @printf("SKIP — %s\n", msg)
        println("       run  julia --project paper_figures/check_inputs.jl $(f.id)  for the full list")
        push!(skipped, (f.id, msg)); continue
    end

    isempty(f.env) || println("  environment variables this script reads:")
    for e in f.env; println("    ", e.first, " = ", e.second); end
    println("  \$ ", f.cmd)

    if dry
        push!(skipped, (f.id, "DRY_RUN")); continue
    end

    t = @elapsed try
        run(Cmd(`sh -c $(f.cmd)`; dir=REPO))
        push!(ran, f.id)
    catch err
        @error "figure $(f.id) failed" exception=err
        push!(failed, f.id)
    end
    @printf("  [%.1f min]\n", t/60)

    for o in f.outputs
        full = joinpath(REPO, o)
        @printf("  %-6s %s\n", isfile(full) ? "[ok]" : "[MISS]", o)
    end
end

println("\n", "─"^78)
@printf("built %d, skipped %d, failed %d\n", length(ran), length(skipped), length(failed))
for (id, why) in skipped; @printf("  skipped %-40s %s\n", id, why); end
for id in failed;         @printf("  FAILED  %s\n", id); end
exit(isempty(failed) ? 0 : 1)
