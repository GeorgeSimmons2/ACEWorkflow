module Models

using ACEpotentials, DelimitedFiles, LinearAlgebra, ExtXYZ, SparseArrays, AtomsCalculators, Distributed, ACEfit


# ── Path helpers ──────────────────────────────────────────────────────────────

"""
    model_name(elements, totaldegree, smoothness, rcut, order) -> String

Return the canonical model name string.  Elements are sorted and concatenated
so the name is independent of input order, e.g. `[:Al, :Ti]` → `"AlTi_20_5_6A_3"`.
Single-element example: `[:Al]` → `"Al_20_5_6A_3"`.
"""
function model_name(elements::AbstractVector{Symbol}, totaldegree::Int, smoothness::Int,
                    rcut::Real, order::Int)
    elem_str = join(sort(String.(elements)))
    return "$(elem_str)_$(totaldegree)_$(smoothness)_$(Int(rcut))A_$(order)"
end

# Single-element convenience (backward compat)
model_name(element::Symbol, totaldegree::Int, smoothness::Int, rcut::Real, order::Int) =
    model_name([element], totaldegree, smoothness, rcut, order)

"""
    model_dir(name) -> String

Return the absolute path to the model subdirectory `models/<name>/`.
"""
function model_dir(name::AbstractString)
    root = joinpath(@__DIR__, "..", "..", "models", name)
    return abspath(root)
end

model_dir(elements::AbstractVector{Symbol}, totaldegree::Int, smoothness::Int,
          rcut::Real, order::Int) =
    model_dir(model_name(elements, totaldegree, smoothness, rcut, order))

# Single-element convenience (backward compat)
model_dir(element::Symbol, totaldegree::Int, smoothness::Int, rcut::Real, order::Int) =
    model_dir(model_name(element, totaldegree, smoothness, rcut, order))


# ── load_model ────────────────────────────────────────────────────────────────

"""
    load_model(elements, totaldegree, smoothness, rcut, order; training_xyz=nothing)

Load an ACE model and its associated data from `models/<name>/`.

`elements` may be a `Vector{Symbol}` (multispecies) or a single `Symbol`
(single-species convenience).

If the model does not exist yet, `build_model` is called automatically.
`training_xyz` is only required when building a new model; if omitted and the
model is missing, an `ArgumentError` is thrown.

Returns a `NamedTuple` with fields:
  - `model`            – ACE model with fitted linear parameters set
  - `A`, `Y`, `P`, `W` – design-matrix and prior data (matrices/vectors)
  - `lin_params`       – fitted linear parameters (Vector)
  - `pops_corrections` – POPS pointwise corrections matrix (may be `nothing`)
"""
function load_model(elements::AbstractVector{Symbol}, totaldegree::Int, smoothness::Int,
                    rcut::Real, order::Int; training_xyz::Union{String,Nothing}=nothing,
                    dataset_name::String = "full")
    name = model_name(elements, totaldegree, smoothness, rcut, order)
    if (dataset_name == "full")
        dir  = model_dir(name)
    else
        dir  = model_dir(name * "_" * dataset_name)
    end
    json = joinpath(dir, "$(name).json")

    if !isfile(json)
        if training_xyz === nothing
            throw(ArgumentError(
                "Model $(name) not found at $(dir).\n" *
                "Pass training_xyz=<path to .xyz file> to build it automatically."
            ))
        end
        @info "Model $name not found — building from scratch..."
        return build_model(elements, totaldegree, smoothness, rcut, order;
                           training_xyz=training_xyz)
    end

    model, _ = ACEpotentials.load_model(json)

    A  = readdlm(joinpath(dir, "A.csv"), ',')
    Y  = vec(readdlm(joinpath(dir, "Y.csv"), ','))
    P  = readdlm(joinpath(dir, "P.csv"), ',')
    W  = vec(readdlm(joinpath(dir, "W.csv"), ','))
    lin_params = vec(readdlm(joinpath(dir, "lin_params.csv"), ','))

    ACEpotentials.Models.set_linear_parameters!(model, lin_params)

    return (model=model, A=A, Y=Y, P=P, W=W,
            lin_params=lin_params,
            name=name, dir=dir)
end

# Single-element convenience (backward compat)
load_model(element::Symbol, totaldegree::Int, smoothness::Int,
           rcut::Real, order::Int; kw...) =
    load_model([element], totaldegree, smoothness, rcut, order; kw...)


# ── build_model ───────────────────────────────────────────────────────────────

"""
    build_model(elements, totaldegree, smoothness, rcut, order; training_xyz,
                energy_key=:dft_energy, force_key=:dft_forces, virial_key=:dft_virials,
                stride=1)

Build an ACE model, fit it with POPS, and save everything to `models/<name>/`.

`elements` may be a `Vector{Symbol}` (multispecies) or a single `Symbol`
(single-species convenience).

Saves:
  - `<name>.json`          – serialised ACE model
  - `A.csv`, `Y.csv`, `P.csv`, `W.csv` – design matrix and prior
  - `lin_params.csv`       – fitted OLS parameters
  - `pops_corrections.csv` – POPS pointwise corrections

Returns the same `NamedTuple` as `load_model`.
"""
function build_model(elements::AbstractVector{Symbol}, totaldegree::Int, smoothness::Int,
                     rcut::Real, order::Int;
                     training_xyz::String,
                     energy_key::Symbol = :dft_energy,
                     force_key::Symbol  = :dft_forces,
                     virial_key::Symbol = :dft_virials,
                     stride::Int        = 1,
                     dataset_name::String = "full",
                     ace_model_kwargs...)

    name = model_name(elements, totaldegree, smoothness, rcut, order)
    if (dataset_name == "full")
        dir  = model_dir(name)
    else
        dir  = model_dir(name * "_" * dataset_name)
    end

    mkpath(joinpath(dir, "results"))

    @info "Building model $name"
    @info "  totaldegree=$totaldegree  order=$order  rcut=$(rcut)Å  smoothness=$smoothness"
    @info "  training data: $training_xyz  (stride=$stride)"

    # ── Load training data ────────────────────────────────────────────────────
    training_configs = ExtXYZ.load(training_xyz)
    stride > 1 && (training_configs = training_configs[1:stride:end])

    # ── Build ACE model ───────────────────────────────────────────────────────
    model = ace1_model(elements=elements, totaldegree=totaldegree,
                       order=order, rcut=rcut; ace_model_kwargs...)

    data = ACEpotentials.make_atoms_data(training_configs, model;
                energy_key = energy_key,
                force_key  = force_key,
                virial_key = virial_key,
                weights    = Dict("default" => Dict("E"=>30., "F"=>1., "V"=>1.)))

    P = ACEpotentials._make_prior(model, smoothness, nothing)
    A, Y, W = ACEfit.assemble(data, model)

    # ── Weighted / regularised fit ────────────────────────────────────────────
    Ap       = Diagonal(W) * A / P
    Yw       = W .* Y
    A_reg    = Ap'*Ap .+ P'*P ./ size(Ap, 1)
    lin_params = P \ (A_reg \ (Ap' * Yw))
    ACEpotentials.Models.set_linear_parameters!(model, lin_params)
    json = joinpath(dir, "$(name).json")
    ACEpotentials.save_model(model, json)
    writedlm(joinpath(dir, "A.csv"),              A,               ',')
    writedlm(joinpath(dir, "Y.csv"),              Y,               ',')
    writedlm(joinpath(dir, "P.csv"),              P,               ',')
    writedlm(joinpath(dir, "W.csv"),              W,               ',')
    writedlm(joinpath(dir, "lin_params.csv"),     lin_params,      ',')

    # # ── POPS pointwise corrections ────────────────────────────────────────────
    # # load POPSRegression from the package src (avoids circular include)
    # pops_mod = Base.require(Main, :POPSRegression)
    # pops_corrections = pops_mod.corrections(Ap, Yw, P; leverage_percentile=0.0)

    # # ── Persist ───────────────────────────────────────────────────────────────

    # writedlm(joinpath(dir, "pops_corrections.csv"), pops_corrections, ',')

    @info "Saved to $dir"

    return (model=model, A=A, Y=Yw, P=P, W=W,
            lin_params=lin_params, #pops_corrections=pops_corrections,
            name=name, dir=dir)
end

# Single-element convenience (backward compat)
build_model(element::Symbol, totaldegree::Int, smoothness::Int,
            rcut::Real, order::Int; kw...) =
    build_model([element], totaldegree, smoothness, rcut, order; kw...)


export model_name, model_dir, load_model, build_model

end
