using LinearAlgebra, Random, Statistics, DelimitedFiles, POPSRegression;
using JSON, Unitful;
using AtomsBuilder, GeomOpt, AtomsCalculators, AtomsBase;
using AtomsCalculators: potential_energy, forces, virial;
using ACEpotentials;
using AtomsBase: FlexibleSystem, FastSystem, position;
using AtomsBase;
using StaticArrays;
using CairoMakie;
using DelimitedFiles

function corrections(X, Y, Gamma; coeffs=nothing, leverage_percentile=0.0)
    C      = (Gamma' * Gamma ./ size(X, 1) .+ X' * X)
    A      = C \ X'
    leverage = diag(X * A)
    if (coeffs == nothing)
        coeffs = C \ (X' * Y)
    end
    errors = Y .- (X * coeffs)
    leverage_threshold = quantile(leverage, leverage_percentile)
    mask = leverage .>= leverage_threshold
    pointwise_corrections = A[:,mask]'
    pointwise_corrections = pointwise_corrections .* (errors[mask] ./ leverage[mask])
    return pointwise_corrections, coeffs
end

function hypercube(pointwise_corrections; percentile_clipping = 0.0)
    eig = eigen(Symmetric(pointwise_corrections' * pointwise_corrections))
    eigvals = eig.values
    eigvecs = eig.vectors

    mask = eigvals .> maximum(eigvals) * 1e-8
    eigvecs = eigvecs[:, mask]
    eigvals = eigvals[mask]

    projections = eigvecs
    projected = pointwise_corrections * projections

    lower = [quantile(projected[:, j], percentile_clipping / 100) for j in 1:size(projected, 2)]
    upper = [quantile(projected[:, j], 1.0 - percentile_clipping / 100) for j in 1:size(projected, 2)]

    bounds = vcat(lower', upper')  # (2 x N)

    return eigvecs, bounds
end

function sample_hypercube(projections, bounds, coeffs; number_of_committee_members = 50)
    lower, upper = bounds[1, :], bounds[2, :]

    U = rand(Float64, (number_of_committee_members, size(lower, 1)))

    committee = projections * (lower[:, :]' .+ (upper .- lower)[:,:]' .* U)'
    δθ        = committee * committee' ./ size(committee, 2)

    committee = coeffs[:,:] .+ committee
    
    return committee, δθ  
end

function read_model(totaldegree; suffix = "_dia_with_test_set_aside_total_degree_$totaldegree.csv", folder = "Si_totdeg_$(totaldegree)")
    model = ace1_model(elements = [:Si,],
                       order = 4, totaldegree = totaldegree,
                       rcut = 6.0)
    Ap = readdlm("./$(folder)/design_matrix$suffix",  ',')
    Y  = readdlm("./$(folder)/Y$suffix",              ',')
    P  = readdlm("./$(folder)/prior$suffix",          ',')
    Y = Y[:,1]
    params = P \ (Ap \ Y)
    ACEpotentials.Models.set_linear_parameters!(model, params)
    return model, Ap, Y, P
end

function volume_energy_curve_(;element, lattice_consts_, model)
    volumes = zeros(length(lattice_consts_))
    energies = zeros(length(lattice_consts_))
    for i = 1:length(lattice_consts_)
        a = lattice_consts_[i]u"Å"
        bulk_system = bulk(element, cubic = true; a = a)
        vec_1, vec_2, vec_3 = bulk_system.cell.cell_vectors
        volume = dot(vec_1, cross(vec_2, vec_3))
        E = potential_energy(bulk_system, model)
        energies[i] = ustrip(E)
        volumes[i] = ustrip(volume)
    end
    return volumes, energies
end

function OLS(x::Vector, y::Vector; degree::Int=3)
    n = length(x)
    X = zeros(n, degree + 1)
    for i in 0:degree
        X[:, i+1] .= x .^ i
    end
    coeffs = X \ y
    return coeffs, X
end

function QoI(coeffs)
    c0, c1, c2 = coeffs
    c = c0
    b = c1
    a = c2
    discriminant = b^2 - 4*a*c

    if discriminant < 0
        return NaN, NaN, NaN
    end

    v0 = -b / (2 * a)
    e0 = a*v0^2 + b*v0 + c
    B  = - b
    
    B_quantity = B * u"eV/Å^3"           
    B_GPa = uconvert(u"GPa", B_quantity) |> ustrip
    return v0, e0, B_GPa
end

function wrap()
    vol_en     = [[17.813109023568057, -163.06663280425], [18.322054995670012, -163.110046112], 
                 [18.831000967771949, -163.1415028745], [19.339946939873919, -163.162464704125],
                 [19.848892911975845, -163.174242557], [20.3578388840778, -163.177966801875], 
                 [20.866784856179741, -163.17463056225], [21.375730828281693, -163.165132595375], 
                 [21.884676800383637, -163.150242437875], [22.393622772485578, -163.130671150875], 
                 [22.902568744587512, -163.107009425875], [23.411514716689464, -163.079814823625]]
    gap_volumes    = [x[1] for x in vol_en]; gap_energies = [x[2] for x in vol_en];
    
    lattice_consts = cbrt.(gap_volumes * 8)
    volumes, energies, co_energies = volume_energy_curve_(element = :Si, lattice_consts_ = lattice_consts, model = model, num_in_committee = 50);
    
    volumes_per_atom     = volumes     ./ 8
    energies_per_atom    = energies    ./ 8
    co_energies_per_atom = co_energies ./ 8
    
    coeffs, X = OLS(volumes_per_atom, energies_per_atom; degree=2)
    ace_V0, ace_e0, ace_B     = QoI(coeffs)
    return ace_B
end
using LinearAlgebra, Statistics, Random

function solve_C(C, B)
    # Solve C X = B for X robustly. Works when B is vector or 2D array
    return C \ B
end

# Data generation
Random.seed!(0)
N = 30
x = range(-4π, 4π, length=N)
f_true(x) = (x^3 + 0.01x^4)*0.1 + sin(x)*x*10.0
y = f_true.(x) + 0.05 * randn(N)

# Design matrix function
function design_matrix(x::AbstractArray, degree::Int=3)
    # returns (N, P) matrix with columns [1, x, x^2, ... , x^degree]
    return hcat([x.^d for d in 0:degree]...)
end

degree = 3
X = design_matrix(x, degree)
N, P = size(X)

# Test grid for plotting
xt = range(-4π, 4π, length=1000)
Xt = design_matrix(xt, degree)

# Global ridge least-squares
λ_reg = 1e-8
# Normal matrix C = X^T X + lambda I
C = X'X + λ_reg * I(P)
b = X'y
Theta_L = solve_C(C, b)

# Global prediction
y_hat_global = Xt * Theta_L

function compute_soft_pops_allpoints(X, y, C, Theta_L; tau=0.05, eps=1e-12)
    """
    Compute softened POPS parameter vectors for every training point.
    Returns: Theta_list (N x P), residuals_after, rhos
    """
    N, P = size(X)
    # Pre-solve C^{-1} applied to all columns of X
    U = solve_C(C, X')  # shape (P, N) where column i is u_i
    Theta_list = zeros(N, P)
    r_after = zeros(N)
    rhos = zeros(N)
    
    for i in 1:N
        Fi = X[i, :]            # shape (P,)
        Ei = y[i]
        delta = Ei - Fi'Theta_L
        ui = U[:, i]            # P-vector = C^{-1} Fi
        h = Float64(Fi'ui) + eps   # leverage (scalar), add tiny eps to avoid zero
        
        if abs(delta) <= tau
            rho = 0.0
            DeltaTheta = zeros(P)
            r_new = delta
        else
            # minimal rho that ensures |r(ρ)| <= tau : rho >= (|delta|/tau - 1)/h
            rho = max(0.0, (abs(delta) / tau - 1.0) / h)
            factor = (rho / (1.0 + rho * h))
            DeltaTheta = factor * ui * delta
            r_new = delta - Fi'DeltaTheta
        end
        
        Theta_list[i, :] = Theta_L + DeltaTheta
        r_after[i] = r_new
        rhos[i] = rho
    end
    
    return Theta_list, r_after, rhos
end

# Add ExtXYZ to imports
using ExtXYZ
using DelimitedFiles

# Replace data loading section with:
# Load Si dataset and prepare model
# ...existing code...

# Load Si dataset and prepare model
Si_dataset = ExtXYZ.load("Si_dataset.xyz")
deleteat!(Si_dataset, 1)

# Take every 10th structure
Si_dataset_subset = Si_dataset[1:10:end]

# Set up model parameters
model = ace1_model(elements = [:Si],
                  order = 3,
                  totaldegree = 12, 
                  rcut = 6.0)

# Prepare training data with subset
data_keys = (energy_key = "dft_energy", force_key = "dft_force", virial_key = "dft_virial")
train = ACEpotentials.make_atoms_data(Si_dataset_subset, model; 
                                    energy_key = data_keys[1], 
                                    force_key = data_keys[2], 
                                    virial_key = data_keys[3],
                                    weights = Dict("default"=>Dict("E"=>30.0, "F"=>1.0, "V"=>1.0)))

# actual assembly of the least square system 
P = ACEpotentials._make_prior(model, 4, nothing)
A, Y, W = ACEfit.assemble(train, model)
# Change appendix to reflect subset
appendix = "si_tiny_subset"

# ...rest of code remains the same...
writedlm("design_matrix_$appendix.csv", A, ',')
writedlm("W_$appendix.csv", W, ',')
writedlm("Y_$appendix.csv", Y, ',')
writedlm("prior_$appendix.csv", P, ',')

# Calculate weighted design matrix and targets
#W = W[:, 1]  # Extract vector from matrix
#Y = Y[:, 1]  # Extract vector from matrix
Ap = Diagonal(W) * (A/P)
Y = W .* Y
params = P \ (Ap \ Y)
ACEpotentials.Models.set_linear_parameters!(model, params)


# Regular POPS
leverage_percentile = 0.5
pointwise_corrections, _ = corrections(Ap, Y, P; 
    leverage_percentile=leverage_percentile, 
    coeffs=nothing)
corr_ = (P \ pointwise_corrections')'

# Generate hypercube samples for regular POPS
percentile_clipping = 5.0
hypercube_support, hypercube_bounds = hypercube(corr_; 
    percentile_clipping=percentile_clipping)
committee, misspec_sigma = sample_hypercube(hypercube_support, 
    hypercube_bounds, params)
co_coeffs = committee
co_ps_vec = [co_coeffs[:,i] for i in 1:size(co_coeffs,2)]

# Set up model with regular POPS committee
set_committee!(model, co_ps_vec)

# Soft POPS implementation
N, _ = size(A)
C = Ap'Ap / N + P'P
tau = 0.05 # Soft POPS tolerance parameter

# Generate soft POPS ensemble
Theta_list, residuals_after, rhos = compute_soft_pops_allpoints(
    A, Y, C, params, tau=tau)

# Convert soft POPS parameters to model format
soft_pops_coeffs = [P \ Theta_list[i,:] for i in 1:N]

# Save results
writedlm("regular_pops_coeffs_$appendix.csv", co_ps_vec, ',')
writedlm("soft_pops_coeffs_$appendix.csv", soft_pops_coeffs, ',')


# Test energy predictions on a bulk Si system
bulk_Si = bulk(:Si)

# Regular POPS predictions
E, co_E = @committee potential_energy(bulk_Si, model)
regular_pops_energies = ustrip.(co_E)

# Soft POPS predictions 
soft_pops_energies = Float64[]
for coeff in soft_pops_coeffs
    ACEpotentials.Models.set_linear_parameters!(model, coeff)
    E = potential_energy(bulk_Si, model)
    push!(soft_pops_energies, ustrip(E))
end

# Save energy predictions
writedlm("regular_pops_energies_$appendix.csv", regular_pops_energies, ',')
writedlm("soft_pops_energies_$appendix.csv", soft_pops_energies, ',')

# ...existing code...

# Plotting functions
function plot_energy_predictions()
    fig = Figure(resolution=(800, 400))
    
    # First subplot: Regular vs Soft POPS energy distributions
    ax1 = Axis(fig[1,1], 
        xlabel="Energy (eV)", 
        ylabel="Density",
        title="Energy Predictions Distribution")
    
    hist!(ax1, regular_pops_energies, bins=20, 
          normalization=:pdf, label="Regular POPS")#, alpha=0.5)
    hist!(ax1, soft_pops_energies, bins=20, 
          normalization=:pdf, label="Soft POPS")#, alpha=0.5)
    axislegend(ax1)
    
    # Second subplot: Correlation plot
    ax2 = Axis(fig[1,2],
        xlabel="Regular POPS Energy (eV)",
        ylabel="Soft POPS Energy (eV)",
        title="Regular vs Soft POPS")
    
    scatter!(ax2, ones(length(regular_pops_energies)), regular_pops_energies, 
            alpha=0.5, markersize=5, color = :green)
    scatter!(ax2, ones(length(soft_pops_energies)) * 2, soft_pops_energies, 
            alpha=0.5, markersize=5, color = :green)
            
    
    # Add identity line
    lims = (minimum([regular_pops_energies; soft_pops_energies]),
            maximum([regular_pops_energies; soft_pops_energies]))
    # lines!(ax2, lims, lims, linestyle=:dash, color=:black)
    
    save("energy_predictions_comparison_$appendix.png", fig)
    return fig
end

# Energy-volume curve analysis
function calculate_bulk_modulus()
    # Define volume range based on lattice constants
    lattice_consts = range(5.0, 6.0, length=20)  # Å
    
    # Regular POPS E-V curves
    volumes_reg = Float64[]
    energies_committee = Vector{Float64}[]
    
    for a in lattice_consts
        bulk_sys = bulk(:Si, cubic=true, a=a*u"Å")
        cell_vecs = bulk_sys.cell.cell_vectors
        vol = ustrip(dot(cell_vecs[1], cross(cell_vecs[2], cell_vecs[3])))
        push!(volumes_reg, vol)
        
        # Get committee predictions
        _, co_E = @committee potential_energy(bulk_sys, model)
        push!(energies_committee, ustrip.(co_E))
    end
    
    # Soft POPS E-V curves
    volumes_soft = Float64[]
    energies_soft = [Float64[] for _ in 1:length(soft_pops_coeffs)]
    
    for a in lattice_consts
        bulk_sys = bulk(:Si, cubic=true, a=a*u"Å")
        cell_vecs = bulk_sys.cell.cell_vectors
        vol = ustrip(dot(cell_vecs[1], cross(cell_vecs[2], cell_vecs[3])))
        push!(volumes_soft, vol)
        
        # Get soft POPS predictions
        for (i, coeff) in enumerate(soft_pops_coeffs)
            ACEpotentials.Models.set_linear_parameters!(model, coeff)
            E = potential_energy(bulk_sys, model)
            push!(energies_soft[i], ustrip(E))
        end
    end
    
    # Plot E-V curves
    fig = Figure(resolution=(800, 400))
    
    ax1 = Axis(fig[1,1],
        xlabel="Volume (Å³)",
        ylabel="Energy (eV)",
        title="Energy-Volume Curves")
    
    # Plot regular POPS committee
    for i in 1:size(energies_committee[1], 1)
        E_curve = [E[i] for E in energies_committee]
        lines!(ax1, volumes_reg, E_curve, color=(:blue, 0.1))
    end
    
    # Plot soft POPS ensemble
    for E_curve in energies_soft
        lines!(ax1, volumes_soft, E_curve, color=(:red, 0.1))
    end
    
    # Calculate bulk modulus distributions
    B_reg = Float64[]
    for i in 1:size(energies_committee[1], 1)
        E_curve = [E[i] for E in energies_committee]
        coeffs, _ = OLS(volumes_reg, E_curve, degree=2)
        _, _, B = QoI(coeffs)
        push!(B_reg, B)
    end
    
    B_soft = Float64[]
    for E_curve in energies_soft
        coeffs, _ = OLS(volumes_soft, E_curve, degree=2)
        _, _, B = QoI(coeffs)
        push!(B_soft, B)
    end
    
    # Plot bulk modulus distributions
    ax2 = Axis(fig[1,2],
        xlabel="Bulk Modulus (GPa)",
        ylabel="Density",
        title="Bulk Modulus Distribution")
    
    hist!(ax2, B_reg, bins=20, normalization=:pdf, 
          label="Regular POPS", alpha=0.5)
    hist!(ax2, B_soft, bins=20, normalization=:pdf, 
          label="Soft POPS", alpha=0.5)
    axislegend(ax2)
    
    save("EV_curves_and_bulk_modulus_$appendix.png", fig)
    writedlm("bulk_modulus_regular_$appendix.csv", B_reg, ',')
    writedlm("bulk_modulus_soft_$appendix.csv", B_soft, ',')
    
    return fig
end

# Generate and save all plots
energy_plot = plot_energy_predictions()
ev_plot = calculate_bulk_modulus()