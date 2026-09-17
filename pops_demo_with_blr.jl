using ACEWorkflow, CairoMakie, LinearAlgebra, Random

f(x) = (x^3 + 0.01 * x^4) * 0.1 + sin(x) * x * 10.0

x = LinRange(-10,10, 20)
x_true = LinRange(-13, 13, 1000)
y_true = f.(x_true)
y = f.(x)

function OLS(x, y; d=4, σ²=nothing)
    X = zeros(length(y), d)
    X_pred  = zeros(1000, d)
    x_pred  = LinRange(-13, 13, 1000)
    for i = 1:length(y)
        for j = 1:d
            X[i,j] = x[i] ^ (j-1)
        end
    end
    for i = 1:size(X_pred, 1)
        for j = 1:d
            X_pred[i,j] = x_pred[i] ^ (j-1)
        end
    end
    C = X'*X
    A = C \ X'
    lev = diag(X * A)
    coeffs = C \ (X' * y)
    errors = y .- (X * coeffs)
    pointwise_corrections = A' .* (errors ./ lev)

    y_mean = [sum(coeffs .* vec(X_pred[i, :])) for i=1:size(X_pred, 1)]
    y_pops = [[sum((pointwise_corrections[j,:] .+ coeffs) .* vec(X_pred[i, :])) for i=1:size(X_pred, 1)] for j =1:length(y)]

    # --- Bayesian linear regression, prior N(0, I) on coefficients ---
    s2 = isnothing(σ²) ? sum(abs2, errors) / max(length(y) - d, 1) : σ²
    Λ = C ./ s2 + I                      # posterior precision
    Σ = inv(Λ)                           # posterior covariance
    coeffs_blr = Σ * (X' * y) ./ s2      # posterior mean
    y_blr = X_pred * coeffs_blr
    # predictive std: epistemic + noise
    y_blr_std = [sqrt(dot(view(X_pred, i, :), Σ, view(X_pred, i, :)) + s2)
                 for i = 1:size(X_pred, 1)]

    return y_mean, y_pops, y_blr, y_blr_std, x_pred
end

y_mean, y_pops, y_blr, y_blr_std, x_pred = OLS(x, y; d=3)

fig = Figure()
ax  = Axis(fig[1,1], xlabel="x", ylabel="y")
scatter!(ax, x, y, label="Data")
lines!(ax, x_true, y_true, label="True")
lines!(ax, x_pred, y_mean, label="OLS")
for (i, y_pop) in enumerate(y_pops)
    if (i == 1)
        lines!(ax, x_pred, y_pop, linestyle=:dash, label="POPS", color=(:grey, 0.5))
    else
        lines!(ax, x_pred, y_pop, linestyle=:dash, color=(:grey, 0.5))
    end
end
band!(ax, x_pred, y_blr .- 2 .* y_blr_std, y_blr .+ 2 .* y_blr_std,
      color=(:dodgerblue, 0.25), label="BLR ±2σ")
lines!(ax, x_pred, y_blr, color=:dodgerblue, label="BLR")
axislegend(ax, position=:lt)
save("pops_demo_with_blr.png", fig)