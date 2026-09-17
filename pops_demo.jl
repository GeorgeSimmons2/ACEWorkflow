using ACEWorkflow, CairoMakie, LinearAlgebra

f(x) = sin(6*x) * exp(- (x ^ 2 / 3))

x = LinRange(-1, 1, 100)
x_true = LinRange(-1, 1, 1000)
y_true = f.(x_true)
y = f.(x)

function OLS(x, y; d=4)
    X = zeros(length(y), d)
    X_pred  = zeros(1000, d)
    x_pred  = LinRange(-1, 1, 1000)
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


    return y_mean, y_pops, x_pred
end

y_mean, y_pops, x_pred = OLS(x, y)

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
axislegend(ax, position=:rt)
save("pops_demo.png", fig)