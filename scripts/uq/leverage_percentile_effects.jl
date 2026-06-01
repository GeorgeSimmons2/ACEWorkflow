using ACEWorkflow

result = load_model(:Al, 12, 4, 6, 3)
model  = result.model

# ── POPS ─────────────────────────────────────────────────────────────────────
Ap = Diagonal(result.W) * result.A / result.P
Yw = result.W .* result.Y
full_pops_corrections = corrections(Ap, Yw, result.P; leverage_percentile=0.0)
half_pops_corrections = corrections(Ap, Yw, result.P; leverage_percentile=0.5)

full_max_min = ([], [])
half_max_min = ([], [])

for i = 1:size(full_pops_corrections, 1)
    full_max = maximum(full_pops_corrections[i,:])
    full_min = minimum(full_pops_corrections[i,:])
    push!(full_max_min[1], full_min)
    push!(full_max_min[2], full_max)
end
for i = 1:size(half_pops_corrections, 1)
    half_max = maximum(half_pops_corrections[i,:])
    half_min = minimum(half_pops_corrections[i,:])
    push!(half_max_min[1], half_min)
    push!(half_max_min[2], half_max)
end

println("Same bounds?")
println(full_max_min == half_max_min)
