"""
    phonon_committee(model, coeffs_committee, result, element::Symbol;
                     N_per_seg=30, N_cell=3, file_prefix="")

Compute phonon band structures for the mean model and each committee member,
then save plots and raw frequency data.

Arguments:
  - `model`             : ACE model (mean parameters)
  - `coeffs_committee`  : Vector of coefficient vectors, one per committee member
  - `result`            : ACEWorkflow result (provides `dir`, `lin_params`)
  - `element`           : Element symbol, e.g. `:Al`
  - `N_per_seg`         : q-points per BZ segment (default 30)
  - `N_cell`            : Supercell repetitions along each axis (default 3)
  - `file_prefix`       : Prepended to output file names

Returns `(x_vals, all_freqs, x_ticks, labels)` where `all_freqs[1]` is the
mean model and `all_freqs[2:end]` are the committee members.
"""
function phonon_committee(model, coeffs_committee, result, element::Symbol;
                          N_per_seg=30, N_cell=3, file_prefix="")
    orig_coeffs = result.lin_params
    N = length(coeffs_committee)
    all_freqs = Vector{Matrix{Float64}}(undef, N + 1)
    x_vals_out  = nothing
    x_ticks_out = nothing
    labels_out  = nothing

    # ── Mean model (member 0) first: establishes x_vals / x_ticks / labels ──
    println("\n--- Committee member 0 / $N (mean model) ---")
    let a0 = relax_lattice_constant(model, element)
        sp0 = bulk(element; a=a0*u"Å")
        ss0 = bulk(element; a=a0*u"Å", cubic=true) * (N_cell, N_cell, N_cell)
        x_vals_out, all_freqs[1], x_ticks_out, labels_out =
            compute_phonon_bands(sp0, ss0, model; N_per_seg, n_modes=nothing)
        ω_min = round(minimum(all_freqs[1]), sigdigits=4)
        ω_max = round(maximum(all_freqs[1]), sigdigits=4)
        println("  Frequency range : $ω_min … $ω_max THz")
    end

    # ── Committee members 1:N ─────────────────────────────────────────────
    models_t = [deepcopy(model) for _ in 1:Threads.nthreads()]
    Threads.@threads for i in 1:N
        m = models_t[Threads.threadid()]
        ACEpotentials.Models.set_linear_parameters!(m, coeffs_committee[i])

        a_i  = relax_lattice_constant(m, element)
        sp_i = bulk(element; a=a_i*u"Å")
        ss_i = bulk(element; a=a_i*u"Å", cubic=true) * (N_cell, N_cell, N_cell)

        _, freqs, _, _ = compute_phonon_bands(sp_i, ss_i, m; N_per_seg, n_modes=nothing)
        all_freqs[i + 1] = freqs

        ω_min  = round(minimum(freqs), sigdigits=4)
        ω_max  = round(maximum(freqs), sigdigits=4)
        n_imag = count(freqs .< 0)
        @printf("\n  member %d: %.4g … %.4g THz%s\n", i, ω_min, ω_max,
                n_imag > 0 ? "  ($n_imag imaginary modes)" : "")
    end

    # Restore original (mean) model
    ACEpotentials.Models.set_linear_parameters!(model, orig_coeffs)

    # ── Plotting helper ───────────────────────────────────────────────────
    function _committee_plot(unit_label, scale)
        fig = Figure(size=(750, 500))
        ax  = Axis(fig[1, 1];
                   xlabel       = "Wave vector",
                   ylabel       = unit_label == "THz" ? "Frequency (THz)" : "Energy (eV)",
                   title        = "$element phonon bands — ACE committee",
                   xticks       = (x_ticks_out, labels_out),
                   xgridvisible = false)

        if unit_label == "eV"
            energies_all = [f .* scale for f in all_freqs]
            emin = floor(minimum(minimum.(energies_all)) / 0.01) * 0.01
            emax = ceil( maximum(maximum.(energies_all)) / 0.01) * 0.01
            ax.yticks = emin:0.01:emax
        end

        # Committee members — light grey
        for freqs in all_freqs[2:end]
            data = freqs .* scale
            for b in 1:size(data, 1)
                lines!(ax, x_vals_out, data[b, :];
                       color=RGBAf(0.6, 0.6, 0.6, 0.4), linewidth=1.0)
            end
        end

        # Mean model — blue/red on top
        mean_data = all_freqs[1] .* scale
        for b in 1:size(mean_data, 1)
            branch = mean_data[b, :]
            color  = minimum(branch) < 0 ? RGBAf(0.8, 0.1, 0.1, 0.95) :
                                            RGBAf(0.2, 0.4, 0.7, 0.95)
            lines!(ax, x_vals_out, branch; color, linewidth=2.0)
        end

        hlines!(ax, [0.0]; color=:black, linestyle=:dash, linewidth=0.8)
        vlines!(ax, x_ticks_out; color=(:black, 0.3), linewidth=0.8)
        return fig
    end

    fig_thz = _committee_plot("THz", 1.0)
    save("$(result.dir)/results/$(file_prefix)phonon_committee_THz_$(N_cell)x$(N_cell)x$(N_cell).png", fig_thz)
    display(fig_thz)

    fig_ev = _committee_plot("eV", THz_to_meV / 1000)
    save("$(result.dir)/results/$(file_prefix)phonon_committee_eV_$(N_cell)x$(N_cell)x$(N_cell).png", fig_ev)
    display(fig_ev)

    # ── Save data ─────────────────────────────────────────────────────────
    writedlm("$(result.dir)/results/$(file_prefix)phonon_committee_x_vals_$(N_cell)x$(N_cell)x$(N_cell).csv",
             x_vals_out, ',')
    stacked = reduce(vcat, all_freqs)   # ((N+1)*Nmodes) × Nq
    writedlm("$(result.dir)/results/$(file_prefix)phonon_committee_freqs_THz_$(N_cell)x$(N_cell)x$(N_cell).csv",
             stacked, ',')

    return x_vals_out, all_freqs, x_ticks_out, labels_out
end
