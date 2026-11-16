using Printf
using Statistics

function generate_latex_table(csv_path, tex_path)
    lines = readlines(csv_path)
    rows = [split(strip(line), ',') for line in lines[2:end] if !isempty(strip(line))]

    # Exclude pair 1, used for warm-up
    rows = filter(row -> parse(Int, row[1]) > 1, rows)

    abbreviations = Dict(
        "Gradient Descent" => "GD",
        "BFGS" => "BFGS",
        "Regularized Newton" => "RN"
    )

    methods = ["Gradient Descent", "BFGS", "Regularized Newton"]
    converged_statuses = ["Strict local min", "Not a minimum", "Inconclusive"]
    row_end = repeat("\\", 2)

    open(tex_path, "w") do io
        println(io, raw"\begin{table}[ht]")
        println(io, raw"    \centering")
        println(io, raw"    \caption{Average numerical results for the seismic travel-time minimization problem.}")
        println(io, raw"    \label{tab:seismic-results}")
        println(io, raw"    \small")
        println(io, raw"    \begin{tabular}{lrrrrrr}")
        println(io, raw"        \hline")

        header = [
            "Method",
            "Avg. obj.",
            "Avg. iter.",
            "Avg. time (s)",
            "Converged",
            "Strict min",
            "Not min"
        ]

        println(io, "        ", join(header, " & "), " ", row_end)
        println(io, raw"        \hline")

        for method in methods
            method_rows = filter(row -> row[2] == method, rows)

            objectives = Float64[]
            iterations = Float64[]
            times = Float64[]

            for row in method_rows
                fx = tryparse(Float64, row[3])
                k = tryparse(Float64, row[4])
                t = tryparse(Float64, row[5])

                if fx !== nothing && isfinite(fx)
                    push!(objectives, fx)
                end

                if k !== nothing && isfinite(k)
                    push!(iterations, k)
                end

                if t !== nothing && isfinite(t)
                    push!(times, t)
                end
            end

            avg_objective = isempty(objectives) ? NaN : mean(objectives)
            avg_iterations = isempty(iterations) ? NaN : mean(iterations)
            avg_time = isempty(times) ? NaN : mean(times)

            converged = count(row -> row[7] in converged_statuses, method_rows)
            strict_min = count(row -> row[7] == "Strict local min", method_rows)
            not_min = count(row -> row[7] == "Not a minimum", method_rows)

            cells = [
                abbreviations[method],
                @sprintf("%.3e", avg_objective),
                @sprintf("%.2f", avg_iterations),
                @sprintf("%.3e", avg_time),
                string(converged),
                string(strict_min),
                string(not_min)
            ]

            println(io, "        ", join(cells, " & "), " ", row_end)
        end

        println(io, raw"        \hline")
        println(io, raw"    \end{tabular}")
        println(io, raw"\end{table}")
    end
end

generate_latex_table("./seismic/results/results.csv", "./seismic/results/results_table.tex")