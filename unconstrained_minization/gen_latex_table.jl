using Printf

function parse_csv_line(line)
    fields = String[]
    buffer = IOBuffer()
    in_quotes = false
    i = firstindex(line)

    while i <= lastindex(line)
        c = line[i]

        if c == '"'
            j = nextind(line, i)

            if in_quotes && j <= lastindex(line) && line[j] == '"'
                print(buffer, '"')
                i = j
            else
                in_quotes = !in_quotes
            end
        elseif c == ',' && !in_quotes
            push!(fields, String(take!(buffer)))
        else
            print(buffer, c)
        end

        i = nextind(line, i)
    end

    push!(fields, String(take!(buffer)))
    return fields
end

function format_number(s)
    value = tryparse(Float64, strip(s))
    return value === nothing ? "--" : @sprintf("%.3e", value)
end

function generate_latex_table(csv_path, tex_path)
    lines = readlines(csv_path)

    headers = strip.(parse_csv_line(first(lines)))
    headers[1] = replace(headers[1], "\ufeff" => "")

    rows = Dict{String,String}[]

    for line in lines[2:end]
        isempty(strip(line)) && continue

        values = parse_csv_line(line)
        length(values) == length(headers) ||
            error("Invalid CSV row: $line")

        push!(rows, Dict(zip(headers, strip.(values))))
    end

    abbreviations = Dict(
        "Gradient Descent" => "GD",
        "Conjugate Gradient" => "CG",
        "BFGS" => "BFGS",
        "Regularized Newton" => "RN",
        "Pure Newton" => "PN"
    )

    statuses = Dict(
        "Strict local min" => "Strict min",
        "Not a minimum" => "Not min",
        "Not stationary" => "Not stationary",
        "Inconclusive" => "Inconclusive",
        "Max iterations" => "Max iter"
    )

    functions = unique(row["Function"] for row in rows)

    # Produces exactly \\ in the LaTeX output.
    row_end = repeat("\\", 2)

    open(tex_path, "w") do io
        println(io, raw"\begin{table}[ht]")
        println(io, raw"    \centering")
        println(io, raw"    \caption{Numerical results for the benchmark functions.}")
        println(io, raw"    \label{tab:numerical-results}")
        println(io, raw"    \small")
        println(io, raw"    \begin{tabular}{llrrrl}")
        println(io, raw"        \hline")

        header = [
            "Function",
            "Method",
            raw"$f(x_k)$",
            "Iterations",
            "Time (s)",
            "Status"
        ]

        println(io, "        ", join(header, " & "), " ", row_end)
        println(io, raw"        \hline")

        for function_name in functions
            function_rows = filter(
                row -> row["Function"] == function_name,
                rows
            )

            for (i, row) in enumerate(function_rows)
                fname = i == 1 ? function_name : ""
                method = get(abbreviations, row["Method"], row["Method"])
                fx = format_number(row["Objective Value"])

                iterations = isempty(row["Iterations"]) ?
                    "--" : row["Iterations"]

                time = format_number(row["Time (s)"])
                status = get(statuses, row["Status"], row["Status"])

                cells = [
                    fname,
                    method,
                    fx,
                    iterations,
                    time,
                    status
                ]

                println(io, "        ", join(cells, " & "), " ", row_end)
            end

            println(io, raw"        \hline")
        end

        println(io, raw"    \end{tabular}")
        println(io, raw"\end{table}")
    end

    println("LaTeX table saved to: ", tex_path)
end

generate_latex_table(
    "hw_unconstrained_minization/results.csv",
    "hw_unconstrained_minization/results_table.tex"
)