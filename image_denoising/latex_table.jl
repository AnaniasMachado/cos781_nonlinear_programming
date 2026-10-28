using Printf

csv_path = "image_denoising/results/statistics.csv"
output_path = "image_denoising/results/table.txt"

lines = readlines(csv_path)
rows = [split(strip(line), ",") for line in lines[2:end] if !isempty(strip(line))]

row_end = repeat("\\", 2)

open(output_path, "w") do io
    println(io, raw"\begin{table}[ht]")
    println(io, raw"    \centering")
    println(io, raw"    \caption{Numerical results for different values of $\lambda$.}")
    println(io, raw"    \label{tab:denoising_results}")
    println(io, raw"    \small")
    println(io, raw"    \begin{tabular}{llrrrr}")
    println(io, raw"        \hline")
    println(io, "        ", raw"$\lambda$ & Method & $f(x)$ & Iterations & Time (s) & MSE ", row_end)
    println(io, raw"        \hline")

    previous_lambda = ""

    for row in rows
        instance_size, sigma, lambda, method, obj_value, iterations, runtime, mse = row

        if lambda != previous_lambda && previous_lambda != ""
            println(io, raw"        \hline")
        end

        lambda_label = @sprintf("%.2f", parse(Float64, lambda))
        method_label = method == "Conjugate Gradient" ? "CG" : "Direct"

        obj_label = @sprintf("%.3e", parse(Float64, obj_value))
        time_label = @sprintf("%.4f", parse(Float64, runtime))
        mse_label = @sprintf("%.4f", parse(Float64, mse))

        displayed_lambda = lambda == previous_lambda ? "" : lambda_label

        println(
            io,
            "        $displayed_lambda & $method_label & $obj_label & $iterations & $time_label & $mse_label ",
            row_end
        )

        previous_lambda = lambda
    end

    println(io, raw"        \hline")
    println(io, raw"    \end{tabular}")
    println(io, raw"\end{table}")
end

println("LaTeX table saved to: ", output_path)