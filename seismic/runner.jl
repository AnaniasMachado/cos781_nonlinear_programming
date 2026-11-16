using LinearAlgebra
using DelimitedFiles

include("../methods/gradient_descent.jl")
include("../methods/bfgs.jl")
include("../methods/regularized_newton.jl")
include("./util.jl")

function check_minimum(grad, hess, x; tol=1e-6)
    g = norm(grad(x))

    if !isfinite(g) || g > tol
        return "Not stationary"
    end

    eigenvalues = eigvals(Symmetric(hess(x)))

    if minimum(eigenvalues) > tol
        return "Strict local min"
    elseif minimum(eigenvalues) < -tol
        return "Not a minimum"
    else
        return "Inconclusive"
    end
end

function warmup_methods()
    f(x) = sum(x.^2)
    grad(x) = 2 .* x
    hess(x) = 2.0 * Matrix{Float64}(I, length(x), length(x))
    x0 = [1.0, 2.0]

    gradient_descent(f, grad, x0)
    bfgs(f, grad, x0)
    regularized_newton(f, grad, hess, x0)
end

function run_method(name, method, grad, hess; tol=1e-6, max_iter=10000)
    try
        result = nothing
        elapsed = @elapsed result = method()

        x, fx, k = result
        gnorm = norm(grad(x))
        status = check_minimum(grad, hess, x; tol=tol)

        if k >= max_iter && gnorm > tol
            status = "Max iterations"
        end

        return (name=name, fx=fx, iterations=k, time=elapsed, grad_norm=gnorm, status=status)
    catch e
        return (name=name, fx=NaN, iterations=missing, time=NaN, grad_norm=NaN, status="Failed: $(sprint(showerror, e))")
    end
end

function run_experiment(; v=1.0, tol=1e-6, max_iter=10000)
    # Read the interface parameters
    interface = readdlm("./seismic/dataset/interface.csv", ',', Float64; skipstart=1)

    A = interface[:, 1]
    a = interface[:, 2]
    b = interface[:, 3]
    sigma = interface[:, 4]

    # Read the background depth
    depth = readdlm("./seismic/dataset/depth.csv", ',', Any)
    h0 = parse(Float64, string(depth[2, 1]))

    # Read the transmitter-receiver pairs
    pairs = readdlm("./seismic/dataset/pairs.csv", ',', Float64; skipstart=1)

    warmup_methods()

    results = []

    for i in 1:size(pairs, 1)
        source = pairs[i, 1:2]
        receiver = pairs[i, 3:4]

        # Midpoint as the initial point
        x0 = (source + receiver) / 2

        f(x) = travel_time(x[1], x[2], source, receiver, v, h0, A, a, b, sigma)
        grad(x) = grad_travel_time(x[1], x[2], source, receiver, v, h0, A, a, b, sigma)
        hess(x) = hessian_travel_time(x[1], x[2], source, receiver, v, h0, A, a, b, sigma)

        methods = [
            ("Gradient Descent", () -> gradient_descent(f, grad, x0; tol=tol, max_iter=max_iter)),
            ("BFGS", () -> bfgs(f, grad, x0; tol=tol, max_iter=max_iter)),
            ("Regularized Newton", () -> regularized_newton(f, grad, hess, x0; tol=tol, max_iter=max_iter))
        ]

        for (name, method) in methods
            result = run_method(name, method, grad, hess; tol=tol, max_iter=max_iter)
            push!(results, (i, result.name, result.fx, result.iterations, result.time, result.grad_norm, result.status))
        end
    end

    # Create the results directory
    mkpath("./seismic/results")

    # Save the results
    open("./seismic/results/results.csv", "w") do io
        writedlm(io, ["pair_index" "method" "objective_value" "iterations" "time_seconds" "gradient_norm" "status"], ',')

        for result in results
            writedlm(io, reshape(collect(result), 1, :), ',')
        end
    end
end

run_experiment(v=1.0, tol=1e-6, max_iter=10000)