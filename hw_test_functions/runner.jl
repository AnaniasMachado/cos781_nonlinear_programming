
using LinearAlgebra
using Printf
using Random

include("line_searches.jl")

include("../methods/gradient_descent.jl")
include("../methods/conjugate_gradient.jl")
include("../methods/bfgs.jl")
include("../methods/regularized_newton.jl")
include("../methods/newton.jl")

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
    line_search(x, d) = -dot(grad(x), d) / dot(d, hess(x) * d)
    x0 = [1.0, 2.0]

    gradient_descent(f, grad, x0; line_search=line_search)
    gradient_descent(f, grad, x0)

    conjugate_gradient(f, grad, hess, x0; line_search=line_search)

    bfgs(f, grad, x0; line_search=line_search)
    bfgs(f, grad, x0)

    regularized_newton(f, grad, hess, x0; line_search=line_search)
    regularized_newton(f, grad, hess, x0)

    newton(f, grad, hess, x0)
end

function run_method(name, method, grad, hess; tol=1e-8, max_iter=1000)
    try
        result = nothing
        time = @elapsed result = method()

        x, fx, k = result
        gnorm = norm(grad(x))
        status = check_minimum(grad, hess, x; tol=tol)

        if k >= max_iter && gnorm > tol
            status = "Max iterations"
        end

        @printf(
            "%-22s %14.6e %12d %14.6e %14.6e  %s\n",
            name, fx, k, time, gnorm, status
        )

        println("Solution: ", x)

        return (
            name=name,
            x=x,
            fx=fx,
            iterations=k,
            time=time,
            grad_norm=gnorm,
            status=status
        )
    catch e
        println(name, " failed: ", e)
        return nothing
    end
end

function run_comparison(name, f, grad, hess, line_search, x0; quadratic=false, tol=1e-8, max_iter=1000)
    println("\n", "="^110)
    println("Function: ", name)
    println("Initial point: ", x0)
    println("="^110)

    @printf(
        "%-22s %14s %12s %14s %14s  %s\n",
        "Method", "f(x)", "Iterations", "Time (s)", "Grad norm", "Status"
    )

    println("-"^110)

    methods = [
        ("Gradient Descent", () -> gradient_descent(f, grad, x0; line_search=line_search, tol=tol, max_iter=max_iter)),
        ("BFGS", () -> bfgs(f, grad, x0; line_search=line_search, tol=tol, max_iter=max_iter)),
        ("Regularized Newton", () -> regularized_newton(f, grad, hess, x0; line_search=line_search, tol=tol, max_iter=max_iter)),
        ("Pure Newton", () -> newton(f, grad, hess, x0; tol=tol, max_iter=max_iter))
    ]

    if quadratic
        push!(methods, (
            "Conjugate Gradient",
            () -> conjugate_gradient(f, grad, hess, x0; line_search=line_search, tol=tol, max_iter=max_iter)
        ))
    end

    results = []

    for (method_name, method) in methods
        result = run_method(method_name, method, grad, hess; tol=tol, max_iter=max_iter)

        if result !== nothing
            push!(results, result)
        end
    end

    return results
end

Random.seed!(1)

tol = 1e-6
max_iter = 10000

tests = [
    ("Sphere", sphere, sphere_grad, sphere_hess, sphere_line_search, rand(2), true),
    ("Matyas", matyas, matyas_grad, matyas_hess, matyas_line_search, rand(2), true),
    ("Booth", booth, booth_grad, booth_hess, booth_line_search, rand(2), true),
    ("Himmelblau", himmelblau, himmelblau_grad, himmelblau_hess, nothing, rand(2), false),
    ("Rastrigin", rastrigin, rastrigin_grad, rastrigin_hess, nothing, rand(2), false),
    ("Styblinski-Tang", styblinski_tang, styblinski_tang_grad, styblinski_tang_hess, nothing, rand(2), false),
    ("Rosenbrock", rosenbrock, rosenbrock_grad, rosenbrock_hess, nothing, rand(2), false),
    ("Beale", beale, beale_grad, beale_hess, nothing, rand(2), false)
]

warmup_methods()

all_results = Dict()

for (name, f, grad, hess, line_search, x0, quadratic) in tests
    all_results[name] = run_comparison(
        name, f, grad, hess, line_search, x0;
        quadratic=quadratic,
        tol=tol,
        max_iter=max_iter
    )
end
