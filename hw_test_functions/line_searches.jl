using LinearAlgebra

include("objective_functions.jl")

function golden_line_search(f, x, d; tol=1e-8, max_expand=50)
    phi(t) = f(x + t * d)

    a = 0.0
    b = 1.0
    k = 0

    while phi(b) < phi(a) && k < max_expand
        a, b = b, 2b
        k += 1
    end

    return golden_search(phi, 0.0, b; tol=tol)[1]
end


sphere_line_search(x, d) = -dot(sphere_grad(x), d) / (2 * dot(d, d))

matyas_line_search(x, d) = -dot(matyas_grad(x), d) / dot(d, matyas_hess(x) * d)

booth_line_search(x, d) = -dot(booth_grad(x), d) / dot(d, booth_hess(x) * d)


himmelblau_line_search(x, d) = golden_line_search(himmelblau, x, d)

rastrigin_line_search(x, d) = golden_line_search(rastrigin, x, d)

styblinski_tang_line_search(x, d) = golden_line_search(styblinski_tang, x, d)

rosenbrock_line_search(x, d) = golden_line_search(rosenbrock, x, d)

beale_line_search(x, d) = golden_line_search(beale, x, d)
