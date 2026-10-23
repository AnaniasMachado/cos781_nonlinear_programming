
function newton(f, grad, hess, x0; tol=1e-8, max_iter=1000)
    x = copy(float.(x0))
    k = 0

    while norm(grad(x)) > tol && k < max_iter
        k += 1
        d = -hess(x) \ grad(x)
        x = x + d
    end

    return x, f(x), k
end
