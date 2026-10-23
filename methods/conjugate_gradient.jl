
function conjugate_gradient(f, grad, hess, x0; line_search=nothing, tol=1e-8, max_iter=1000)
    x = copy(float.(x0))
    g = grad(x)
    d = -g
    k = 0

    while norm(g) > tol && k < max_iter
        k += 1

        if line_search === nothing
            t = dot(g, g) / dot(d, hess(x) * d)
        else
            t = line_search(x, d)
        end

        x = x + t * d
        g_new = grad(x)

        if norm(g_new) <= tol
            break
        end

        beta = dot(g_new, g_new) / dot(g, g)
        d = -g_new + beta * d
        g = g_new
    end

    return x, f(x), k
end
