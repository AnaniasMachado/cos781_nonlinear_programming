
function bfgs(f, grad, x0; line_search=nothing, tol=1e-8, max_iter=1000, alpha=1.0, beta=0.5, c=1e-4)
    x = copy(float.(x0))
    n = length(x)
    H = Matrix{Float64}(I, n, n)
    k = 0

    while norm(grad(x)) > tol && k < max_iter
        k += 1
        g = grad(x)
        d = -H * g

        if line_search === nothing
            t = alpha
            while f(x + t * d) > f(x) + c * t * dot(g, d)
                t *= beta
            end
        else
            t = line_search(x, d)
        end

        s = t * d
        x_new = x + s
        y = grad(x_new) - g

        rho = 1 / dot(y, s)
        V = Matrix{Float64}(I, n, n) - rho * s * y'
        H = V * H * V' + rho * s * s'

        x = x_new
    end

    return x, f(x), k
end
