using LinearAlgebra
using SparseArrays

function difference_operators(m, n)
    N = m * n

    rows_v = Int[]
    cols_v = Int[]
    vals_v = Float64[]

    rows_h = Int[]
    cols_h = Int[]
    vals_h = Float64[]

    for j in 1:n, i in 1:m
        p = i + (j - 1) * m

        if i < m
            push!(rows_v, p, p)
            push!(cols_v, p, p + 1)
            push!(vals_v, -1.0, 1.0)
        end

        if j < n
            push!(rows_h, p, p)
            push!(cols_h, p, p + m)
            push!(vals_h, -1.0, 1.0)
        end
    end

    Dv = sparse(rows_v, cols_v, vals_v, N, N)
    Dh = sparse(rows_h, cols_h, vals_h, N, N)

    return Dv, Dh
end

function denoising_problem(noisy, lambda)
    m, n = size(noisy)
    b = vec(Float64.(noisy))

    Dv, Dh = difference_operators(m, n)

    N = length(b)
    H = spdiagm(0 => ones(N)) +
        2 * lambda * (Dv' * Dv + Dh' * Dh)

    function objective(x)
        return 0.5 * dot(x - b, x - b) +
               lambda * (dot(Dv * x, Dv * x) + dot(Dh * x, Dh * x))
    end

    gradient(x) = H * x - b
    hessian(x) = H

    function line_search(x, d)
        return -dot(gradient(x), d) / dot(d, H * d)
    end

    return objective, gradient, hessian, line_search, H, b
end