
function h(x, y, h0, A, a, b, sigma)
    n = length(A)

    @assert length(a) == length(b) == length(sigma) == n
    @assert all(sigma .> 0) "All widths must be positive."
    @assert h0 > sum(abs, A) "The interface must stay underground."

    return h0 + sum(
        A[i] * exp(-((x - a[i])^2 + (y - b[i])^2) / (2sigma[i]^2))
        for i in eachindex(A)
    )
end

function grad_h(x, y, A, a, b, sigma)
    hx = 0.0
    hy = 0.0

    for i in eachindex(A)
        dx = x - a[i]
        dy = y - b[i]
        s2 = sigma[i]^2

        g = A[i] * exp(-(dx^2 + dy^2) / (2s2))

        hx -= g * dx / s2
        hy -= g * dy / s2
    end

    return [hx, hy]
end

function travel_time(x, y, source, receiver, v, h0, A, a, b, sigma)
    @assert v > 0 "Velocity must be positive."

    z = h(x, y, h0, A, a, b, sigma)

    ds = sqrt((x - source[1])^2 + (y - source[2])^2 + z^2)
    dr = sqrt((x - receiver[1])^2 + (y - receiver[2])^2 + z^2)

    return (ds + dr) / v
end

function grad_travel_time(x, y, source, receiver, v, h0, A, a, b, sigma)
    @assert v > 0 "Velocity must be positive."

    z = h(x, y, h0, A, a, b, sigma)
    hx, hy = grad_h(x, y, A, a, b, sigma)

    ds = sqrt((x - source[1])^2 + (y - source[2])^2 + z^2)
    dr = sqrt((x - receiver[1])^2 + (y - receiver[2])^2 + z^2)

    tx = ((x - source[1] + z * hx) / ds +
          (x - receiver[1] + z * hx) / dr) / v

    ty = ((y - source[2] + z * hy) / ds +
          (y - receiver[2] + z * hy) / dr) / v

    return [tx, ty]
end

function hessian_h(x, y, A, a, b, sigma)
    hxx = 0.0
    hxy = 0.0
    hyy = 0.0

    for i in eachindex(A)
        dx = x - a[i]
        dy = y - b[i]
        s2 = sigma[i]^2

        g = A[i] * exp(-(dx^2 + dy^2) / (2s2))

        hxx += g * (dx^2 / s2^2 - 1 / s2)
        hxy += g * dx * dy / s2^2
        hyy += g * (dy^2 / s2^2 - 1 / s2)
    end

    return [hxx hxy; hxy hyy]
end

function hessian_travel_time(x, y, source, receiver, v, h0, A, a, b, sigma)
    @assert v > 0 "Velocity must be positive."

    z = h(x, y, h0, A, a, b, sigma)
    hx, hy = grad_h(x, y, A, a, b, sigma)

    H = hessian_h(x, y, A, a, b, sigma)
    hxx = H[1, 1]
    hxy = H[1, 2]
    hyy = H[2, 2]

    ds = sqrt((x - source[1])^2 + (y - source[2])^2 + z^2)
    dr = sqrt((x - receiver[1])^2 + (y - receiver[2])^2 + z^2)

    txx = ((1 + hx^2 + z * hxx) / ds -
           (x - source[1] + z * hx)^2 / ds^3 +
           (1 + hx^2 + z * hxx) / dr -
           (x - receiver[1] + z * hx)^2 / dr^3) / v

    txy = ((hx * hy + z * hxy) / ds -
           (x - source[1] + z * hx) * (y - source[2] + z * hy) / ds^3 +
           (hx * hy + z * hxy) / dr -
           (x - receiver[1] + z * hx) * (y - receiver[2] + z * hy) / dr^3) / v

    tyy = ((1 + hy^2 + z * hyy) / ds -
           (y - source[2] + z * hy)^2 / ds^3 +
           (1 + hy^2 + z * hyy) / dr -
           (y - receiver[2] + z * hy)^2 / dr^3) / v

    return [txx txy; txy tyy]
end