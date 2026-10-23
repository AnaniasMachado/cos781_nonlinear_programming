using LinearAlgebra

sphere(x) = sum(x.^2)

sphere_grad(x) = 2 .* x

sphere_hess(x) = 2.0 * Matrix{Float64}(I, length(x), length(x))


matyas(x) = 0.26 * (x[1]^2 + x[2]^2) - 0.48 * x[1] * x[2]

matyas_grad(x) = [
    0.52 * x[1] - 0.48 * x[2],
    0.52 * x[2] - 0.48 * x[1]
]

matyas_hess(x) = [0.52 -0.48; -0.48 0.52]


booth(x) = (x[1] + 2x[2] - 7)^2 + (2x[1] + x[2] - 5)^2

booth_grad(x) = [
    2 * (x[1] + 2x[2] - 7) + 4 * (2x[1] + x[2] - 5),
    4 * (x[1] + 2x[2] - 7) + 2 * (2x[1] + x[2] - 5)
]

booth_hess(x) = [10.0 8.0; 8.0 10.0]


himmelblau(x) = (x[1]^2 + x[2] - 11)^2 + (x[1] + x[2]^2 - 7)^2

himmelblau_grad(x) = [
    4x[1] * (x[1]^2 + x[2] - 11) + 2 * (x[1] + x[2]^2 - 7),
    2 * (x[1]^2 + x[2] - 11) + 4x[2] * (x[1] + x[2]^2 - 7)
]

himmelblau_hess(x) = [
    12x[1]^2 + 4x[2] - 42    4x[1] + 4x[2];
    4x[1] + 4x[2]            4x[1] + 12x[2]^2 - 26
]


rastrigin(x) = 10 * length(x) + sum(x.^2 .- 10 .* cos.(2pi .* x))

rastrigin_grad(x) = 2 .* x .+ 20pi .* sin.(2pi .* x)

rastrigin_hess(x) = Matrix(Diagonal(2 .+ 40pi^2 .* cos.(2pi .* x)))


styblinski_tang(x) = sum(x.^4 .- 16 .* x.^2 .+ 5 .* x) / 2

styblinski_tang_grad(x) = 2 .* x.^3 .- 16 .* x .+ 2.5

styblinski_tang_hess(x) = Matrix(Diagonal(6 .* x.^2 .- 16))


rosenbrock(x) = 100 * (x[2] - x[1]^2)^2 + (1 - x[1])^2

rosenbrock_grad(x) = [
    -400x[1] * (x[2] - x[1]^2) + 2 * (x[1] - 1),
    200 * (x[2] - x[1]^2)
]

rosenbrock_hess(x) = [
    1200x[1]^2 - 400x[2] + 2    -400x[1];
    -400x[1]                    200.0
]


beale(x) = (1.5 - x[1] + x[1]*x[2])^2 +
           (2.25 - x[1] + x[1]*x[2]^2)^2 +
           (2.625 - x[1] + x[1]*x[2]^3)^2

function beale_grad(x)
    a, b = x
    r1 = 1.5 - a + a*b
    r2 = 2.25 - a + a*b^2
    r3 = 2.625 - a + a*b^3

    return [
        2 * (r1*(b-1) + r2*(b^2-1) + r3*(b^3-1)),
        2 * (r1*a + r2*2a*b + r3*3a*b^2)
    ]
end

function beale_hess(x)
    a, b = x
    r1 = 1.5 - a + a*b
    r2 = 2.25 - a + a*b^2
    r3 = 2.625 - a + a*b^3

    h11 = 2 * ((b-1)^2 + (b^2-1)^2 + (b^3-1)^2)

    h12 = 2 * (
        a*(b-1) + r1 +
        2a*b*(b^2-1) + 2b*r2 +
        3a*b^2*(b^3-1) + 3b^2*r3
    )

    h22 = 2 * (
        a^2 + (2a*b)^2 + (3a*b^2)^2 +
        2a*r2 + 6a*b*r3
    )

    return [h11 h12; h12 h22]
end
