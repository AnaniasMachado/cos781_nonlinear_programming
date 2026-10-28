using LinearAlgebra
using SparseArrays
using Printf
using Random
using FileIO
using Images

include("util.jl")
include("../methods/conjugate_gradient.jl")

image_path = "image_denoising/data/tabby.jpg"
results_dir = "image_denoising/results"

sigma = 15.0
lambdas = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
seed = 1
tol = 1e-6
max_iter = 10000
warmup_size = 250

mkpath(results_dir)

Random.seed!(seed)

original = 255.0 .* Float64.(getproperty.(Gray.(load(image_path)), :val))
m, n = size(original)

noisy = original + sigma .* randn(m, n)

x0 = rand(m * n)

mse(A, B) = sum(abs2, A - B) / length(A)
to_image(A) = Gray.(clamp.(A ./ 255.0, 0.0, 1.0))

save(joinpath(results_dir, "original.png"), to_image(original))
save(joinpath(results_dir, "noisy.png"), to_image(noisy))

warmup_noisy = rand(warmup_size, warmup_size)

warmup_f, warmup_grad, warmup_hess, warmup_line_search, warmup_H, warmup_b =
    denoising_problem(warmup_noisy, first(lambdas))

warmup_x0 = rand(length(warmup_b))

conjugate_gradient(
    warmup_f, warmup_grad, warmup_hess, warmup_x0;
    line_search=warmup_line_search,
    tol=tol,
    max_iter=max_iter
)

warmup_H \ warmup_b

csv_path = joinpath(results_dir, "statistics.csv")

open(csv_path, "w") do io
    println(io, "instance_size,sigma,lambda,method,obj_value,iterations,runtime,mse")

    for lambda in lambdas
        f, grad, hess, line_search, H, b =
            denoising_problem(noisy, lambda)

        x_cg = nothing
        fx_cg = NaN
        k = 0

        time_cg = @elapsed begin
            x_cg, fx_cg, k = conjugate_gradient(
                f, grad, hess, x0;
                line_search=line_search,
                tol=tol,
                max_iter=max_iter
            )
        end

        x_direct = nothing
        time_direct = @elapsed x_direct = H \ b

        recovered_cg = reshape(x_cg, m, n)
        recovered_direct = reshape(x_direct, m, n)

        mse_cg = mse(recovered_cg, original)
        mse_direct = mse(recovered_direct, original)

        lambda_label = string(lambda)

        save(
            joinpath(results_dir, "recovered_cg_lambda_$(lambda_label).png"),
            to_image(recovered_cg)
        )

        save(
            joinpath(results_dir, "recovered_direct_lambda_$(lambda_label).png"),
            to_image(recovered_direct)
        )

        instance_size = "$(m)x$(n)"

        @printf(
            io,
            "%s,%.2f,%.6f,%s,%.10e,%d,%.10e,%.10e\n",
            instance_size, sigma, lambda, "Conjugate Gradient",
            fx_cg, k, time_cg, mse_cg
        )

        @printf(
            io,
            "%s,%.2f,%.6f,%s,%.10e,,%.10e,%.10e\n",
            instance_size, sigma, lambda, "Direct Solve",
            f(x_direct), time_direct, mse_direct
        )

        @printf(
            "Lambda: %.4f | CG: %d iterations, %.4f s, MSE %.4f | Direct: %.4f s, MSE %.4f\n",
            lambda, k, time_cg, mse_cg, time_direct, mse_direct
        )
    end
end

@printf("\nImage size: %d x %d\n", m, n)
@printf("Sigma: %.2f\n", sigma)
@printf("Seed: %d\n", seed)
@printf("Noisy MSE: %.6f\n", mse(noisy, original))
println("Statistics saved to: ", csv_path)