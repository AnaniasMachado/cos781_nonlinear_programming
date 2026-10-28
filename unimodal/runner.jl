include("../methods/unimodal.jl")

f1(x) = (x - 2.0)^2 + 1.0
f2(x) = abs(x + 1.5) + 0.5
f3(x) = (x - 3.0)^4 + 2.0 * (x - 3.0)^2 + 4.0

problems = [
    ("Function 1", f1, -5.0, 8.0),
    ("Function 2", f2, -6.0, 5.0),
    ("Function 3", f3, -2.0, 7.0)
]

algorithms = [
    ("Ternary search", ternary_search),
    ("Bisection search", bisection_search),
    ("Golden search", golden_search)
]

tol = 1e-3

for (problem_name, f, a, b) in problems
    println("="^60)
    println(problem_name)
    println("Interval: [$a, $b]")
    println("Tolerance: $tol")
    println("="^60)

    for (algorithm_name, algorithm) in algorithms
        x, fx, k = algorithm(f, a, b; tol=tol)

        println(algorithm_name)
        println("  Solution:        ", x)
        println("  Objective value: ", fx)
        println("  Iterations:      ", k)
        println()
    end
end