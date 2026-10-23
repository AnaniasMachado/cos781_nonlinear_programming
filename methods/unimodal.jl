function ternary_search(f, a, b; tol=1e-8)
    k = 0
    while b - a > tol
        k += 1
        x1 = a + (b - a) / 3
        x2 = b - (b - a) / 3

        if f(x1) <= f(x2)
            b = x2
        else
            a = x1
        end
    end

    x = (a + b) / 2
    return x, f(x), k
end


function bisection_search(f, a, b; tol=1e-8)
    k = 0
    while b - a > tol
        k += 1
        m = (a + b) / 2
        delta = min(tol / 2, (b - a) / 4)

        x1 = m - delta
        x2 = m + delta

        if f(x1) <= f(x2)
            b = x2
        else
            a = x1
        end
    end

    x = (a + b) / 2
    return x, f(x), k
end


function golden_search(f, a, b; tol=1e-8)
    ratio = (sqrt(5) - 1) / 2

    x1 = b - ratio * (b - a)
    x2 = a + ratio * (b - a)

    f1 = f(x1)
    f2 = f(x2)

    k = 0
    while b - a > tol
        k += 1
        if f1 <= f2
            b = x2
            x2 = x1
            f2 = f1
            x1 = b - ratio * (b - a)
            f1 = f(x1)
        else
            a = x1
            x1 = x2
            f1 = f2
            x2 = a + ratio * (b - a)
            f2 = f(x2)
        end
    end

    x = (a + b) / 2
    return x, f(x), k
end