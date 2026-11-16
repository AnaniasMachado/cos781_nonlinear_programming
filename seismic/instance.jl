
using Random
using DelimitedFiles

function generate_instance(; n_gaussians=5, n_pairs=30, seed=1234, area=(-10.0, 10.0, -10.0, 10.0), amplitude_max=0.8, sigma_range=(1.0, 3.0), depth_margin=1.0)
    @assert n_gaussians > 0
    @assert n_pairs > 0

    xmin, xmax, ymin, ymax = area

    @assert xmin < xmax && ymin < ymax
    @assert amplitude_max > 0
    @assert depth_margin > 0
    @assert 0 < sigma_range[1] <= sigma_range[2]

    rng = MersenneTwister(seed)

    randx() = xmin + (xmax - xmin) * rand(rng)
    randy() = ymin + (ymax - ymin) * rand(rng)

    # Generate the Gaussian interface parameters
    A = amplitude_max .* (2 .* rand(rng, n_gaussians) .- 1)
    a = [randx() for _ in 1:n_gaussians]
    b = [randy() for _ in 1:n_gaussians]
    sigma = sigma_range[1] .+ (sigma_range[2] - sigma_range[1]) .* rand(rng, n_gaussians)

    # Guarantee that h(x, y) > 0 everywhere
    h0 = sum(abs, A) + depth_margin

    # Generate transmitter-receiver pairs inside the area
    sources = [(randx(), randy()) for _ in 1:n_pairs]
    receivers = [(randx(), randy()) for _ in 1:n_pairs]

    # Create the dataset directory
    mkpath("./seismic/dataset")

    # Save the interface parameters
    interface_data = hcat(A, a, b, sigma)

    open("./seismic/dataset/interface.csv", "w") do io
        writedlm(io, ["A" "a" "b" "sigma"], ',')
        writedlm(io, interface_data, ',')
    end

    # Save the background depth
    writedlm("./seismic/dataset/depth.csv", ["h0"; h0], ',')

    # Save the transmitter-receiver pairs
    pairs_data = hcat(
        [s[1] for s in sources],
        [s[2] for s in sources],
        [r[1] for r in receivers],
        [r[2] for r in receivers]
    )

    open("./seismic/dataset/pairs.csv", "w") do io
        writedlm(io, ["source_x" "source_y" "receiver_x" "receiver_y"], ',')
        writedlm(io, pairs_data, ',')
    end
end

generate_instance(
    n_gaussians=10,
    n_pairs=51,
    seed=1,
    area=(-100.0, 100.0, -100.0, 100.0),
    amplitude_max=5.0,
    sigma_range=(1.0, 10.0),
    depth_margin=1.0
)