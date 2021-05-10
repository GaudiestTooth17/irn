using PyPlot
using Dates

include("percolation-sim.jl")
include("sim-static.jl")
include("fileio.jl")

function main()
    println("Beginning.")
    M = read_adj_list("../graphs/cavemen-50-10.txt")
    N = size(M, 1)
    num_trials = 10_000
    β = .25
    τ = 5
    disease = Dizeez(1, τ, β)
    classic_start = Dates.now()
    classic_results = [calc_remaining_S_nodes(simulate(M, make_starting_seir(N, 1), disease, 300))
                       for i in 1:num_trials]
    percolation_start = Dates.now()
    println("$num_trials classic sims finished in $(percolation_start-classic_start)")
    percolation_results = [N-simulate_static(M, β, τ, 1) for i in 1:num_trials]
    println("$num_trials percolation sims finished in $(Dates.now()-percolation_start)")
    title("Classic")
    hist(classic_results)
    figure()
    title("Percolation")
    hist(percolation_results)
    PyPlot.show()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end