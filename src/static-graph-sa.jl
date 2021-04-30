using ProgressBars, LightGraphs, Statistics

include("sa.jl")
include("fileio.jl")
include("net-encode-lib.jl")
include("lib-sim.jl")
include("sim-static.jl")

function sparse_graph_objective(ϵ)
    G = Graph(encoding_to_adj_matrix(ϵ))
    return if !is_connected(G)
        length(ϵ)
    else
        sum(ϵ)
    end
end

function network_neighbor(ϵ::Vector{Int8})::Vector{Int8}
    edge = rand(1:length(ϵ))
    neighbor = copy(ϵ)
    neighbor[edge] = 1 - neighbor[edge]
    neighbor
end

function make_resilient_objective()
    # NOTE: After lots of experimentation, it appears that SA needs the difference between
    # energies to be pretty high to work well. I.E. having them range from 0 to 1, isn't enough.
    rated_networks = Dict{Vector{Int8}, Float64}()
    function resilient_objective(encoding::Vector{Int8})::Float64
        if encoding in keys(rated_networks)
            return rated_networks[encoding]
        end

        # Graphs are rated on 3 different categories that each have a max of 1.0
        edge_rating = -(sum(encoding) / length(encoding))*10
        return -edge_rating
        # edge_rating = 0.0
        M = encoding_to_adj_matrix(encoding)
        G = Graph(M)
        components = connected_components(G)
        largest_component = maximum(length, components)
        num_nodes = length(vertices(G))
        max_sim_steps = 300
        num_sims = 300
        max_edges = num_nodes * (num_nodes - 1) ÷ 2
        rating = if is_connected(G)
            disease = Dizeez(3, 10, .25)
            sim_lens = run_sim_batch(M, make_fixed_starting_seir(num_nodes, 1),
                                    disease, max_sim_steps, num_sims)
            -(1 + median(sim_lens) / max_sim_steps + edge_rating)
        else
            -(length(largest_component) / num_nodes + edge_rating)
        end
        rated_networks[encoding] = rating
        rating
    end
end

function find_resilient_network(max_steps=500, T₀=100.0)
    start_time = Dates.now()
    println("Beginning.")
    # number of nodes
    N = 100
    # initial encoding
    ϵ₀ = rand((Int8(0), Int8(1)), N*(N+1)÷2)
    # initial temperature is T₀
    resilient_objective = make_resilient_objective()

    optimizer_step = make_sa_optimizer(resilient_objective,
        # make_linear_schedule(T₀, T₀/150),
        make_fast_schedule(T₀),
        network_neighbor, ϵ₀)

    best_ϵ = missing
    energies = zeros(max_steps)
    pbar = ProgressBar(1:max_steps)
    # for step in pbar
    for step = 1:max_steps
        best_ϵ, energy = optimizer_step()
        energies[step] = energy
    end
    println("Done. ($(Dates.now()-start_time))")
    PyPlot.plot(energies)
    PyPlot.show()
    best_ϵ
end

function find_sparse_connected_graph(max_steps=500, T₀=100.0)
    start_time = Dates.now()
    println("Beginning.")
    # number of nodes
    N = 50
    # initial encoding
    ϵ₀ = rand((Int8(0), Int8(1)), N*(N+1)÷2)
    # initial temperature is T₀

    optimizer_step = make_sa_optimizer(sparse_graph_objective,
        make_fast_schedule(T₀),
        network_neighbor, ϵ₀)

    best_ϵ = missing
    energies = zeros(max_steps)
    pbar = ProgressBar(1:max_steps)
    # for step in pbar
    for step = 1:max_steps
        best_ϵ, energy = optimizer_step()
        energies[step] = energy
    end
    println("Done. ($(Dates.now()-start_time))")
    PyPlot.plot(energies)
    PyPlot.show()
    best_ϵ
end