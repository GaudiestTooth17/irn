"""
An encoding is an array of Int8's specifying whether or not an edge exists. (1 true, 0 false)
ϵ is used for encodings.
"""


using ProgressBars, LightGraphs, Statistics, Random

include("sa.jl")
include("fileio.jl")
include("net-encode-lib.jl")
include("lib-sim.jl")
include("sim-static.jl")

Encoding = Vector{Int8}

function sparse_graph_objective(ϵ)
    G = Graph(encoding_to_adj_matrix(ϵ))
    return if !is_connected(G)
        length(ϵ)
    else
        sum(ϵ)
    end
end

function network_neighbor(ϵ::Encoding)::Encoding
    edge = rand(1:length(ϵ))
    neighbor = copy(ϵ)
    neighbor[edge] = 1 - neighbor[edge]
    neighbor
end

function farther_network_neighbor(ϵ::Encoding)::Encoding
    # edges = [rand(1:length(ϵ)) for i=1:rand(1:100)]
    edges = [rand(1:length(ϵ)) for i=1:5]
    neighbor = copy(ϵ)
    neighbor[edges] .= 1 .- neighbor[edges]
    neighbor
end

function perm_network_neighbor(ϵ::Encoding)::Encoding
    edge_index = rand(findall(!iszero, ϵ))
    no_edge_index = rand(findall(iszero, ϵ))
    neighbor = copy(ϵ)
    neighbor[edge_index] = 0
    neighbor[no_edge_index] = 1
    neighbor
end

function make_resilient_objective()
    # NOTE: After lots of experimentation, it appears that SA needs the difference between
    # energies to be pretty high to work well. I.E. having them range from 0 to 1, isn't enough.
    rated_networks = Dict{Encoding, Float64}()
    function resilient_objective(encoding::Encoding)::Float64
        if encoding in keys(rated_networks)
            return rated_networks[encoding]
        end

        # Graphs are rated on 3 different categories that each have a max of 1.0
        edge_rating = -(sum(encoding) / length(encoding))*10
        # return -edge_rating
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

function make_no_spread_objective()
    rated_networks = Dict{Encoding, Float64}()
    function objective(ϵ::Encoding)
        if ϵ in keys(rated_networks)
            return rated_networks[ϵ]
        end

        M = encoding_to_adj_matrix(ϵ)
        N = size(M, 1)
        if !is_connected(Graph(M))
            rated_networks[ϵ] = N
            return N
        end
        max_sim_steps = 100
        num_sims = 100
        num_S = run_sim_batch(M, make_fixed_starting_seir(N, 1), Dizeez(3, 10, 0.1),
                                 max_sim_steps, num_sims)
        # negate sim_lens so that we maximize that sim length
        energy = N - (sum(num_S)/num_sims)
        rated_networks[ϵ] = energy
        energy
    end
    objective
end

"""
The percolation objective removes ϕ*100 percent of the edges in a network and then
assigns an energy based off of the size of the largest component
and the total number of components.
"""
seen = 0
function make_percolation_objective(ϕ::Float64)::Function
    rated_networks = Dict{Encoding, Float64}()
    function objective(ϵ::Encoding)
        if ϵ in keys(rated_networks)
            # println("already calculated: $(rated_networks[ϵ])")
            return rated_networks[ϵ]
        end

        M = encoding_to_adj_matrix(ϵ)
        G = Graph(M)
        N = size(M, 1)
        if !is_connected(G)
            return 2*N
        end

        num_trials = 101
        energy = 0.0
        for i = 1:num_trials
            G_temp = copy(G)
            # remove edges
            shuffled_edges = shuffle(collect(edges(G_temp)))
            num_to_remove = Int(floor(length(shuffled_edges)*ϕ))
            for edge in shuffled_edges[1:num_to_remove]
                rem_edge!(G_temp, edge)
            end
            # calculate energy
            comps = connected_components(G_temp)
            biggest_comp_size = maximum(length, comps)
            energy += (N - length(comps) + biggest_comp_size)
            # energy += N - length(comps)
            # energy += biggest_comp_size
            # println("num components: $(length(comps)) biggest's size: $(length(biggest_comp))")
        end
        energy /= num_trials
        rated_networks[ϵ] = energy
        # println(energy)
        energy
    end
    objective
end

function find_resilient_network(max_steps=500, T₀=100.0)
    # Random.seed!(42)
    start_time = Dates.now()
    println("Beginning.")
    # number of nodes
    N = 100
    # initial encoding
    # ϵ₀ = rand((Int8(0), Int8(1)), N*(N-1)÷2)
    # ϵ₀ = adj_matrix_to_encoding(read_adj_list("../graphs/cavemen-10-10.txt"))
    # E is the number of edges we want
    E = Int(floor(N*(N-1)÷2 * .05))
    ϵ₀ = shuffle(Int8.([if i <= E 1 else 0 end for i=1:N*(N-1)÷2]))
    # initial temperature is T₀
    # resilient_objective = make_resilient_objective()
    # no_spread_obj = make_no_spread_objective()
    perc_obj = make_percolation_objective(.75)

    optimizer_step = make_sa_optimizer(perc_obj,
        make_linear_schedule(T₀, T₀/(max_steps/4)),
        # make_fast_schedule(T₀),
        perm_network_neighbor, ϵ₀)

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