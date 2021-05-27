"""
An encoding is an array of Int8's specifying whether or not an edge exists. (1 true, 0 false)
ϵ is used for encodings.
"""


using ProgressBars, LightGraphs, Statistics, Random, Printf

include("sa.jl")
include("fileio.jl")
include("net-encode-lib.jl")
include("lib-sim.jl")
include("percolation-sim.jl")
include("betweeness.jl")

"""
The goal is to find a connected graph with the fewest edges possible.
"""
function sparse_graph_objective(ϵ)
    G = Graph(encoding_to_adj_matrix(ϵ))
    return if !is_connected(G)
        length(ϵ)
    else
        sum(ϵ)
    end
end

"""
This neighbor function chooses an edge to toggle, toggles it, and returns the new encoding.
It also keeps track of the edges it has tried toggling. If all the edges get tried for
a certain encoding, it will always return that encoding.
"""
function make_neighbor_explorer()::Function
    encoding_to_tried_edges = Dict{Encoding, Set{Int}}()
    function network_neighbor(ϵ::Encoding)::Encoding
        if !(ϵ in keys(encoding_to_tried_edges))
            encoding_to_tried_edges[ϵ] = Set{Int}()
        # if all of the edges have already been tried, just return the original encoding
        elseif length(encoding_to_tried_edges[ϵ]) == length(ϵ)
            return ϵ
        end
        edge = rand(setdiff(Set(1:length(ϵ)), encoding_to_tried_edges[ϵ]))
        push!(encoding_to_tried_edges[ϵ], edge)
        neighbor = copy(ϵ)
        neighbor[edge] = 1 - neighbor[edge]
        neighbor
    end
    network_neighbor
end

"""
This is a simplified version of the closure from make_neighbor_explorer that doesn't
keep track of what has been tried.
"""
function network_neighbor(ϵ::Encoding)::Encoding
    edge = rand(1:length(ϵ))
    neighbor = copy(ϵ)
    neighbor[edge] = 1 - neighbor[edge]
    neighbor
end

"""
Similar to network_neighbor, but toggles 5 edges.
"""
function farther_network_neighbor(ϵ::Encoding)::Encoding
    # edges = [rand(1:length(ϵ)) for i=1:rand(1:100)]
    edges = [rand(1:length(ϵ)) for i=1:5]
    neighbor = copy(ϵ)
    neighbor[edges] .= 1 .- neighbor[edges]
    neighbor
end

"""
Instead of choosing an edge to toggle, this finds an active edge, disables it, and enables
some disabled edge.
"""
function perm_network_neighbor(ϵ::Encoding)::Encoding
    edge_index = rand(findall(!iszero, ϵ))
    no_edge_index = rand(findall(iszero, ϵ))
    neighbor = copy(ϵ)
    neighbor[edge_index] = 0
    neighbor[no_edge_index] = 1
    neighbor
end

function make_resilient_objective()
    rated_networks = Dict{Encoding, Float64}()
    function resilient_objective(encoding::Encoding)::Float64
        if encoding in keys(rated_networks)
            return rated_networks[encoding]
        end

        # Graphs are rated on 3 different categories that each have a max of 1.0
        M = encoding_to_adj_matrix(encoding)
        G = Graph(M)
        components = connected_components(G)
        largest_component = maximum(length, components)
        N = length(vertices(G))
        num_sims = 500

        max_edges = N * (N - 1) ÷ 2
        E = length(edges(G))
        edge_rating = N - (E/max_edges) * N

        rating = if is_connected(G)
            num_infected = [simulate_static(M, .1, 5, .01) for i=1:num_sims]
            sum(num_infected) / num_sims + edge_rating
        else
            2*N - length(largest_component) + edge_rating
        end
        rated_networks[encoding] = rating
        rating
    end
end

"""
The only goal is to stop infection.
"""
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
Maximize the betweeness centrality of num_edges edges. In practice it seems like it makes
2*num_edges have high centrality.
"""
function make_high_betweeness_objective(num_edges::Int, diameter_weight::Float64)::Function
    rated_networks = Dict{Encoding, Float64}()
    function objective(ϵ::Encoding)::Float64
        if haskey(rated_networks, ϵ)
            return rated_networks[ϵ]
        end

        G = Graph(encoding_to_adj_matrix(ϵ))
        if !is_connected(G)
            rated_networks[ϵ] = .1
            return rated_networks[ϵ]
        end
        edge_betweenesses = collect(values(edge_betweeness(G)))
        sort!(edge_betweenesses, rev=true)
        rated_networks[ϵ] = -sum(edge_betweenesses[1:num_edges]) - diameter(G)*diameter_weight
        rated_networks[ϵ]
    end
    objective
end

"""
Calculate the proportion of neighbors u has that v also has.
"""
function calc_prop_common_neighbors(G::Graph, u::Int, v::Int)::Float64
    u_neighbors = Set(neighbors(G, u))
    v_neighbors = Set(neighbors(G, v))
    n_common_neighbors = length(intersect(u_neighbors, v_neighbors))
    n_common_neighbors / length(u_neighbors)
end

"""
This objective is for minimizing the number of common neighbors a few vertices have.
The hope is to form a few cliques in the network.
num_edges is the number of edges to report on.
diameter_weight represents how important the diameter of the network is. 0.1 is quite high. 
"""
function make_clique_objective(num_edges::Int, diameter_weight::Float64)::Function
    rated_networks = Dict{Encoding, Float64}()
    function objective(ϵ::Encoding)::Float64
        if haskey(rated_networks, ϵ)
            return rated_networks[ϵ]
        end

        G = Graph(encoding_to_adj_matrix(ϵ))
        if !is_connected(G)
            rated_networks[ϵ] = 1.0
            return rated_networks[ϵ]
        end
        edge_strength = [calc_prop_common_neighbors(G, src(e), dst(e)) for e in edges(G)]
        sort!(edge_strength)
        rated_networks[ϵ] = sum(edge_strength[1:num_edges]) - diameter(G)*diameter_weight
        rated_networks[ϵ]
    end
    objective
end

function find_resilient_network(max_steps=500, T₀=100.0)
    # Random.seed!(42)
    start_time = Dates.now()
    println("Beginning.")
    # number of nodes
    # N = 100
    # initial encoding
    # ϵ₀ = rand((Int8(0), Int8(1)), N*(N-1)÷2)
    ϵ₀ = adj_matrix_to_encoding(read_adj_list("../graphs/elitist-500-500.txt"))
    # E is the number of edges we want
    # E = Int(floor(N*(N-1)÷2 * .03))
    # ϵ₀ = shuffle(Int8.([if i <= E 1 else 0 end for i=1:N*(N-1)÷2]))
    # initial temperature is T₀
    # objective = make_resilient_objective()
    objective = make_high_betweeness_objective(2, .05)

    optimizer_step = make_sa_optimizer(objective,
        # make_linear_schedule(T₀, T₀/(max_steps/4)),
        make_fast_schedule(T₀),
        # perm_network_neighbor,
        make_neighbor_explorer(),
        ϵ₀)

    best_ϵ = missing
    energies = zeros(max_steps)
    pbar = ProgressBar(1:max_steps)
    for step in pbar
    # for step = 1:max_steps
        best_ϵ, energy = optimizer_step()
        energies[step] = energy
        set_description(pbar, string(@sprintf("Energy: %.2f", energy)))
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

if abspath(PROGRAM_FILE) == @__FILE__
    ϵ = find_resilient_network(2500, 100.0)
    M = encoding_to_adj_matrix(ϵ)
    write_adj_list("annealed.txt", M)
end
