using LightGraphs, PyPlot, ProgressBars, Printf, Distributions, Random
include("sa.jl")
include("fileio.jl")
include("config-graph-optim-lib.jl")

function main()
    n_steps = 20_000
    # N = 500
    # normal_distribution = Normal(5, 3)
    # node_to_degree = shuffle(floor.(Int, clamp.(rand(normal_distribution, N), 0, 20)))
    # if sum(node_to_degree) % 2 != 0
    #     node_to_degree[argmin(node_to_degree)] += 1
    # end
    # edge_list = shuffle([node for (node, degree) in enumerate(node_to_degree) for _ in 1:degree])
    edge_list = read_edge_list("../graphs/elitist-500-500.txt")
    N = maximum(edge_list)
    optimizer_step = make_sa_optimizer(make_component_objective(), make_fast_schedule(10.0),
                                       edge_list_neighbor, edge_list)
    pbar = ProgressBar(1:n_steps)
    energies = zeros(n_steps)
    for step in pbar
        edge_list, energy = optimizer_step()
        energies[step] = energy
        set_description(pbar, string(@sprintf("Energy: %.2f", energy)))
    end

    write_adj_list("config-network-$N.txt",
                   adjacency_matrix(network_from_edge_list(edge_list)))
    plot(energies)
    PyPlot.show(block=false)
    figure()
    hist(edge_list, bins=(maximum(edge_list)-minimum(edge_list))÷4)
    PyPlot.show(block=false)
    println("Press <enter> to continue.")
    readline()
end

function make_component_objective()
    configuration_to_energy = Dict()
    function objective(edge_list::EdgeList)::Float64
        if haskey(configuration_to_energy, edge_list)
            return configuration_to_energy[edge_list]
        end

        G = network_from_edge_list(edge_list)
        largest_component = maxval(length, connected_components(G))
        dmtr = diameter(subgraph(G, largest_component))
        energy = -length(largest_component) - dmtr
        configuration_to_energy[edge_list] = energy
        energy
    end
    objective
end

function edge_list_neighbor(edge_list::Vector{Int})::Vector{Int}
    index1 = rand(1:size(edge_list, 1))
    index2 = rand(1:size(edge_list, 1))

    # to eliminate self-loops check the value adjacent to index0 to make sure edge_list[index1] != that_value
    # Note that the value of offset is the opposite of what it is in Python because of
    # differences in indexing.
    offset = if index1 % 2 == 1 1 else -1 end
    while edge_list[index1+offset] == edge_list[index2]
        index2 = rand(1:size(edge_list, 1))
    end

    new_edge_list = copy(edge_list)
    new_edge_list[index1], new_edge_list[index2] = new_edge_list[index2], new_edge_list[index1]
    new_edge_list
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end