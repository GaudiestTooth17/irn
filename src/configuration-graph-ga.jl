using ProgressBars, PyPlot
include("ga.jl")
include("config-graph-optim-lib.jl")

function main()
    println("Hello world!")
end

function make_component_fitness(γ::EdgeList)
    genotype_to_fitness = Dict()
    function objective(edge_list::Vector{Int})::Float64
        if haskey(genotype_to_fitness, edge_list)
            return genotype_to_fitness[edge_list]
        end

        G = network_from_edge_list(edge_list)
        largest_component = maxval(length, connected_components(G))
        dmtr = diameter(subgraph(G, largest_component))
        energy = dmtr + length(largest_component)
        genotype_to_fitness[edge_list] = energy
        energy
    end
    objective
end

function make_next_gen(parent_selection_fn, crossover_fn, mutate_fn!)::Function
    function next_gen(max_fitness, population)::Vector{Genotype}
        parent_pairs = (parent_selection_fn(max_fitness, population) for _=1:length(population)÷2)
        children = collect(flatten((crossover_fn(parents...) for parents in parent_pairs)))
        mutate_fn!(children, .01)
        children
    end
    next_gen
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end