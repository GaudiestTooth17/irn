using Plots
using LightGraphs, ProgressBars, Printf

include("ga.jl")
include("fileio.jl")

function genotype_to_adj_matrix(genotype::Vector{Int8})::Matrix{Int8}
    # if calculating num_nodes fails, it is probably because the genotype
    # just doesn't have the right number of entries.
    # This is basically the inverse of the triangular sum.
    num_nodes = Int(sqrt(2*length(genotype)+.25) + .5)
    adj_matrix = zeros(Int8, num_nodes, num_nodes)
    current_edge = 1
    for i = 1:size(adj_matrix, 1)
        for j = i+1:size(adj_matrix, 2)
            adj_matrix[i, j] = genotype[current_edge]
            adj_matrix[j, i] = genotype[current_edge]
            current_edge += 1
        end
    end
    adj_matrix
end

function adj_matrix_to_genotype(M::AbstractMatrix)::Vector{Int8}
    num_nodes = size(M, 1)
    # genotype contains an entry for every edge on the graph
    genotype = zeros(Int8, (num_nodes*(num_nodes-1) ÷ 2))
    current_loc = 1
    for i = 1:size(M, 1)
        for j = i+1:size(M, 2)
            genotype[current_loc] = M[i, j]
            current_loc += 1
        end
    end
    genotype
end

function graph_fitness(genotype::Vector{Int8})::Float64
    # todo: write a fitness function that lets the disease spread to all the nodes,
    # but rewards graphs that make the disease take longer to spread
    M = genotype_to_adj_matrix(genotype)
    G = Graph(M)
    # Nothing useful seems to be happening when non connected graphs get immediately shut down
    # if !is_connected(G)
    #     return 0
    # end

    # disease = Dizeez(3, 5, .3)
    # num_nodes = size(M, 1)
    # num_S_nodes = run_sim_batch(M, make_starting_seir(num_nodes, 5), disease, 250, 100)
    # avg_num_S_nodes = sum(num_S_nodes) / length(num_S_nodes)

    # avg_num_S_nodes / length(connected_components(G))
    components = connected_components(G)
    largest_component = maximum(length, components)
    num_nodes = length(vertices(G))
    if is_connected(G)
        disease = Dizeez(3, 5, .3)
        num_nodes = size(M, 1)
        num_S_nodes = run_sim_batch(M, make_starting_seir(num_nodes, 5), disease, 250, 100)
        avg_num_S_nodes = sum(num_S_nodes) / length(num_S_nodes)
        return 1 + avg_num_S_nodes / num_nodes
    else
        return length(largest_component) / num_nodes # max is 1
    end
end

function mutate_graph!(population::Vector{Vector{T}}, prob::Float64) where T
    for i = 1:length(population)
        for j = 1:length(population[i])
            if prob < rand()
                population[i][j] = 1 - population[i][j]
            end
        end
    end
end

function next_graph_generation(max_fitness::Float64,
    population::Vector{Tuple{Float64, Vector{T}}})::Vector{Vector{T}} where T

    parent_pairs = (roullete_wheel_selection(max_fitness, population)
                    for i=1:div(length(population), 2))
    children = collect(flatten((crossover(parents...)
                                for parents in parent_pairs)))
    mutate_graph!(children, .0)
    children
end

function make_observer(max_steps::Int)
    max_fitnesses = zeros(max_steps)
    pop_diversity = zeros(max_steps)
    steps_taken = 0
    function observe(fitness_to_genotype, most_fit)
        steps_taken += 1
        max_fitnesses[steps_taken] = most_fit[1]
        unique_genotypes = Set(map(x->x[2], fitness_to_genotype))
        pop_diversity[steps_taken] = length(unique_genotypes) / length(fitness_to_genotype)
    end
    function report()
        max_fitnesses[1:steps_taken], pop_diversity[1:steps_taken]
    end
    observe, report
end

function make_starting_population(num_nodes::Int, pop_size::Int)
    # generate values for each of the edges
    # Only generate half-ish of the edges because the matrix is symmetric
    # subtract pop_size from the amount to account for zeros along the diagonal (no self-loops)
    # pop = [adj_matrix_to_genotype(adjacency_matrix(barabasi_albert(num_nodes, 3, 3, seed=0)))
    #        for i=1:pop_size]
    pop = [Int8.(abs.(rand(Int8, div(num_nodes*(num_nodes+1), 2) - num_nodes)) .% 2)
           for i=1:pop_size]
    # for (i, gen) in enumerate(pop)
    #     if !is_connected(Graph(genotype_to_adj_matrix(gen)))
    #         println("Graph $i is not connected.")
    #     end
    # end
    pop
end

if abspath(PROGRAM_FILE) == @__FILE__
    max_steps = 10
    num_nodes = 500
    pop_size = 40
    max_fitness = Float64(2)  # Unachievable, but it means that no nodes got infected
    starting_pop = make_starting_population(num_nodes, pop_size)
    observe, report = make_observer(max_steps)
    take_step = make_optimizer(graph_fitness, next_graph_generation, max_fitness, starting_pop)

    progress_bar = ProgressBar(1:max_steps)
    best_graph = missing
    for i in progress_bar
        global best_graph
        (best_fitness, best_graph), fitness_to_answer = take_step()
        observe(fitness_to_answer, (best_fitness, best_graph))
        set_description(progress_bar, string(@sprintf("Best Fitness %.3f", best_fitness)))
    end

    max_fitness, pop_diversity = report()
    println("Max fitnesses")
    println(max_fitness)
    println("Population Diversity")
    println(pop_diversity)
    display(plot(max_fitness))
    readline()
    display(plot(pop_diversity))
    readline()
    M = genotype_to_adj_matrix(best_graph)
    num_compenents = length(connected_components(Graph(M)))
    println("num components: $num_compenents")
    if num_compenents == 1
        println("diameter: $(diameter(Graph(M)))")
    end
    write_adj_list("evolved.txt", M)
end
