using Plots
using Dates
using StatsBase
using Base.Iterators: flatten

Genotype = Vector
Population = Vector{Genotype}

"""
A genotype is an array of Numbers. fitness_func takes in the population and returns fitness for each. next_gen_
"""
function optimize(fitness_fn, next_gen_fn, observer_fn, max_steps, population, max_fitness)
    rand_genotype = rand(population)
    most_fit = (fitness_fn(rand_genotype), rand_genotype)
    for step = 1:max_steps
        fitness_to_genotype = [(fitness_fn(genotype), genotype) for genotype in population]
        sort!(fitness_to_genotype, by=first, rev=true)  # sort the array in place from highest fitness to lowest
        most_fit = tuple_max([most_fit, fitness_to_genotype[1]], 1)
        population = next_gen_fn(max_fitness, fitness_to_genotype)
        observer_fn(fitness_to_genotype, most_fit)
    end
    most_fit
end

function calc_normalized_fitnesses(max_fitness::Real, fitnesses)
    standardized_fitnesses = max_fitness .- fitnesses
    adjusted_fitnesses = 1 ./ (1 .+ standardized_fitnesses)
    sum_adjusted_fitnesses = sum(adjusted_fitnesses)
    adjusted_fitnesses ./ sum_adjusted_fitnesses
end

"""
fitness_to_genotype is a list of tuples of (fitness, genotype)
"""
function roullete_wheel_selection(max_fitness, fitness_to_genotype)::Tuple{Genotype, Genotype}
    normalized_fitnesses = calc_normalized_fitnesses(max_fitness, first.(fitness_to_genotype))
    genotypes = map(x->x[2], fitness_to_genotype)
    wsample(genotypes, normalized_fitnesses), wsample(genotypes, normalized_fitnesses)
end

"""
Chooses a locus (point in the array) to cross the parents over. Returns a genotype for each parent.
"""
function crossover(α::Genotype, ω::Genotype)::Tuple{Genotype, Genotype}
    # If length(α)+1 is selected, no cross over will occur. This is intentional.
    locus = rand(1:length(α)+1)
    [i < locus ? α[i] : ω[i] for i=1:length(α)], [i < locus ? ω[i] : α[i] for i=1:length(α)]
end

"""
prob is the probability of a mutation happening at each location in the genotype
"""
function mutate!(population, prob::Float64)
    for i = 1:length(population)
        for j = 1:length(population[i])
            if rand() < prob
                population[i][j] += rand([1, -1])
            end
        end
    end
end

# try to create an array where each value is the same as its index
function example_fitness(genotype::Vector{Int})::Int
    size = length(genotype)
    fitness = 10*size
    for (i, gene) in enumerate(genotype)
        fitness -= abs(i-gene)
    end
    fitness
end

"""
All next_gen function should adhere to this type signature (except that Int can be any comparable value)
"""
function example_next_gen(max_fitness, population::Vector{Tuple{Int, Vector{Int}}})::Vector{Vector{Int}}
    parent_pairs = (roullete_wheel_selection(max_fitness, population)
                    for i=1:div(length(population), 2))
    children = collect(flatten((crossover(parents...)
                                for parents in parent_pairs)))
    mutate!(children, .25)
    children
end

function make_example_observer(max_steps::Int)
    max_fitnesses = zeros(max_steps)
    steps_taken = 0
    function observer(fitness_to_genotype, most_fit)
        steps_taken += 1
        max_fitnesses[steps_taken] = most_fit[1]
        # println(steps_taken)
    end
    function reporter()
        max_fitnesses[1:steps_taken]
    end
    observer, reporter
end

function tuple_max(tuples, important_index)
    max_val = tuples[1]
    for tuple in tuples[2:end]
        if max_val[important_index] < tuple[important_index]
            max_val = tuple
        end
    end
    max_val
end

start_time = Dates.now()
println("Beginning.")
pop_size = 20
sequence_length = 10
max_fitness = example_fitness([i for i=1:sequence_length])
max_steps = 1000
observe, report = make_example_observer(max_steps)
population = [abs.(rand(Int, sequence_length)) .% sequence_length .+ 1 for i=1:pop_size]
answer = optimize(example_fitness, example_next_gen, observe, max_steps, population, max_fitness)
println(answer)
println("Finished $(Dates.now()-start_time)")
display(plot(report()))
readline()
