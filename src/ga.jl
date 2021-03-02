using Plots
using Dates
using StatsBase
using Base.Iterators: flatten

Genotype = Vector
Population = Vector{Genotype}

"""
make_optimizer aggregates its parameters to create a closure that acts as the body of a GA optimization loop
fitness_fn takes a genotype and returns a fitness value. This must not exceed max_fitness.
next_gen_fn takes the maximum fitness a genotype can have and sorted list of (fitness, genotype) ordered from
highest to lowest fitness and returns a vector of genotypes to be the next generation.
max_fitness is a Real that is the highest a genotype could possibly score in the fitness_fn.
starting_population is the population of genotypes that the optimizer begins with.
"""
function make_optimizer(fitness_fn, next_gen_fn, max_fitness, starting_population)
    population = starting_population
    rand_genotype = rand(population)
    most_fit = (fitness_fn(rand_genotype), rand_genotype)

    function optimizer_step(verbose::Bool=false)
        fitness_to_genotype = [(fitness_fn(gen), gen) for gen in population]
        sort!(fitness_to_genotype, by=first, rev=true)  # sort the array in place from highest fitness to lowest
        most_fit = tuple_max([most_fit, fitness_to_genotype[1]], 1)
        population = next_gen_fn(max_fitness, fitness_to_genotype)
        if verbose
            println("End of step $step.")
        end
        most_fit, fitness_to_genotype
    end
    optimizer_step
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
    pop_diversity = zeros(max_steps)
    steps_taken = 0
    function observer(fitness_to_genotype, most_fit)
        steps_taken += 1
        max_fitnesses[steps_taken] = most_fit[1]
        unique_genotypes = Set(map(x->x[2], fitness_to_genotype))
        pop_diversity[steps_taken] = length(unique_genotypes) / length(fitness_to_genotype)
        # println(steps_taken)
    end
    function reporter()
        max_fitnesses[1:steps_taken], pop_diversity[1:steps_taken]
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

if abspath(PROGRAM_FILE) == @__FILE__
    start_time = Dates.now()
    println("Beginning.")
    pop_size = 20
    sequence_length = 10
    max_fitness = example_fitness([i for i=1:sequence_length])
    max_steps = 1000

    observe, report = make_example_observer(max_steps)
    population = [abs.(rand(Int, sequence_length)) .% sequence_length .+ 1 for i=1:pop_size]
    optimizer_step = make_optimizer(example_fitness, example_next_gen, max_fitness, population)
    best_answer = missing
    for i=1:max_steps
        global best_answer
        best_answer, fitness_to_answer = optimizer_step()
        observe(fitness_to_answer, best_answer)
    end

    println(best_answer)
    println("Finished $(Dates.now()-start_time)")
    max_fitnesses_over_time, pop_diversity = report()
    display(plot(max_fitnesses_over_time))
    readline()
    display(plot(pop_diversity))
    readline()
end
