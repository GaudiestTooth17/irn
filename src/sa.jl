using Dates
using PyPlot

"""
Make a simulated annealing optimizer.
objective outputs the quality of a solution (Vector -> Float64)
next_temp returns the next temperature to use. The algorithm stops when it is 0.
σ₀ is the starting solution
"""
function make_sa_optimizer(objective::Function, next_temp::Function, neighbor::Function,
    σ₀::Vector)::Function
    T = next_temp()
    σ = σ₀

    function step()
        σ′ = neighbor(σ)
        energy = objective(σ)
        energy′ = objective(σ′)
        curr_energy = energy
        if P(energy, energy′, T) ≥ rand()
            σ = σ′
            curr_energy = energy′
        end
        T = next_temp()

        σ, curr_energy
    end

    step
end

function P(energy, energy′, T)::Float64
    acceptance_prob = if energy′ < energy
        1.0
    else
        exp(-(energy′-energy)/T)
    end

    acceptance_prob
end

"""
Outputs a simple annealing schedule function.
Notice that the first time the return function is called, it will return T₀.
This one seems to work a lot better than the linear schedule.
"""
function make_fast_schedule(T₀::Float64)::Function
    num_steps = -1
    function next_temp()
        num_steps += 1
        T₀ / (num_steps + 1)
    end

    next_temp
end

"""
Make a temperature schedule that decreases linearly. The first temperature returned is
T₀. Each subsequent temperature is δT less than the last. The minimum temperature is 0.
"""
function make_linear_schedule(T₀::Float64, ΔT::Float64)::Function
    T = T₀ + ΔT
    function schedule()::Float64
        T -= ΔT
        max(0.0, T)
    end
    schedule
end

"""
try to create an array where each value is the same as its index
"""
function example_objective(genotype::Vector{Int})::Int
    size = length(genotype)
    fitness = 10*size
    for (i, gene) in enumerate(genotype)
        fitness -= abs(i-gene)
    end
    -fitness
end

function example_neighbor(solution::Vector{Int})::Vector{Int}
    new = copy(solution)
    new[rand(1:length(new))] += rand((1, -1))
    new
end

function sa_demo()
    start_time = Dates.now()
    println("Beginning.")
    sequence_length = 10
    T₀ = 100.0
    max_steps = 500
    σ₀ = [1 for i in 1:sequence_length]
    optimizer_step = make_sa_optimizer(example_objective, make_fast_schedule(T₀),
        example_neighbor, σ₀)
    
    best_solution = missing
    energies = zeros(max_steps)
    for step = 1:max_steps
        best_solution, energy = optimizer_step()
        energies[step] = energy
    end
    println(best_solution)
    println("Done. ($(Dates.now()-start_time))")
    PyPlot.plot(energies)
    PyPlot.show()
end

if abspath(PROGRAM_FILE) == @__FILE__
    sa_demo()
end