using Random, PyPlot

# TODO: try visualizing a simulation to verify that it's running correctly
include("sim-datastructures.jl")
include("fileio.jl")

# "Use the disease to make the next SEIR matrix also returns whether or not the old one differs from the new"
function next_seir(old_seir::Matrix{Int}, adj_matrix::Matrix{T},
    disease::Dizeez)::Tuple{Matrix{T}, Bool} where T

    # keep track of whether or not any values were changed from the original matrix
    update_occurred = false
    seir = copy(old_seir)
    probs = rand(size(adj_matrix, 1))
    # Infectious to recovered
    to_r_filter = seir[:, 3] .> disease.days_infectious
    seir[to_r_filter, 4] .= -1
    seir[to_r_filter, 3] .= 0
    # Exposed to infectious
    to_i_filter = seir[:, 2] .> disease.days_exposed
    seir[to_i_filter, 3] .= -1
    seir[to_i_filter, 2] .= 0
    # Susceptible to exposed
    i_filter = seir[:, 3] .> 0
    to_e_probs = reshape(1 .- prod(1 .- (adj_matrix .* disease.transmission_prob)[:, i_filter], dims=2),
                         size(adj_matrix, 1))
    to_e_filter = (seir[:, 1] .> 0) .& (probs .< to_e_probs)
    seir[to_e_filter, 2] .= -1
    seir[to_e_filter, 1] .= 0
    # Tracking days and seirs
    seir[(seir .> 0)] .+= 1
    seir[(seir) .< 0] .= 1
    # This should be faster than checking if the two matrices are equal because of short circuiting
    # and empirically it is faster
    return seir, any(to_r_filter) || any(to_i_filter) || any(to_e_filter)
end

function next_seis(old_seis::Matrix{Int}, M::Matrix{T},
    disease::Dizeez) where T
    seis = copy(old_seis)
    probs = rand(size(M, 1))

    # infectious to susceptible
    to_s_filter = seis[:, 3] .> disease.days_infectious
    seis[to_s_filter, 1] .= -1
    seis[to_s_filter, 2] .= 0
    # Exposed to infectious
    to_i_filter = seis[:, 2] .> disease.days_exposed
    seis[to_i_filter, 3] .= -1
    seis[to_i_filter, 2] .= 0
    # susceptible to exposed
    i_filter = seis[:, 3] > 0
    to_e_probs = 1 .- prod(1 .- (M .* disease.transmission_prob)[:, i_filter], dims=2)
    to_e_filter = (seis[:, 1] .> 0) .& (probs .< to_e_probs)
    seis[to_e_filter, 2] = -1
    seis[to_e_filter, 1] = 0
    # Tracking days and seis
    seis[(seis .> 0)] .+= 1
    seis[(seis .< 0)] .= 1
    return seis, length(to_e_filter)
end

function simulate(adj_matrix::Matrix{T}, starting_seir::Matrix{Int}, disease::Dizeez, max_steps::Int) where T
    seirs = Vector{Matrix{Int}}(undef, max_steps)
    seirs[1] = copy(starting_seir)
    num_nodes = size(adj_matrix, 1)

    for step = 2:max_steps
        seirs[step], states_changed = next_seir(seirs[step-1], adj_matrix, disease)
        # if no states have changed and all the nodes have been infected, cut the sim short
        # because there isn't anything interesting left to simulate
        all_nodes_infected = length(seirs[step][:, 4][seirs[step][:, 4] .> 0]) == num_nodes
        if !states_changed && all_nodes_infected
            # println("All nodes infected after $step steps.")
            return seirs[1:step]
        end
        disease_gone = (sum(seirs[step][:, 2]) + sum(seirs[step][:, 3])) == 0
        # if the disease is completely gone, there is no need to continue the actual simulation,
        # but it is important for the genetic algorithm to be able to reward this network for being
        # resistant to the disease by having a longer simulation.
        if !states_changed && disease_gone
            for i=step:max_steps
                seirs[i] = copy(seirs[step])
            end
            return seirs
        end
    end

    seirs
end

function simulate_seir_seis(M::Matrix{T}, starting_seir::Matrix{Int}, starting_seis::Matrix{Int},
                            seir_disease::Dizeez, seis_disease::Dizeez, max_steps::Int) where T
    seirs = Vector{Matrix{Int}}(undef, max_steps)
    seirs[1] = copy(starting_seir)
    seiss = Vector{Matrix{Int}}
    seiss = copy(starting_seis)
    seis_nodes_infected = sum(starting_seis[3, :])

    for step = 2:max_steps
        seiss[step], new_infections = next_seis(seis[step-1], M, seis_disease)
        seis_nodes_infected += new_infections
        seirs[step], states_changed = next_seir(seirs[step-1], M, seir_disease)
        # The rest of this function is just logic based around the progression of seir_disease to see
        # if the simulation can be ended early.
        all_nodes_infected = length(seirs[step][:, 4][seirs[step][:, 4] .> 0]) == num_nodes
        # see comment in simulate
        if !states_changed && all_nodes_infected
            return seirs[1:step]
        end
        disease_gone = (sum(seirs[step][:, 2]) + sum(seirs[step][:, 3])) == 0
        # see comment in simulate
        if !states_changed && disease_gone
            for i=step:max_steps
                seirs[i] = copy(seirs[step])
            end
            return seirs
        end
    end

    seirs, seis_nodes_infected
end

function make_starting_seir(num_nodes::Int, num_infected::Int)::Matrix{Int}
    seir = zeros(num_nodes, 4)
    to_infect = Random.shuffle(1:num_nodes)[1:num_infected]
    seir[:, 1] .= 1
    seir[to_infect, 1] .= 0
    seir[to_infect, 3] .= 1
    seir
end

function make_starting_seis(num_nodes::Int, num_infected)::Matrix{Int}
    seis = zeros(num_nodes, 3)
    to_infect = Random.shuffle(1:num_nodes)[1:num_infected]
    seis[:, 1] .= 1
    seis[to_infect, 1] .= 0
    seis[to_infect, 3] .= 1
    seis
end

function calc_remaining_S_nodes(seirs::Vector{Matrix{Int}})::Int
    sum(seirs[end][:, 1])
end

function run_sim_batch(adj_matrix::Matrix{T}, starting_seir::Matrix{Int},
    disease::Dizeez, max_steps::Int, num_sims::Int)::Vector{Int} where T

    # vector_of_remaining_S_nodes = map(i->calc_remaining_S_nodes(
    #     simulate(adj_matrix, starting_seir, disease, max_steps)),
    #     1:num_sims)
    sim_lengths = [length(simulate(adj_matrix, starting_seir, disease, max_steps))
                   for i=1:num_sims]
    sim_lengths
end

if abspath(PROGRAM_FILE) == @__FILE__
    # ev_M = read_adj_list("../graphs/evolved.txt")
    M = read_adj_list("../graphs/spatial-network.txt")
    N = size(M, 1)
    disease = Dizeez(3, 10, .5)
    max_sim_steps = 300
    num_sims = 100
    simulation_results = [x->(calc_remaining_S_nodes(x[1]), x[2])
                          for x in (simulate_seir_seis(M, make_starting_seir(N, 5), make_starting_seis(N, 5)))]
    hist(cavemen_sim_lens)
    show()
end
