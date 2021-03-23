using Random, PyPlot

# TODO: try visualizing a simulation to verify that it's running correctly
include("lib-sim.jl")
include("fileio.jl")

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
    seiss = Vector{Matrix{Int}}(undef, max_steps)
    seiss[1] = copy(starting_seis)
    seis_nodes_infected = sum(starting_seis[3, :])
    N = size(M, 1)

    for step = 2:max_steps
        seirs[step], states_changed = next_seir(seirs[step-1], M, seir_disease)
        seiss[step], new_infections = next_seis(seiss[step-1], M, seis_disease)
        seis_nodes_infected += new_infections
        # The rest of this function is just logic based around the progression of seir_disease to see
        # if the simulation can be ended early.
        all_nodes_infected = length(seirs[step][:, 4][seirs[step][:, 4] .> 0]) == N
        # see comment in simulate
        if !states_changed && all_nodes_infected
            return seirs[1:step], seis_nodes_infected
        end
        disease_gone = (sum(seirs[step][:, 2]) + sum(seirs[step][:, 3])) == 0
        # see comment in simulate
        if !states_changed && disease_gone
            for i=step:max_steps
                seirs[i] = copy(seirs[step])
            end
            return seirs, seis_nodes_infected
        end
    end

    seirs, seis_nodes_infected
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
    max_sim_steps = 150
    num_sims = 1000
    simulation_results = [(calc_remaining_S_nodes(x[1]), x[2])
                          for x in (simulate_seir_seis(M, make_starting_seir(N, 5), make_starting_seis(N, 5),
                                                       disease, disease, max_sim_steps)
                                    for i=1:num_sims)]
    remaining_S_nodes = sort(collect(map(x->x[1], simulation_results)))
    good_interactions = sort(collect(map(x->x[2], simulation_results)))
    hist(remaining_S_nodes, bins=20)
    title("Remaining Susceptible Agents (No Quarantine)")
    # figure()
    savefig("Remaining Susceptible Agents (No Quarantine)")
    clf()
    title("Good Interactions (No Quarantine)")
    hist(good_interactions, bins=20)
    # show()
    savefig("Good Interactions (No Quarantine)")
    println("Median agents left susceptible: $(remaining_S_nodes[length(remaining_S_nodes) ÷ 2])")
    println("Median good interactions: $(good_interactions[length(good_interactions) ÷ 2])")
end
