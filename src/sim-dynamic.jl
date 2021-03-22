using Random, PyPlot
include("lib-sim.jl")
include("fileio.jl")

function update_connections(D::Matrix{T}, M::Matrix{T}, seir::Matrix{Int}, seis::Matrix{Int}) where T Matrix{T} 
    non_detection_prob = .1
    N = size(M, 1)
    i_agents = seir[:, 3] .> 0 .& (rand(N) .< non_detection_prob)
    r_agents = seir[:, 4] .> 0

    new_D = copy(D)
    # remove connections to infectious agents
    new_D[i_agents, :] .= 0
    new_D[:, i_agents] .= 0
    # recovered agents restore their old connections
    new_D[r_agents, :] = M[r_agents, :]
    new_D[:, r_agents] = M[:, r_agents]
    new_D
end

"This simulates both a SEIR and SEIS disease. M acts as the base matrix which remains unchanged, but the infection
happens on a dynamically changing matrix that has a subset of the edges M has."
function simulate(M::Matrix{T}, starting_seir::Matrix{Int}, starting_seis::Matrix{Int},
                  seir_disease::Dizeez, seis_disease::Dizeez, max_steps) where T
    seirs = Vector{Matrix{Int}}(undef, max_steps)
    seirs[1] = copy(starting_seir)
    seiss = Vector{Matrix{Int}}(undef, max_steps)
    seiss[1] = copy(starting_seis)
    seis_nodes_infected = sum(starting_seis[3, :])
    D = copy(M)  # D is the dynamic graph
    N = size(M, 1)

    for step = 2:max_steps
        seiss[step], new_infections = next_seis(seiss[step-1], D, seis_disease)
        seis_nodes_infected += new_infections
        seirs[step], states_changed = next_seir(seirs[step-1], D, seir_disease)
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
        D = update_connections(D, M, seirs[step], seiss[step])
    end

    seirs, seis_nodes_infected
end

if abspath(PROGRAM_FILE) == @__FILE__
    # ev_M = read_adj_list("../graphs/evolved.txt")
    M = read_adj_list("../graphs/spatial-network.txt")
    N = size(M, 1)
    disease = Dizeez(3, 10, .5)
    max_sim_steps = 150
    num_sims = 100
    simulation_results = [(calc_remaining_S_nodes(x[1]), x[2])
                          for x in (simulate(M, make_starting_seir(N, 5), make_starting_seis(N, 5),
                                             disease, disease, max_sim_steps)
                                    for i=1:num_sims)]
    hist(map(x->x[1], simulation_results), bins=20)
    title("Remaining Susceptible Agents")
    figure()
    title("Good Interactions")
    hist(map(x->x[2], simulation_results))
    show()
end