using Random, PyPlot, ProgressBars
include("lib-sim.jl")
include("fileio.jl")

"Creates a function to use in simulate that updates the connections on the graph.
detection_prob is the probability every step of an infectious person being discovered."
function make_behavior_function(detection_prob::Float64)::Function
    function update_connections(D::Matrix{T}, M::Matrix{T}, seir::Matrix{Int}, seis::Matrix{Int}) where T Matrix{T}
        probs = rand(Float64, size(M, 1))
        i_agents = (seir[:, 3] .> 0) .& (probs .< detection_prob)
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
    update_connections
end

"""
If an agent detects that a neighbor is infectious, it will disconnect from it with a probability
that is positively correlated with its own degree.
"""
function make_degree_behavior_function(detection_prob::Float64)::Function
    # TODO: incorporate willingness_to_disconnect in deciding whether to disconnect
    function update_connections(D::Matrix, M::Matrix, seir::Matrix{Int}, seis::Matrix{Int})::Matrix
        N = size(M, 1)
        # calculate the degree of each agent
        degrees = sum(D, dims=1)
        max_deg = maximum(degrees)
        observation_power = rand(Float64, N)
        # an agent's willingness to connect increases as their degree increases
        willingness_to_disconnect = [1/(max_deg-deg+1) for deg in degrees]
        i_agents = (seir[:, 3] .> 0) .& (observation_power .< detection_prob)
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
end

"This simulates both a SEIR and SEIS disease. M acts as the base matrix which remains unchanged, but the infection
happens on a dynamically changing matrix that has a subset of the edges M has."
function simulate(M::Matrix{T}, starting_seir::Matrix{Int}, starting_seis::Matrix{Int},
                  seir_disease::Dizeez, seis_disease::Dizeez, update_connections::Function,
                  max_steps::Int) where T
    seirs = Vector{Matrix{Int}}(undef, max_steps)
    seirs[1] = copy(starting_seir)
    seiss = Vector{Matrix{Int}}(undef, max_steps)
    seiss[1] = copy(starting_seis)
    seis_nodes_infected = sum(starting_seis[3, :])
    D = copy(M)  # D is the dynamic adjacency matrix
    N = size(M, 1)

    for step = 2:max_steps
        D = update_connections(D, M, seirs[step-1], seiss[step-1])
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
    end

    seirs, seis_nodes_infected
end

if abspath(PROGRAM_FILE) == @__FILE__
    # ev_M = read_adj_list("../graphs/evolved.txt")
    name = "spatial-network"
    M = read_adj_list("../graphs/$name.txt")
    N = size(M, 1)
    disease = Dizeez(3, 10, .5)
    max_sim_steps = 150
    num_sims = 2000
    # for detection_prob in (0.0, .1, .25, .5, .75, .9, 1.0)
    for detection_prob in (.5, .75, .9)
    # for detection_prob in (1.0, 0.0)
        update_connections = make_behavior_function(detection_prob)
        simulation_results = [(calc_remaining_S_nodes(x[1]), x[2])
                              for x in (simulate(M, make_starting_seir(N, 5), make_starting_seis(N, 5),
                                                 disease, disease, update_connections, max_sim_steps)
                                        for i=ProgressBar(1:num_sims))]
        # Scatterplot Generation
        scatter(map(x->x[1], simulation_results), map(x->x[2], simulation_results), s=4.0)
        title("Infection vs Interaction\nDetection Prob = $detection_prob on $name")
        xlabel("Susceptible Nodes")
        ylabel("Good Interactions")
        savefig("Infection vs Interaction Detection Prob = $detection_prob on $name.png")
        clf()

        # Histogram Generation
        # fig, (top, bottom) = subplots(2, 1)
        # fig.suptitle("$name\n Detection Prob = $detection_prob Sim Length = $max_sim_steps Num Trials = $num_sims")
        # top.set_xlabel("Remaining Susceptible Agents")
        # top.set_ylabel("Num trials")
        # top.hist(map(x->x[1], simulation_results), bins=20, color="orange")
        # bottom.set_xlabel("Good Interactions")
        # bottom.set_ylabel("Num trials")
        # bottom.hist(map(x->x[2], simulation_results), bins=20, color="orange")
        # tight_layout()
        # subplots_adjust(top=.9)
        # savefig("Results for Detection Probability = $detection_prob on $name.png")
    end
end
