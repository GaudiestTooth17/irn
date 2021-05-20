using Random, PyPlot, ProgressBars
include("lib-sim.jl")
include("fileio.jl")

SEIR = Matrix{Int}
SEIRS = Vector{SEIR}

"""
Creates a function to use in simulate that updates the connections on the graph.
detection_prob is the probability every step of an infectious person being discovered.
"""
function make_behavior_function_dual(detection_prob::Float64)::Function
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
Creates a function to use in simulate that updates the connections on the graph.
detection_prob is the probability every step of an infectious person being discovered.
"""
function make_detection_behavior(detection_prob::Float64)::Function
    function update_connections(D::Matrix, M::Matrix, seir::Matrix)::Matrix
        probs = rand(Float64, size(M, 1))
        i_agents = (seir[:, 3] .> 0) .& (probs .< detection_prob)
        r_agents = seir[:, 4] .> 0

        new_D = copy(D)
        #remove connections to infectious agents
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
This behavior function is based off of the social circles model. Agents exist and move around
in a square grid world. They form connections based off of which other agents are within
mutual range.
"""
function make_moving_agent_behavior_function(N::Int, grid_density::Float64)::Function
    # TODO: implement
end

"This simulates both a SEIR and SEIS disease. M acts as the base matrix which remains unchanged, but the infection
happens on a dynamically changing matrix that has a subset of the edges M has."
function dual_simulate(M::Matrix{T}, starting_seir::Matrix{Int}, starting_seis::Matrix{Int},
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

# TODO: Think about why a total quarantine is undesirable and how this could be represented
# in the simulation. Perhaps each agent could have a mental well-being that needs
# to be maximized.
"""
This simulation function keeps track of the number of edges present at each step instead of
simulating a second disease. The goal is to maximize the number of edges present because
this represents the number of fulfilling social interactions.
"""
function simulate(M::Matrix{T}, seir₀::SEIR, disease::Dizeez,
    update_connections::Function, max_steps::Int)::Tuple{SEIRS, Vector{Int}} where T
    seirs = SEIRS(undef, max_steps)
    seirs[1] = copy(seir₀)
    edges_present = zeros(Int, max_steps)
    edges_present[1] = sum(M) ÷ 2
    D = copy(M)
    N = size(M, 1)

    for step=2:max_steps
        # Get the adjacency matrix to use at this step
        D = update_connections(D, M, seirs[step-1])
        # update the statistic about the number of edges in D
        edges_present[step] = sum(D) ÷ 2
        # println("step $step num_edges $(edges_present[step])")

        # next seir is the workhorse of the simulation because it is responsible
        # for simulating the disease spread
        seirs[step], states_changed = next_seir(seirs[step-1], D, disease)

        # find all the agents that are in the removed state. If that number is N,
        # the simulation is done.
        all_nodes_infected = length(findall(>(0), seirs[step][:, 4])) == N
        if !states_changed && all_nodes_infected
            return seirs[1:step], edges_present[1:step]
        end

        # If there aren't any exposed or infectious agents, the disease is gone and we
        # can take a short cut to finish the simulation.
        disease_gone = (sum(seirs[step][:, 2]) + sum(seirs[step][:, 3])) == 0
        if !states_changed && disease_gone
            for i=step:max_steps
                seirs[i] = copy(seirs[step])
                # TODO: This may not be the best way to auto-populate edges_present
                # Change this to not add anything extra to edges_present. Just return
                # the current numbers without extrapolating into the future.
                edges_present[i] = edges_present[1]
            end
            return seirs, edges_present
        end
    end

    seirs, edges_present
end

"""
If an agent detects that a neighbor is infectious, it will disconnect from it with a probability
that is positively correlated with its own degree.
"""
function make_degree_behavior_function(M::Matrix)::Function
    max_M_deg = maximum(sum(M, dims=1))
    degrees = sum(M, dims=1)
    willingness_to_disconnect = [deg/max_M_deg for deg in degrees]
    function update_connections(D::Matrix, M::Matrix, seir::Matrix{Int})::Matrix
        new_D = copy(M)
        N = size(M, 1)
        for u=1:N
            for v=u+1:N
                if rand() < willingness_to_disconnect[u]
                    disconnect_agents!(new_D, u, v)
                end
            end
        end
        new_D
    end
    update_connections
end

"""
Modifies the adjacency matrix M by removing any edges between vertices u and v.
"""
function disconnect_agents!(M::Matrix, u::Int, v::Int)
    M[u, v] = 0
    M[v, u] = 0
end

function run_dual_sims()
    # ev_M = read_adj_list("../graphs/evolved.txt")
    name = "square-lattice"
    M = read_adj_list("../graphs/$name.txt")
    N = size(M, 1)
    disease = Dizeez(3, 10, .5)
    max_sim_steps = 150
    num_sims = 2000
    # for detection_prob in (0.0, .1, .25, .5, .75, .9, 1.0)
    for detection_prob in (.5, .75, .9)
    # for detection_prob in (1.0, 0.0)
        update_connections = make_behavior_function_dual(detection_prob)
        simulation_results = [(calc_remaining_S_nodes(x[1]), x[2])
                              for x in (dual_simulate(M, make_starting_seir(N, 5), make_starting_seis(N, 5),
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

function run_sims()
    name = "cavemen-50-10"
    M = read_adj_list("../graphs/$name.txt")
    N = size(M, 1)
    disease = Dizeez(3, 10, .5)
    max_sim_steps = 150
    num_sims = 2000

    # for detection_prob in (.5, .75, .9)
    update_connections = make_degree_behavior_function(M)
    # update_connections = make_detection_behavior(detection_prob)
    simulation_results = [(calc_remaining_S_nodes(x[1]), average(x[2])/N)
                            for x in (simulate(M, make_starting_seir(N, 5), disease,
                                                update_connections, max_sim_steps)
                                        for _=ProgressBar(1:num_sims))]
    scatter(map(x->x[1], simulation_results), map(x->x[2], simulation_results), s=4.0)
    title("$name\nInfection vs Interaction\nShunning Behavior")
    xlabel("Susceptible Nodes at End of Simulation")
    ylabel("Average Number of Edges per Agent")
    savefig("$name Infection vs Interaction Shunning Behavior.png")
    clf()
    # end
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_sims()
end
