include("lib-sim.jl")

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

    for step = 2:max_steps
        seiss[step], new_infections = next_seis(seiss[step-1], M, seis_disease)
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
