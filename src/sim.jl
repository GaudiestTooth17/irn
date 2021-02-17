using Random
include("datastructures.jl")

struct Dizeez
    days_exposed::Int
    days_infectious::Int
    transmission_prob::Float64
end

function next_seir(old_seir::AbstractMatrix{Int}, adj_matrix::AbstractMatrix{Int})::AbstractMatrix{Int}
    seir = copy(old_seir)  # TODO: Find out if it is even necessary to make a copy
    probs = rand(size(adjmatrix, 1))
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
    to_e_probs = reshape(1 .- prod(1 .- (adjmatrix .* disease.transmission_prob)[:, i_filter], dims=2), size(adjmatrix, 1))
    to_e_filter = (seir[:, 1] .> 0) .& (probs .< to_e_probs)
    seir[to_e_filter, 2] .= -1
    seir[to_e_filter, 1] .= 0
    # Tracking days and seirs
    seir[(seir .> 0)] .+= 1
    seir[(seir) .< 0] .= 1
    seir
end

function simulate(adjmatrix::AbstractMatrix{Int}, starting_seir::AbstractMatrix{Int},
    disease::Dizeez, max_steps::UInt=1000)::Array{Union{AbstractMatrix{Int}, Nothing}, max_steps}

    seir = copy(starting_seir)
    seirs = Array{Union{AbstractMatrix{Int}, Nothing}, max_steps}(nothing)

    for step = 1:max_steps
        # Probabilistic vales to use during the simulation
        probs = rand(size(adjmatrix, 1))
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
        to_e_probs = reshape(1 .- prod(1 .- (adjmatrix .* disease.transmission_prob)[:, i_filter], dims=2), size(adjmatrix, 1))
        to_e_filter = (seir[:, 1] .> 0) .& (probs .< to_e_probs)
        seir[to_e_filter, 2] .= -1
        seir[to_e_filter, 1] .= 0
        # Tracking days and seirs
        seir[(seir .> 0)] .+= 1
        seir[(seir) .< 0] .= 1
        seirs[step] = copy(seir)
        # TODO Exit early if there has been no change
    end
    seirs
end

function make_starting_seir(num_nodes::Int, num_infected::Int)::AbstractMatrix{Int}
    seir = zeros(num_nodes, 4)
    to_infect = Random.shuffle(1:num_nodes)[1:num_infected]
    seir[:, 1] .= 1
    seir[to_infect, 1] .= 0
    seir[to_infect, 3] .= 1
    seir
end
