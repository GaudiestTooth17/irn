using Random
include("datastructures.jl")

struct Dizeez
    days_exposed::Int
    days_infectious::Int
    transmission_prob::Float64
end

"Use the disease to make the next SEIR matrix also returns whether or not the old one differs from the new"
function next_seir(old_seir::Matrix{Integer}, adj_matrix::Matrix{Integer}, disease::Dizeez)::Tuple{Matrix{Integer}, Bool}
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
    to_e_probs = reshape(1 .- prod(1 .- (adj_matrix .* disease.transmission_prob)[:, i_filter], dims=2), size(adj_matrix, 1))
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

function simulate(adj_matrix::AbstractMatrix{Int}, starting_seir::AbstractMatrix{Int}, disease::Dizeez, max_steps::Int)
    seirs = Array{Union{AbstractMatrix{Int}, Nothing}}(nothing, max_steps)
    seirs[1] = copy(starting_seir)

    for step = 2:max_steps
        seirs[step], kontinue = next_seir(seirs[step-1], adj_matrix, disease)
        # This could slow down the simulation a lot
        # My preliminary test (caveman and 1000 steps) suggests that it doesn't
        # significantly slow it down.
        # Maybe a faster check could be added in next_seir
        if !kontinue
            return seirs[1:step]
        end
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
