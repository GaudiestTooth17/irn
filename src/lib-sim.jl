struct Dizeez
    days_exposed::Int
    days_infectious::Int
    transmission_prob::Float64
end

"Use the disease to make the next SEIR matrix also returns whether or not the old one differs from the new"
function next_seir(old_seir::Matrix{Int}, M::Matrix{T},
    disease::Dizeez)::Tuple{Matrix{Int}, Bool} where T
    # keep track of whether or not any values were changed from the original matrix
    update_occurred = false
    seir = copy(old_seir)
    N = size(M, 1)
    probs = rand(N)
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
    to_e_probs = reshape(1 .- prod(1 .- (M .* disease.transmission_prob)[:, i_filter], dims=2), N)
    to_e_filter = (seir[:, 1] .> 0) .& (probs .< to_e_probs)
    # println("seir $(typeof(to_e_filter)) $(size(to_e_filter))")
    seir[to_e_filter, 2] .= -1
    seir[to_e_filter, 1] .= 0
    # Tracking days and seirs
    seir[(seir .> 0)] .+= 1
    seir[(seir .< 0)] .= 1
    # This should be faster than checking if the two matrices are equal because of short circuiting
    # and empirically it is faster
    rf = any(to_r_filter)
    if_ = any(to_i_filter)
    ef = any(to_e_filter)
    return seir, rf || if_ || ef
end

function next_seis(old_seis::Matrix{Int}, M::Matrix{T},
                   disease::Dizeez) where T
    seis = copy(old_seis)
    N = size(M, 1)
    probs = rand(N)

    # infectious to susceptible
    to_s_filter = seis[:, 3] .> disease.days_infectious
    seis[to_s_filter, 1] .= -1
    seis[to_s_filter, 2] .= 0
    # Exposed to infectious
    to_i_filter = seis[:, 2] .> disease.days_exposed
    seis[to_i_filter, 3] .= -1
    seis[to_i_filter, 2] .= 0
    # susceptible to exposed
    i_filter = seis[:, 3] .> 0
    to_e_probs = reshape(1 .- prod(1 .- (M .* disease.transmission_prob)[:, i_filter], dims=2), N)
    to_e_filter = (seis[:, 1] .> 0) .& (probs .< to_e_probs)
    # println("seis $(typeof(to_e_filter)) $(size(to_e_filter))")
    seis[to_e_filter, 2] .= -1
    seis[to_e_filter, 1] .= 0
    # Tracking days and seis
    seis[(seis .> 0)] .+= 1
    seis[(seis .< 0)] .= 1
    return seis, sum(to_e_filter)
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
    final_seir = seirs[end]
    sum(final_seir[:, 1] .> 0)
end
