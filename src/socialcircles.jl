using LightGraphs


function make_social_circles_network(agent_reach_to_quantity::Dict{Int, Int},
                                     grid_size::Tuple{Int, Int},
                                     force_connected=true)::Matrix{Int}
    max_tries = 100
    for _=1:max_tries
        agents = sort(collect(agent_reach_to_quantity), by=x->x[1], rev=true)
        grid = zeros(Int, grid_size)
        N = sum(values(agent_reach_to_quantity))
        M = zeros(Int, (N, N))
        # place the agents with the largest reach first (that's what the sorting was about)
        loc_to_id = Dict()
        current_id = 0
        for (reach, quantity) in agents
            new_agents = []
            for _=1:quantity
                (x, y) = choose_empty_spot(grid)
                grid[x, y] = reach
                push!(new_agents, (x, y))
                loc_to_id[(x, y)] = current_id
                current_id += 1
            end
            for (x, y) in new_agents
                neighbors = search_for_neighbors(grid, x, y)
                for (i, j) in neighbors
                    M[loc_to_id[(x, y)], loc_to_id[(i, j)]] = 1
                    M[loc_to_id[(i, j)], loc_to_id[(x, y)]] = 1
                end
            end
        end
        if (!force_connected) || is_connected(Graph(M))
            return M
        end
    end
    println("Failed to generate a connected network after $max_tries tries.")
    exit(1)
end

function choose_empty_spot(grid::Matrix{Int})::Tuple{Int, Int}
    (x, y) = (rand(1:size(grid, 1)), rand(1:size(grid, 2)))
    while grid[x, y] > 0
        (x, y) = (rand(1:size(grid, 1)), rand(1:size(grid, 2)))
    end
    (x, y)
end

function search_for_neighbors(grid::Matrix{Int}, x::Int, y::Int)::Set{Tuple{Int, Int}}
    """
    Search for all of the agents in range of the agent at (x, y).
        
    Note that since this doesn't check to see if (x, y) is in range of anything else,
    all the agents with the longest reach must be placed, then the next longest, down to
    the agents with the shortest reach.
    """
    reach = grid[x, y]
    min_x = max(1, x-reach)
    max_x = min(size(grid, 1), x+reach)
    min_y = max(1, y-reach)
    max_y = min(size(grid, 2), y+reach)
    neighbors = Set((i, j)
                    for i=min_x:max_x, j=min_y:max_y
                    if grid[i, j] > 0 && distance(x, y, i, j) ≤ reach && (x, y) ≠ (i, j))
    neighbors
end

function find_agents_in_mutual_range(grid::Matrix, x::Int, y::Int)::Set{Tuple{Int, Int}}
    """
    Search for all of the agents in range of (x, y) who can also reach (x, y).
    """
    reach = grid[x, y]
    min_x = max(1, x-reach)
    max_x = min(size(grid, 1), x+reach)
    min_y = max(1, y-reach)
    max_y = min(size(grid, 2), y+reach)
    neighbors = Set()
    for i=min_x:max_x, j=min_y:max_y
        reach_neighbor = grid[i, j]
        if reach_neighbor ≤ 0 || (x, y) == (i, j)
            continue
        end
        dist = distance(x, y, i, j)
        if dist ≤ reach_neighbor && dist ≤ reach
            push!(neighbors, (i, j))
        end
    end
    neighbors
end

function grid_to_adjacency_matrix(grid::Matrix)::Matrix{Int}
    agent_locations = Tuple.(findall(>(0), grid))
    loc_to_id = Dict(loc=>id for (id, loc) in enumerate(agent_locations))
    N = length(agent_locations)
    M = zeros(Int, (N, N))
    for (x, y) in agent_locations
        agent_id = loc_to_id[(x, y)]
        neighbors = find_agents_in_mutual_range(grid, x, y)
        for neighbor in neighbors
            n_id = loc_to_id[neighbor]
            M[agent_id, n_id] = 1
            M[n_id, agent_id] = 1
        end
    end
    M
end

function distance(x₁, y₁, x₂, y₂)::Float64
    sqrt((x₁-x₂)^2 + (y₁-y₂)^2)
end