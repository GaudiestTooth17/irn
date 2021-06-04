using LightGraphs

EdgeList = Vector{Int}

"""
edge_list is the concrete implementation of the thing created by a degree configuration.
If a node has degree n, that node's integer ID will appear n times in edge_list. Edges are
created by putting the values in groups of two starting at index 0 and proceeding to the end.
In this way, edge_list can be viewed as a Vector{Tuple{int, int}}.
"""
function network_from_edge_list(edge_list::Vector{Int})::Graph
    N = maximum(edge_list)
    G = Graph(N)
    for i=1:2:length(edge_list)
        add_edge!(G, (edge_list[i], edge_list[i+1]))
    end
    G
end

"""
Create a subgraph of G containing only the provided nodes.
"""
function subgraph(G::Graph, nodes::Vector{Int})::Graph
    H = copy(G)
    N = length(vertices(G))
    rem_vertices!(H, filter(x->!(x in nodes), 1:N))
    H
end

"""
Execute key on each x in xs. Return the x with highest key(x).
"""
function maxval(key, xs)
    if length(xs) == 0
        return nothing
    end

    max_x = xs[1]
    max_x_val = key(max_x)
    for x in xs[2:end]
        x_val = key(x)
        if x_val > max_x_val
            max_x = x
            max_x_val = x_val
        end
    end
    max_x
end

"""
Read the network file and return the edge list that represents it.
"""
function read_edge_list(filename::String)::Vector{Int}
    G = Graph(read_adj_list(filename))
    edge_list = zeros(Int, length(edges(G))*2)
    i = 1
    for edge in edges(G)
        edge_list[i] = src(edge)
        edge_list[i+1] = dst(edge)
        i += 2
    end
    edge_list
end