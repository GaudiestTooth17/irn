"""
There is something wrong with the algorithm. Review the NetworkX code.
"""

using LightGraphs, DataStructures

function edge_betweeness(G::Graph)::Dict{Edge, Float64}
    betweeness = Dict()
    for v ∈ vertices(G)
        betweeness[v] = 0.0
    end
    for edge ∈ edges(G)
        betweeness[edge] = 0.0
    end

    nodes = vertices(G)
    for s ∈ nodes
        S, P, σ, _ = _single_source_shortest_path_basic(G, s)
        betweeness = _accumulate_edges(betweeness, S, P, σ, s)
    end
    for n ∈ vertices(G)
        delete!(betweeness, n)
    end
    betweeness = _rescale_e(betweeness, length(vertices(G)), true, false)
    betweeness
end

function _single_source_shortest_path_basic(G::Graph, s::Int)
    S = []
    P = Dict()
    for v in vertices(G)
        P[v] = []
    end
    # σ tracks weight
    σ = Dict(v => 0.0 for v ∈ vertices(G))
    # D tracks distance
    D = Dict()
    σ[s] = 1.0
    D[s] = 0
    Q = Queue{Int}()
    enqueue!(Q, s)
    while length(Q) > 0
        v = dequeue!(Q)
        push!(S, v)
        Dv = D[v]
        sigmav = σ[v]
        for w ∈ neighbors(G, v)
            if !haskey(D, w)
                enqueue!(Q, w)
                D[w] = Dv + 1
            end
            if D[w] == Dv + 1  # this is a shortest path, count paths
                σ[w] += sigmav
                push!(P[w], v)  # predecessors
            end
        end
    end
    S, P, σ, D
end

function _accumulate_edges(betweeness, S, P, σ, s)
    δ = Dict(s => 0.0 for s ∈ S)
    for i=length(S):-1:1
        w = S[i]
        coeff = (1 + δ[w]) / σ[w]
        for v ∈ P[w]
            c = σ[v] * coeff
            if !haskey(betweeness, Edge(v, w))
                betweeness[Edge(w, v)] += c
            else
                betweeness[Edge(v, w)] += c
            end
            δ[v] += c
        end
        if w != s
            betweeness[w] += δ[w]
        end
    end
    betweeness
end

function _rescale_e(betweeness, n, normalized::Bool, directed::Bool, k=nothing)
    scale = if normalized
        if n <= 1
            nothing  # no normalization b=0 for all nodes
        else
            1 / (n * (n-1))
        end
    else  # rescale by 2 for undirected graphs
        if !directed
            0.5
        else
            nothing
        end
    end
    if !isa(scale, Nothing)
        if !isa(k, Nothing)
            scale = scale * n / k
        end
        for v in keys(betweeness)
            betweeness[v] *= scale
        end
    end
    betweeness 
end

if abspath(PROGRAM_FILE) == @__FILE__
    G = cycle_graph(10)
    # G = complete_graph(10)
    betweeness = edge_betweeness(G)
    for (k, v) ∈ betweeness
        println("$k: $v")
    end
end