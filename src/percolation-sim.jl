using Random
using LightGraphs
import Base.Iterators: flatten
import PyPlot
include("net-encode-lib.jl")

"""
M is an unweighted adjacency matrix.
β is the transmission probability
τ is the time infectious
num_infectious gives the number of agents infectious at the start of the simulation
Returns the number of agents who caught the infection
"""
function simulate_static(M::Matrix, β::Float64, τ::Int,
    num_infectious::Int)::Int
    # percolate edges
    # ϕ = 1 - ϕ′
    # ϕ′ is more convenient for the code than ϕ
    ϕ′ = exp(-β*τ)
    ϵ = adj_matrix_to_encoding(M)
    # findall is like np.where
    all_edges_inds = findall(==(1), ϵ)
    edges_to_remove = shuffle(all_edges_inds)[1:Int(round(ϕ′*length(all_edges_inds)))]
    percolated_ϵ = copy(ϵ)
    percolated_ϵ[edges_to_remove] .= 0
    percolated_M = encoding_to_adj_matrix(percolated_ϵ)

    # identify infected agents
    N = size(M, 1)
    infectious_agents = Set(randperm(N)[1:num_infectious])
    components = connected_components(Graph(percolated_M))
    infected_components = Set(component for component in components
                                        for agent in component
                                        if agent in infectious_agents)
    infected_agents = Set(flatten(infected_components))

    length(infected_agents)
end

"""
M is an unweighted adjacency matrix.
β is the transmission probability
τ is the time infectious
percent_infectious gives the percentage of agents infectious at the start of the simulation
Returns the number of agents who caught the infection
"""
function simulate_static(M::Matrix, β::Float64, τ::Int, percent_infectious::Float64)::Int
    num_infectious = Int(round(percent_infectious * size(M, 1)))
    simulate_static(M, β, τ, num_infectious)
end

if abspath(PROGRAM_FILE) == @__FILE__
    include("fileio.jl")
    using ProgressBars
    for name in ("agent-generated-500", "cavemen-50-10", "cgg-520", "elitist-500-500")
        M = read_adj_list("../graphs/$name.txt")
        N = size(M, 1)
        β = .15
        τ = 5
        results = [N - simulate_static(M, β, τ, 1) for i ∈ ProgressBar(1:2000)]
        PyPlot.title("$name Susceptible Agents")
        PyPlot.hist(results)
        PyPlot.show(block=false)
        PyPlot.figure()
    end
    readline()
end
