module DiseaseSim

using LightGraphs, Dates
include("sim.jl")
include("fileio.jl")
include("datastructures.jl")

function main()
    if length(ARGS) < 3
        println("Usage: <adj_matrix_file> <disease_file> <num_sims>")
        return
    end

    adj_matrix_file = ARGS[1]
    disease_file = ARGS[2]
    num_sims = parse(Int, ARGS[3])
    disease = read_disease_file(disease_file)
    adj_matrix = read_adj_list(adj_matrix_file)
    starting_seir = make_starting_seir(size(adj_matrix, 1), 1)

    starting_time = Dates.now()
    for i = 1:num_sims
        simulate(adj_matrix, starting_seir, disease)
    end
    println("Done ($(Dates.now() - starting_time))")
end

main()

end # module
