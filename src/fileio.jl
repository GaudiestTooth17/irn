using LinearAlgebra
include("sim-datastructures.jl")

function read_adj_list(file_name)::Matrix{Int}
    file = open(file_name, "r")
    line = readline(file)
    n = parse(Int, line)
    adj_matrix = zeros(n, n)

    # this is mimicking a do while structure
    line = readline(file)
    while length(line) > 1
        n1, n2 = split(line, " ")
        # You have to add 1 because julia is 1 indexed -_-
        i = parse(Int, n1) + 1
        j = parse(Int, n2) + 1
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1

        line = readline(file)
    end
    close(file)

    Symmetric(adj_matrix)
end

function write_adj_list(file_name, matrix)
    file = open(file_name, "w")
    write(file, "$(size(matrix, 1))\n")
    for i = 1:size(matrix, 1)
        for j = i:size(matrix, 2)
            if matrix[i, j] > 0
                # subtract one because the other languages are 0 indexed
                write(file, "$(i-1) $(j-1)\n")
            end
        end
    end
    close(file)
end

function read_disease_file(file_name)::Dizeez
    file = open(file_name, "r")
    fields = split(readline(file), " ")
    time_to_i = parse(Int, fields[1])
    time_to_r = parse(Int, fields[2])
    inf_prob = parse(Float64, fields[3])
    # TODO: parse out num_to_infect
    Dizeez(time_to_i, time_to_r, inf_prob)
end
