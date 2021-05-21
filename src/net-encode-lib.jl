Encoding = Vector{Int8}

function encoding_to_adj_matrix(ϵ::Encoding)::Matrix{Int8}
    # if calculating num_nodes fails, it is probably because the encoding
    # just doesn't have the right number of entries.
    # This is basically the inverse of the triangular sum.
    num_nodes = Int(sqrt(2*length(ϵ)+.25) + .5)
    adj_matrix = zeros(Int8, num_nodes, num_nodes)
    current_edge = 1
    for i = 1:size(adj_matrix, 1)
        for j = i+1:size(adj_matrix, 2)
            adj_matrix[i, j] = ϵ[current_edge]
            adj_matrix[j, i] = ϵ[current_edge]
            current_edge += 1
        end
    end
    adj_matrix
end

function adj_matrix_to_encoding(M::AbstractMatrix)::Encoding
    num_nodes = size(M, 1)
    # genotype contains an entry for every edge on the graph
    genotype = zeros(Int8, (num_nodes*(num_nodes-1) ÷ 2))
    current_loc = 1
    for i = 1:size(M, 1)
        for j = i+1:size(M, 2)
            genotype[current_loc] = M[i, j]
            current_loc += 1
        end
    end
    genotype
end
