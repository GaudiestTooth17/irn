using Test
include("static-graph-ga.jl")

N = 4
M = [0 1 0 1;
     1 0 1 0;
     0 1 0 1;
     1 0 1 0]
expected_genotype = Int8[1, 0, 1, 1, 0, 1]
actual_genotype = adj_matrix_to_encoding(M)
@test expected_genotype == actual_genotype
@test size(expected_genotype, 1) == N*(N-1) ÷ 2
new_M = encoding_to_adj_matrix(expected_genotype)
@test M == new_M
# @test false == true
