#%%
import itertools

v = {
    (): 0,
    ('A',): 20,
    ('B',): 20,
    ('C',): 60,
    ('A', 'B'): 60,
    ('A', 'C'): 70,
    ('B', 'C'): 70,
    ('A', 'B', 'C'): 100
}

players = ['A', 'B', 'C']
total_shapley_value = 0
total_permutations = 0

for permutation in itertools.permutations(players):
    total_permutations += 1
    index_A = permutation.index('A')
    if index_A == 0:
        total_shapley_value += v[(permutation[0],)]
    else:
        without_A = tuple(sorted(permutation[:index_A]))
        with_A = tuple(sorted(permutation[:index_A + 1]))
        total_shapley_value += v[with_A] - v[without_A]

shapley_value_A = total_shapley_value / total_permutations
print("The Shapley value for player A is", shapley_value_A)
# %%
