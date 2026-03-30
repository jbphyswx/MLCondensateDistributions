using Lux: Lux
using Random: Random

"""
    CondensateMLP(input_dim::Int, output_dim::Int, hidden_dims::Vector{Int})

Constructs a Lux MLP for condensate prediction.
Uses ReLU activations for hidden layers and Softplus for the output 
to ensure non-negative condensate values.

    Input: 
    Output: 
"""
function CondensateMLP(input_dim::Int, output_dim::Int, hidden_dims::Vector{Int})
    layers = []
    prev_dim = input_dim
    
    for h_dim in hidden_dims
        push!(layers, Lux.Dense(prev_dim => h_dim, Lux.relu))
        prev_dim = h_dim
    end
    
    # Output layer with softplus to ensure non-negativity
    push!(layers, Lux.Dense(prev_dim => output_dim, Lux.softplus))
    
    return Lux.Chain(layers...)
end


"""
    ToDo: A transformer that predicts entire columns at once
"""