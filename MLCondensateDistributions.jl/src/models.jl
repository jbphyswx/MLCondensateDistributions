using Lux: Lux
using Random: Random

"""
    CondensateMLP(input_dim::Int, output_dim::Int, hidden_dims::Vector{Int}; output_activation = Lux.identity)

Construct a feed-forward Lux MLP used by the training pipeline.

Behavior:
- Hidden layers use `relu` activations.
- Output layer activation is configurable (default linear/identity).

Arguments:
- `input_dim`: number of input features.
- `output_dim`: number of model outputs (depends on target mapping).
- `hidden_dims`: widths of hidden dense layers.
- `output_activation`: activation for the final layer.

Returns:
- `Lux.Chain` model.

Note:
- The current `train_lux.jl` pipeline uses standardized targets and therefore
    expects a linear output activation.
"""
function CondensateMLP(input_dim::Int, output_dim::Int, hidden_dims::Vector{Int}; output_activation = Lux.identity)
    layers = []
    prev_dim = input_dim
    
    for h_dim in hidden_dims
        push!(layers, Lux.Dense(prev_dim => h_dim, Lux.relu))
        prev_dim = h_dim
    end
    
    push!(layers, Lux.Dense(prev_dim => output_dim, output_activation))
    
    return Lux.Chain(layers...)
end


"""
    ToDo: A transformer that predicts entire columns at once

    This is a project for later though because right now in the we store .arrow files for each row with condensate, not full columns.
    And even for the full column version, we probably still want to cut out like the stratosphere etc, any excess unnecessary data.

    The idea here though is that e.g. covariances at cloud top are intimately linked to what's going on beneat and the motions that generated them.
    So pointwise calculations will always be lacking. and convection is intimately tied to e.g. the surface and its fluxes so looking at the whole column and learning the structure of the column as a whole will be really important.

"""


"""
    ToDo: We'd like to do things like the Assumed Double Gaussian Closure.

    A NN that predicts the parameters of an N Gaussian combination, which is then used to reconstruct the PDF and compute moments.
    This would be a more direct way to predict the full PDF, and would allow us to leverage the known structure of the PDF in the loss function.
    It would also probably be much faster and extensible to use inside a climate model

    it's a distribution over w, qt, ql, qi, θli etc, and at the very least we can aim to support 1 2 and 3 moments


    Unlike ADG and SHOC-like closures, which only do sat adjust, we would like to support a) direct support for condensate, rather than from the distirbution, b) both liq and ice, and c) direct prescriptions for w'ql' and w'qi' relations, rather than backing out from qt.
"""