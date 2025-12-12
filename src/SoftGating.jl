"""
    SoftGating{D}(channels)

Soft-gating skip connection for tensors of shape `(spatial_dims..., channels, batch)`.
Computes `x .* w`, where `w` is a learnable per-channel scale broadcast across spatial and batch dims.
"""
struct SoftGating{D} <: Lux.AbstractLuxLayer
    channels::Int
end

function Lux.initialparameters(rng::AbstractRNG, layer::SoftGating{D}) where {D}
    # match neuraloperator implementation: initialize to identity gating (all ones)
    spatial_dims = ntuple(_ -> 1, static(D))
    weights = ones(Float32, spatial_dims..., layer.channels, 1)
    params = (; weights)
    return params
end

function Lux.initialstates(rng::AbstractRNG, layer::SoftGating)
    return (;) # stateless
end

function Lux.parameterlength(layer::SoftGating)
    return layer.channels
end

function Lux.statelength(::SoftGating)
    return 0
end

function (layer::SoftGating)(
    x::AbstractArray{<:Number}, params::NamedTuple, states::NamedTuple
)
    # x: (spatial_dims..., channels, batch)
    weights = params.weights  # (1,...,1, channels, 1), broadcastable to x
    y = x .* weights         # (spatial_dims..., channels, batch)
    return (y, states)
end
