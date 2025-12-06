module TensorizedFourierNeuralOperators

########   Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs   ########
#
# See arxiv.org/abs/2310.00120 for details.

using Lux
using NNlib: batched_mul
using NeuralOperators: FourierTransform, inverse, pad_zeros, transform, truncate_modes
using Random

# Tucker–Tensorized Spectral Convolution Layer
struct TuckerSpectralConv{D} <: Lux.AbstractLuxLayer
    channels_in::Int
    channels_out::Int
    rank_in::Int
    rank_out::Int
    rank_modes::NTuple{D,Int}
    fourier_transform::FourierTransform{ComplexF32,NTuple{D,Int}}
end

function TuckerSpectralConv(
    channels::Pair{Int,Int},
    modes::NTuple{D,Int};
    rank_in::Int,
    rank_out::Int,
    rank_modes::NTuple{D,Int},
    shift::Bool=false
) where {D}
    (channels_in, channels_out) = channels
    fourier_transform = FourierTransform{Float32}(modes, shift)
    return TuckerSpectralConv{D}(
        channels_in, channels_out, rank_in, rank_out, rank_modes, fourier_transform
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::TuckerSpectralConv{D}) where {D}
    fourier_transform = layer.fourier_transform
    T = eltype(fourier_transform)
    channels_in = layer.channels_in
    channels_out = layer.channels_out
    rank_in = layer.rank_in
    rank_out = layer.rank_out
    rank_modes = layer.rank_modes
    modes = fourier_transform.modes
    # simple Glorot-style scaling
    scale = sqrt(2.0 / (channels_in + channels_out))

    # core tensor ordering: (rank_out, rank_in, rank_modes...)
    core = scale .* glorot_uniform(rng, T, rank_out, rank_in, rank_modes...)
    U_in = glorot_uniform(rng, T, channels_in, rank_in)     # (ch_in × r_in)
    U_out = glorot_uniform(rng, T, channels_out, rank_out)   # (ch_out × r_out)
    # U_modes: (rank_dim × mode_dim)
    U_modes = ntuple(i -> glorot_uniform(rng, T, rank_modes[i], modes[i]), Val(D))

    params = (; core, U_modes, U_in, U_out)
    return params
end

function Lux.initialstates(rng::AbstractRNG, layer::TuckerSpectralConv)
    return (;) # stateless
end

function Lux.parameterlength(layer::TuckerSpectralConv{D}) where {D}
    rank_in = layer.rank_in
    rank_out = layer.rank_out
    rank_modes = layer.rank_modes
    modes = layer.fourier_transform.modes

    core_len = rank_out * rank_in * prod(rank_modes)
    U_in_len = layer.channels_in * rank_in
    U_out_len = layer.channels_out * rank_out
    U_modes_len = sum(rank_modes[d] * modes[d] for d in 1:D)
    len = core_len + U_in_len + U_out_len + U_modes_len
    return len
end

function Lux.statelength(layer::TuckerSpectralConv)
    return 0
end

# forward pass definition
# (spatial_dims..., channels_in, batches) -> (spatial_dims..., channels_out, batches)
function (conv::TuckerSpectralConv{D})(
    x::AbstractArray, params::NamedTuple, states::NamedTuple
) where {D}
    # x: (spatial_dims..., channels_in, batches)
    fourier_transform = conv.fourier_transform
    # apply discrete Fourier transform: spatial_dims -> freq_dims
    ω = transform(fourier_transform, x)                    # (freq_dims..., channels_in, batches)
    # truncate higher frequencies: freq_dims -> modes
    ω_truncated = truncate_modes(fourier_transform, ω)     # (modes..., channels_in, batches)

    # perform tensor contractions in truncated frequency space: channels_in -> channels_out
    y = compute_tensor_contractions(ω_truncated, params)   # (modes..., channels_out, batches)

    # pad truncated frequencies with zeros to restore original frequency dimensions: modes -> freq_dims
    pad_dims = size(ω)[1:D] .- size(y)[1:D]
    pad_tuple = expand_pad_dims(pad_dims)
    dims = ntuple(identity, Val(D))
    y_padded = pad_constant(y, pad_tuple, false; dims)     # (freq_dims..., channels_out, batches)
    # apply inverse discrete Fourier transform: freq_dims -> spatial_dims
    output = inverse(fourier_transform, y_padded)          # (spatial_dims..., channels_out, batches)
    return (output, states)
end

# (m₁ × ch_in × b) -> (m₁ × ch_out × b)
function compute_tensor_contractions(ω_truncated::DenseArray{<:Number,3}, params::NamedTuple)
    core = params.core         # (r_out × r_in × r₁)
    U_in = params.U_in         # (ch_in × r_in)
    U_out = params.U_out       # (ch_out × r_out)
    U_modes = params.U_modes   # (r₁ × m₁)

    (m₁, ch_in, b) = size(ω_truncated)
    (r_out, r_in, r₁) = size(core)
    ch_out = size(U_out, 1)

    # contract r₁ -> m₁
    core_flat₁ = reshape(core, r_out * r_in, r₁)        # (r_out⋅r_in × r₁)
    U₁ = U_modes[1]                                     # (r₁ × m₁)
    S_flat₁ = core_flat₁ * U₁                           # (r_out⋅r_in × m₁)
    S = reshape(S_flat₁, r_out, r_in, m₁)               # (r_out × r_in × m₁)

    # project input: contract ch_in -> r_in (batching over batches)
    U_in_fat = reshape(U_in, ch_in, r_in, 1)            # (ch_in × r_in × 1)
    ω_proj_flat = batched_mul(ω_truncated, U_in_fat)    # (m₁ × r_in × b)

    # spectral convolution: contract r_in -> r_out (batching over m₁)
    ω_proj_perm = permutedims(ω_proj_flat, (2, 3, 1))   # (r_in × b × m₁)
    y_flat = batched_mul(S, ω_proj_perm)                # (r_out × b × m₁)

    # project output: contract r_out -> ch_out (batching over m₁)
    U_out_fat = reshape(U_out, ch_out, r_out, 1)        # (ch_out × r_out × 1)
    output_flat = batched_mul(U_out_fat, y_flat)        # (ch_out × b × m₁)
    output = permutedims(output_flat, (3, 1, 2))        # (m₁ × ch_out × b)
    return output
end

# (m₁ × m₂ × ch_in × b) -> (m₁ × m₂ × ch_out × b)
function compute_tensor_contractions(ω_truncated::DenseArray{<:Number,4}, params::NamedTuple)
    core = params.core         # (r_out × r_in × r₁ × r₂)
    U_in = params.U_in         # (ch_in × r_in)
    U_out = params.U_out       # (ch_out × r_out)
    U_modes = params.U_modes   # (rₖ × mₖ)

    (m₁, m₂, ch_in, b) = size(ω_truncated)
    (r_out, r_in, r₁, r₂) = size(core)
    ch_out = size(U_out, 1)

    # contract r₁ -> m₁ (batching over r₂)
    core_flat₁ = reshape(core, r_out * r_in, r₁, r₂)    # (r_out⋅r_in × r₁ × r₂)
    U₁ = reshape(U_modes[1], r₁, m₁, 1)                 # (r₁ × m₁ × 1)
    S_flat₁ = batched_mul(core_flat₁, U₁)               # (r_out⋅r_in × m₁ × r₂)
    S₁ = reshape(S_flat₁, r_out, r_in, m₁, r₂)          # (r_out × r_in × m₁ × r₂)

    # contract r₂ -> m₂
    core_flat₂ = reshape(S₁, r_out * r_in * m₁, r₂)     # (r_out⋅r_in⋅m₁ × r₂)
    U₂ = U_modes[2]                                     # (r₂ × m₂)
    S_flat₂ = core_flat₂ * U₂                           # (r_out⋅r_in⋅m₁ × m₂)
    S = reshape(S_flat₂, r_out, r_in, m₁, m₂)           # (r_out × r_in × m₁ × m₂)

    # project input: contract ch_in -> r_in (batching over batches)
    ω_flat = reshape(ω_truncated, m₁ * m₂, ch_in, b)    # (m₁⋅m₂ × ch_in × b)
    U_in_fat = reshape(U_in, ch_in, r_in, 1)            # (ch_in × r_in × 1)
    ω_proj_flat = batched_mul(ω_flat, U_in_fat)         # (m₁⋅m₂ × r_in × b)

    # spectral convolution: contract r_in -> r_out (batching over m₁ × m₂)
    S_flat = reshape(S, r_out, r_in, m₁ * m₂)           # (r_out × r_in × m₁⋅m₂)
    ω_proj_perm = permutedims(ω_proj_flat, (2, 3, 1))   # (r_in × b × m₁⋅m₂)
    y_flat = batched_mul(S_flat, ω_proj_perm)           # (r_out × b × m₁⋅m₂)

    # project output: contract r_out -> ch_out (batching over m₁ × m₂)
    U_out_fat = reshape(U_out, ch_out, r_out, 1)        # (ch_out × r_out × 1)
    output_flat = batched_mul(U_out_fat, y_flat)        # (ch_out × b × m₁⋅m₂)
    output_perm = permutedims(output_flat, (3, 1, 2))   # (m₁⋅m₂ × ch_out × b)
    output = reshape(output_perm, m₁, m₂, ch_out, b)    # (m₁ × m₂ × ch_out × b)
    return output
end

# (m₁ × m₂ × m₃ × ch_in × b) -> (m₁ × m₂ × m₃ × ch_out × b)
function compute_tensor_contractions(ω_truncated::DenseArray{<:Number,5}, params::NamedTuple)
    core = params.core         # (r_out × r_in × r₁ × r₂ × r₃)
    U_in = params.U_in         # (ch_in × r_in)
    U_out = params.U_out       # (ch_out × r_out)
    U_modes = params.U_modes   # (rₖ × mₖ)

    (m₁, m₂, m₃, ch_in, b) = size(ω_truncated)
    (r_out, r_in, r₁, r₂, r₃) = size(core)
    ch_out = size(U_out, 1)

    # contract r₁ -> m₁ (batching over r₂ × r₃)
    core_flat₁ = reshape(core, r_out * r_in, r₁, r₂ * r₃)   # (r_out⋅r_in × r₁ × r₂⋅r₃)
    U₁ = reshape(U_modes[1], r₁, m₁, 1)                     # (r₁ × m₁ × 1)
    S_flat₁ = batched_mul(core_flat₁, U₁)                   # (r_out⋅r_in × m₁ × r₂⋅r₃)
    S₁ = reshape(S_flat₁, r_out, r_in, m₁, r₂, r₃)          # (r_out × r_in × m₁ × r₂ × r₃)

    # contract r₂ -> m₂ (batching over r₃)
    core_flat₂ = reshape(S₁, r_out * r_in * m₁, r₂, r₃)     # (r_out⋅r_in⋅m₁ × r₂ × r₃)
    U₂ = reshape(U_modes[2], r₂, m₂, 1)                     # (r₂ × m₂ × 1)
    S_flat₂ = batched_mul(core_flat₂, U₂)                   # (r_out⋅r_in⋅m₁ × m₂ × r₃)
    S₂ = reshape(S_flat₂, r_out, r_in, m₁, m₂, r₃)          # (r_out × r_in × m₁ × m₂ × r₃)

    # contract r₃ -> m₃
    core_flat₃ = reshape(S₂, r_out * r_in * m₁ * m₂, r₃)    # (r_out⋅r_in⋅m₁⋅m₂ × r₃)
    U₃ = U_modes[3]                                         # (r₃ × m₃)
    S_flat₃ = core_flat₃ * U₃                               # (r_out⋅r_in⋅m₁⋅m₂ × m₃)
    S = reshape(S_flat₃, r_out, r_in, m₁, m₂, m₃)           # (r_out × r_in × m₁ × m₂ × m₃)

    # project input: contract ch_in -> r_in (batching over batches)
    ω_flat = reshape(ω_truncated, m₁ * m₂ * m₃, ch_in, b)   # (m₁⋅m₂⋅m₃ × ch_in × b)
    U_in_fat = reshape(U_in, ch_in, r_in, 1)                # (ch_in × r_in × 1)
    ω_proj_flat = batched_mul(ω_flat, U_in_fat)             # (m₁⋅m₂⋅m₃ × r_in × b)

    # spectral convolution: contract r_in -> r_out (batching over m₁ × m₂ × m₃)
    S_flat = reshape(S, r_out, r_in, m₁ * m₂ * m₃)          # (r_out × r_in × m₁⋅m₂⋅m₃)
    ω_proj_perm = permutedims(ω_proj_flat, (2, 3, 1))       # (r_in × b × m₁⋅m₂⋅m₃)
    y_flat = batched_mul(S_flat, ω_proj_perm)               # (r_out × b × m₁⋅m₂⋅m₃)

    # project output: contract r_out -> ch_out (batching over m₁ × m₂ × m₃)
    U_out_fat = reshape(U_out, ch_out, r_out, 1)            # (ch_out × r_out × 1)
    output_flat = batched_mul(U_out_fat, y_flat)            # (ch_out × b × m₁⋅m₂⋅m₃)
    output_perm = permutedims(output_flat, (3, 1, 2))       # (m₁⋅m₂⋅m₃ × ch_out × b)
    output = reshape(output_perm, m₁, m₂, m₃, ch_out, b)    # (m₁ × m₂ × m₃ × ch_out × b)
    return output
end

end # module TensorizedFourierNeuralOperators
