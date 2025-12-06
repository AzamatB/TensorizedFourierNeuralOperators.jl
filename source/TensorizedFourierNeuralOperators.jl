module TensorizedFourierNeuralOperators

########   Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs   ########
#
# See arxiv.org/abs/2310.00120 for details.

using Lux
using FFTW
using Random
using NNlib: batched_mul, pad_constant
using NeuralOperators: FourierTransform, expand_pad_dims, inverse, transform, truncate_modes

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
    fourier_transform = FourierTransform{ComplexF32}(modes, shift)
    return TuckerSpectralConv{D}(
        channels_in, channels_out, rank_in, rank_out, rank_modes, fourier_transform
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::TuckerSpectralConv{D}) where {D}
    fourier_transform = layer.fourier_transform
    T = eltype(fourier_transform)
    C = complex(T)
    channels_in = layer.channels_in
    channels_out = layer.channels_out
    rank_in = layer.rank_in
    rank_out = layer.rank_out
    rank_modes = layer.rank_modes
    modes = fourier_transform.modes
    # simple Glorot-style scaling
    scale = sqrt(2.0f0 / (channels_in + channels_out))

    # core tensor ordering: (rank_out, rank_in, rank_modes...)
    core = scale .* glorot_uniform(rng, C, rank_out, rank_in, rank_modes...)
    U_in = glorot_uniform(rng, C, channels_in, rank_in)     # (ch_in × r_in)
    U_out = glorot_uniform(rng, C, channels_out, rank_out)   # (ch_out × r_out)
    # U_modes: (rank_dim × mode_dim)
    U_modes = ntuple(i -> glorot_uniform(rng, C, rank_modes[i], modes[i]), Val(D))

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
    output = inverse(fourier_transform, y_padded, x)          # (spatial_dims..., channels_out, batches)
    return (output, states)
end

# (m₁ × ch_in × b) -> (m₁ × ch_out × b)
function compute_tensor_contractions(x::AbstractArray{<:Number,3}, params::NamedTuple)
    core = params.core          # (r_out × r_in × r₁)
    U_in = params.U_in          # (ch_in × r_in)
    U_out = params.U_out        # (ch_out × r_out)
    U₁ = only(params.U_modes)   # (r₁ × m₁)

    (m₁, ch_in, b) = size(x)
    (r_out, r_in, r₁) = size(core)

    # contract r₁ -> m₁
    core_flat₁ = reshape(core, r_out * r_in, r₁)   # (r_out⋅r_in × r₁)
    S_flat₁ = core_flat₁ * U₁                      # (r_out⋅r_in × m₁)
    S = reshape(S_flat₁, r_out, r_in, m₁)          # (r_out × r_in × m₁)

    # project input: contract ch_in -> r_in (batching over batches)
    y = batched_mul(x, U_in)                       # (m₁ × r_in × b)

    # spectral convolution: contract r_in -> r_out (batching over m₁)
    y_perm = permutedims(y, (2, 3, 1))             # (r_in × b × m₁)
    z = batched_mul(S, y_perm)                     # (r_out × b × m₁)

    # project output: contract r_out -> ch_out (batching over m₁)
    output_flat = batched_mul(U_out, z)            # (ch_out × b × m₁)
    output = permutedims(output_flat, (3, 1, 2))   # (m₁ × ch_out × b)
    return output
end

# (m₁ × m₂ × ch_in × b) -> (m₁ × m₂ × ch_out × b)
function compute_tensor_contractions(x::AbstractArray{<:Number,4}, params::NamedTuple)
    core = params.core          # (r_out × r_in × r₁ × r₂)
    U_in = params.U_in          # (ch_in × r_in)
    U_out = params.U_out        # (ch_out × r_out)
    (U₁, U₂) = params.U_modes   # (rₖ × mₖ)

    (m₁, m₂, ch_in, b) = size(x)
    (r_out, r_in, r₁, r₂) = size(core)
    ch_out = size(U_out, 1)

    # contract r₁ -> m₁ (batching over r₂)
    core_flat₁ = reshape(core, r_out * r_in, r₁, r₂)    # (r_out⋅r_in × r₁ × r₂)
    S_flat₁ = batched_mul(core_flat₁, U₁)               # (r_out⋅r_in × m₁ × r₂)
    S₁ = reshape(S_flat₁, r_out, r_in, m₁, r₂)          # (r_out × r_in × m₁ × r₂)

    # contract r₂ -> m₂
    core_flat₂ = reshape(S₁, r_out * r_in * m₁, r₂)     # (r_out⋅r_in⋅m₁ × r₂)
    S_flat₂ = core_flat₂ * U₂                           # (r_out⋅r_in⋅m₁ × m₂)
    S = reshape(S_flat₂, r_out, r_in, m₁, m₂)           # (r_out × r_in × m₁ × m₂)

    # project input: contract ch_in -> r_in (batching over batches)
    x_flat = reshape(x, m₁ * m₂, ch_in, b)              # (m₁⋅m₂ × ch_in × b)
    y = batched_mul(x_flat, U_in)                       # (m₁⋅m₂ × r_in × b)

    # spectral convolution: contract r_in -> r_out (batching over m₁ × m₂)
    S_flat = reshape(S, r_out, r_in, m₁ * m₂)           # (r_out × r_in × m₁⋅m₂)
    y_perm = permutedims(y, (2, 3, 1))                  # (r_in × b × m₁⋅m₂)
    z = batched_mul(S_flat, y_perm)                     # (r_out × b × m₁⋅m₂)

    # project output: contract r_out -> ch_out (batching over m₁ × m₂)
    output_flat = batched_mul(U_out, z)                 # (ch_out × b × m₁⋅m₂)
    output_perm = permutedims(output_flat, (3, 1, 2))   # (m₁⋅m₂ × ch_out × b)
    output = reshape(output_perm, m₁, m₂, ch_out, b)    # (m₁ × m₂ × ch_out × b)
    return output
end

# (m₁ × m₂ × m₃ × ch_in × b) -> (m₁ × m₂ × m₃ × ch_out × b)
function compute_tensor_contractions(x::AbstractArray{<:Number,5}, params::NamedTuple)
    core = params.core              # (r_out × r_in × r₁ × r₂ × r₃)
    U_in = params.U_in              # (ch_in × r_in)
    U_out = params.U_out            # (ch_out × r_out)
    (U₁, U₂, U₃) = params.U_modes   # (rₖ × mₖ)

    (m₁, m₂, m₃, ch_in, b) = size(x)
    (r_out, r_in, r₁, r₂, r₃) = size(core)
    ch_out = size(U_out, 1)

    # contract r₁ -> m₁ (batching over r₂ × r₃)
    core_flat₁ = reshape(core, r_out * r_in, r₁, r₂ * r₃)   # (r_out⋅r_in × r₁ × r₂⋅r₃)
    S_flat₁ = batched_mul(core_flat₁, U₁)                   # (r_out⋅r_in × m₁ × r₂⋅r₃)
    S₁ = reshape(S_flat₁, r_out, r_in, m₁, r₂, r₃)          # (r_out × r_in × m₁ × r₂ × r₃)

    # contract r₂ -> m₂ (batching over r₃)
    core_flat₂ = reshape(S₁, r_out * r_in * m₁, r₂, r₃)     # (r_out⋅r_in⋅m₁ × r₂ × r₃)
    S_flat₂ = batched_mul(core_flat₂, U₂)                   # (r_out⋅r_in⋅m₁ × m₂ × r₃)
    S₂ = reshape(S_flat₂, r_out, r_in, m₁, m₂, r₃)          # (r_out × r_in × m₁ × m₂ × r₃)

    # contract r₃ -> m₃
    core_flat₃ = reshape(S₂, r_out * r_in * m₁ * m₂, r₃)    # (r_out⋅r_in⋅m₁⋅m₂ × r₃)
    S_flat₃ = core_flat₃ * U₃                               # (r_out⋅r_in⋅m₁⋅m₂ × m₃)
    S = reshape(S_flat₃, r_out, r_in, m₁, m₂, m₃)           # (r_out × r_in × m₁ × m₂ × m₃)

    # project input: contract ch_in -> r_in (batching over batches)
    x_flat = reshape(x, m₁ * m₂ * m₃, ch_in, b)             # (m₁⋅m₂⋅m₃ × ch_in × b)
    y = batched_mul(x_flat, U_in)                           # (m₁⋅m₂⋅m₃ × r_in × b)

    # spectral convolution: contract r_in -> r_out (batching over m₁ × m₂ × m₃)
    S_flat = reshape(S, r_out, r_in, m₁ * m₂ * m₃)          # (r_out × r_in × m₁⋅m₂⋅m₃)
    y_perm = permutedims(y, (2, 3, 1))                      # (r_in × b × m₁⋅m₂⋅m₃)
    z = batched_mul(S_flat, y_perm)                         # (r_out × b × m₁⋅m₂⋅m₃)

    # project output: contract r_out -> ch_out (batching over m₁ × m₂ × m₃)
    output_flat = batched_mul(U_out, z)                     # (ch_out × b × m₁⋅m₂⋅m₃)
    output_perm = permutedims(output_flat, (3, 1, 2))       # (m₁⋅m₂⋅m₃ × ch_out × b)
    output = reshape(output_perm, m₁, m₂, m₃, ch_out, b)    # (m₁ × m₂ × m₃ × ch_out × b)
    return output
end

# expand Tucker core tensor into full tensor via mode products with factor matrices
function expand_tucker_core_tensor(
    core::DenseArray{Complex{R}}, U_modes::NTuple{D,DenseMatrix{R}}
) where {D,R<:Real}
    modes = size.(U_modes, 2)
    # core: (r_out, r_in, dims...)
    for d in 1:D
        # k-mode tensor product via matricization + batched multiplication
        k = d + 2
        dims = size(core)
        ranks_before = dims[1:(k - 1)]
        ranks_after = dims[(k + 1):end]

        core_flat = reshape(core, prod(ranks_before), dims[k], prod(ranks_after))
        S_flat = batched_mul(core_flat, U_modes[d])
        core = reshape(S_flat, ranks_before..., modes[d], ranks_after...)
    end
    return core
end

end # module TensorizedFourierNeuralOperators
