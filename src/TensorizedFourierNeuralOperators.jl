module TensorizedFourierNeuralOperators

########   Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs   ########
#
# See arxiv.org/abs/2310.00120 for details.

export FactorizedSpectralConv

using Lux
using FFTW
using Random
using NNlib: batched_mul, pad_constant
using NeuralOperators: FourierTransform, expand_pad_dims, inverse, transform, truncate_modes

# Tucker–Factorized Spectral Convolution Layer
struct FactorizedSpectralConv{D} <: Lux.AbstractLuxLayer
    channels_in::Int
    channels_out::Int
    rank_ratio::Float32
    fourier_transform::FourierTransform{ComplexF32,NTuple{D,Int}}
end

function FactorizedSpectralConv(
    channels::Pair{Int,Int},
    modes::NTuple{D,Int};
    rank_ratio::Float32=0.5f0,
    shift::Bool=false
) where {D}
    (channels_in, channels_out) = channels
    fourier_transform = FourierTransform{ComplexF32}(modes, shift)
    return FactorizedSpectralConv{D}(
        channels_in, channels_out, rank_ratio, fourier_transform
    )
end

function Lux.initialparameters(rng::AbstractRNG, conv::FactorizedSpectralConv{D}) where {D}
    fourier_transform = conv.fourier_transform
    T = eltype(fourier_transform)
    C = complex(T)
    modes = fourier_transform.modes
    channels_in = conv.channels_in
    channels_out = conv.channels_out
    rank_ratio = conv.rank_ratio

    # determine ranks for Tucker decomposition
    (rank_in, rank_out, rank_modes) = compute_tucker_rank_dims(channels_in, channels_out, modes, rank_ratio)

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

function Lux.initialstates(rng::AbstractRNG, conv::FactorizedSpectralConv)
    return (;) # stateless
end

function Lux.parameterlength(conv::FactorizedSpectralConv{D}) where {D}
    modes = conv.fourier_transform.modes
    channels_in = conv.channels_in
    channels_out = conv.channels_out
    rank_ratio = conv.rank_ratio

    # determine ranks for Tucker decomposition
    (rank_in, rank_out, rank_modes) = compute_tucker_rank_dims(channels_in, channels_out, modes, rank_ratio)

    core_len = rank_out * rank_in * prod(rank_modes)
    U_in_len = conv.channels_in * rank_in
    U_out_len = conv.channels_out * rank_out
    U_modes_len = sum(rank_modes[d] * modes[d] for d in 1:D)
    len = core_len + U_in_len + U_out_len + U_modes_len
    return len
end

function Lux.statelength(conv::FactorizedSpectralConv)
    return 0
end

# forward pass definition
# (spatial_dims..., channels_in, batch) -> (spatial_dims..., channels_out, batch)
function (conv::FactorizedSpectralConv{D})(
    x::AbstractArray, params::NamedTuple, states::NamedTuple
) where {D}
    # x: (spatial_dims..., channels_in, batch)
    fourier_transform = conv.fourier_transform
    # apply discrete Fourier transform: spatial_dims -> freq_dims
    ω = transform(fourier_transform, x)                    # (freq_dims..., channels_in, batch)
    # truncate higher frequencies: freq_dims -> modes
    ω_truncated = truncate_modes(fourier_transform, ω)     # (modes..., channels_in, batch)

    # (r_out, r_in, rank_modes...) -> (r_out, r_in, modes...)
    S = expand_tucker_core_tensor(params.core, params.U_modes)

    # perform tensor contractions in truncated frequency space:
    # (modes..., ch_in, b) -> (modes..., ch_out, b)
    y = compute_tensor_contractions(ω_truncated, params.U_in, params.U_out, S)

    # pad truncated frequencies with zeros to restore original frequency dimensions: modes -> freq_dims
    pad_dims = size(ω)[1:D] .- size(y)[1:D]
    pad_tuple = expand_pad_dims(pad_dims)
    dims = ntuple(identity, Val(D))
    y_padded = pad_constant(y, pad_tuple, false; dims)     # (freq_dims..., channels_out, batch)
    # apply inverse discrete Fourier transform: freq_dims -> spatial_dims
    output = inverse(fourier_transform, y_padded, x)       # (spatial_dims..., channels_out, batch)
    return (output, states)
end

function compute_tucker_rank_dims(
    channels_in::Int,
    channels_out::Int,
    modes::NTuple{D,Int},
    rank_ratio::Float32
) where {D}
    rank_in = max(1, floor(Int, channels_in * rank_ratio))
    rank_out = max(1, floor(Int, channels_out * rank_ratio))
    rank_modes = ntuple(i -> max(1, floor(Int, modes[i] * rank_ratio)), Val(D))
    return (rank_in, rank_out, rank_modes)
end

# (modes..., ch_in, b) -> (modes..., ch_out, b)
function compute_tensor_contractions(
    x::AbstractArray{<:Number,N},       # (modes..., ch_in, b)
    U_in::DenseMatrix{C},               # (ch_in, r_in)
    U_out::DenseMatrix{C},              # (ch_out, r_out)
    S::AbstractArray{C,N}               # (r_out, r_in, modes...)
) where {C<:Complex,N}
    (ch_in, r_in) = size(U_in)
    (ch_out, r_out) = size(U_out)
    dims = size(x)
    b = dims[end]
    modes = NTuple{N-2,Int}(ntuple(i -> dims[i], Val(N - 2)))

    # project input: contract ch_in -> r_in (batching over batch)
    x_flat = reshape(x, :, ch_in, b)                     # (prod(modes), ch_in, b)
    y = batched_mul(x_flat, U_in)                        # (prod(modes), r_in, b)

    # spectral convolution: contract r_in -> r_out (batching over modes)
    S_flat = reshape(S, r_out, r_in, :)                  # (r_out, r_in, prod(modes))
    y_perm = permutedims(y, (2, 3, 1))                   # (r_in, b, prod(modes))
    z = batched_mul(S_flat, y_perm)                      # (r_out, b, prod(modes))

    # project output: contract r_out -> ch_out (batching over modes)
    output_flat = batched_mul(U_out, z)                  # (ch_out, b, prod(modes))
    output_perm = permutedims(output_flat, (3, 1, 2))    # (prod(modes), ch_out, b)
    output = reshape(output_perm, modes..., ch_out, b)   # (modes..., ch_out, b)
    return output
end

struct ModeKProduct{K} end

# mode-k tensor-matrix product via matricization followed by batched multiplication
function (::ModeKProduct{K})(
    tensor::AbstractArray{N,D}, matrix::AbstractMatrix{N}
) where {K,N<:Number,D}
    dims = size(tensor)
    dims_before = NTuple{K-1,Int}(ntuple(i -> dims[i], Val(K-1)))
    dims_after  = NTuple{D-K,Int}(ntuple(i -> dims[K + i], Val(D-K)))
    (dim_in, dim_out) = size(matrix)

    tensor_flat = reshape(tensor, prod(dims_before), dim_in, prod(dims_after))
    result_flat = batched_mul(tensor_flat, matrix)
    result = reshape(result_flat, dims_before..., dim_out, dims_after...)
    return result
end

# 1D case: (r_out × r_in × r₁) -> (r_out × r_in × m₁)
function expand_tucker_core_tensor(
    core::AbstractArray{C,3},                   # (r_out × r_in × r₁)
    U_modes::NTuple{1,DenseMatrix{C}}           # (rₖ × mₖ)
) where {C<:Complex}
    # contract r₁ -> m₁
    mode_3_product = ModeKProduct{3}()
    S = mode_3_product(core, U_modes[1])        # (r_out × r_in × m₁)
    return S
end

# 2D case: (r_out × r_in × r₁ × r₂) -> (r_out × r_in × m₁ × m₂)
function expand_tucker_core_tensor(
    core::AbstractArray{C,4},                   # (r_out × r_in × r₁ × r₂)
    U_modes::NTuple{2,DenseMatrix{C}}           # (rₖ × mₖ)
) where {C<:Complex}
    # contract r₁ -> m₁ (batching over r₂)
    mode_3_product = ModeKProduct{3}()
    core₂ = mode_3_product(core, U_modes[1])    # (r_out × r_in × m₁ × r₂)
    # contract r₂ -> m₂
    mode_4_product = ModeKProduct{4}()
    S = mode_4_product(core₂, U_modes[2])       # (r_out × r_in × m₁ × m₂)
    return S
end

# 3D case: (r_out × r_in × r₁ × r₂ × r₃) -> (r_out × r_in × m₁ × m₂ × m₃)
function expand_tucker_core_tensor(
    core::AbstractArray{C,5},                   # (r_out × r_in × r₁ × r₂ × r₃)
    U_modes::NTuple{3,DenseMatrix{C}}           # (rₖ × mₖ)
) where {C<:Complex}
    # contract r₁ -> m₁ (batching over r₂ × r₃)
    mode_3_product = ModeKProduct{3}()
    core₂ = mode_3_product(core, U_modes[1])    # (r_out × r_in × m₁ × r₂ × r₃)
    # contract r₂ -> m₂ (batching over r₃)
    mode_4_product = ModeKProduct{4}()
    core₃ = mode_4_product(core₂, U_modes[2])   # (r_out × r_in × m₁ × m₂ × r₃)
    # contract r₃ -> m₃
    mode_5_product = ModeKProduct{5}()
    S = mode_5_product(core₃, U_modes[3])       # (r_out × r_in × m₁ × m₂ × m₃)
    return S
end

# # expand Tucker core tensor into full tensor via mode products with factor matrices
# function expand_tucker_core_tensor(
#     core::AbstractArray{N,D},           # (r_out, r_in, dims...)
#     U_modes::NTuple{D,DenseMatrix{N}}   # (rₖ × mₖ)
# ) where {N<:Number,D}
#     for d in 1:D
#         mode_k_product = ModeKProduct{d+2}()
#         core = mode_k_product(core, U_modes[d])
#     end
#     return core
# end

end # module TensorizedFourierNeuralOperators
