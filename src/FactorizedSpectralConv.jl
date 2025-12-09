# Tucker–Factorized Spectral Convolution Layer
struct FactorizedSpectralConv{D} <: Lux.AbstractLuxLayer
    channels_in::Int
    channels_out::Int
    modes::NTuple{D,Int}
    rank_ratio::Float32
end

function FactorizedSpectralConv(
    channels::Pair{Int,Int}, modes::NTuple{D,Int}; rank_ratio::Float32=0.5f0
) where {D}
    (channels_in, channels_out) = channels
    return FactorizedSpectralConv{D}(channels_in, channels_out, modes, rank_ratio)
end

function Lux.initialparameters(rng::AbstractRNG, layer::FactorizedSpectralConv{D}) where {D}
    channels_in = layer.channels_in
    channels_out = layer.channels_out
    modes = 2 .* layer.modes .+ 1  # account for 0th and negative frequencies
    rank_ratio = layer.rank_ratio
    # determine ranks for Tucker decomposition
    (rank_in, rank_out, rank_modes) = compute_tucker_rank_dims(channels_in, channels_out, modes, rank_ratio)

    # core tensor ordering: (rank_out, rank_in, rank_modes...)
    core  = glorot_uniform(rng, ComplexF32, rank_out, rank_in, rank_modes...)
    U_in  = glorot_uniform(rng, ComplexF32, channels_in, rank_in)     # (ch_in × r_in)
    U_out = glorot_uniform(rng, ComplexF32, channels_out, rank_out)   # (ch_out × r_out)
    # U_modes: (rank_dim × mode_dim)
    U_modes = ntuple(i -> glorot_uniform(rng, ComplexF32, rank_modes[i], modes[i]), Val(D))

    params = (; core, U_modes, U_in, U_out)
    return params
end

function Lux.initialstates(rng::AbstractRNG, layer::FactorizedSpectralConv)
    return (;) # stateless
end

function Lux.parameterlength(layer::FactorizedSpectralConv{D}) where {D}
    channels_in = layer.channels_in
    channels_out = layer.channels_out
    modes = 2 .* layer.modes .+ 1  # account for 0th and negative frequencies
    rank_ratio = layer.rank_ratio
    # determine ranks for Tucker decomposition
    (rank_in, rank_out, rank_modes) = compute_tucker_rank_dims(channels_in, channels_out, modes, rank_ratio)

    core_len = rank_out * rank_in * prod(rank_modes)
    U_in_len = layer.channels_in * rank_in
    U_out_len = layer.channels_out * rank_out
    U_modes_len = sum(rank_modes[d] * modes[d] for d in 1:D)
    len = core_len + U_in_len + U_out_len + U_modes_len
    return len
end

function Lux.statelength(layer::FactorizedSpectralConv)
    return 0
end

# forward pass definition
# (spatial_dims..., channels_in, batch) -> (spatial_dims..., channels_out, batch)
function (layer::FactorizedSpectralConv{D})(
    x::AbstractArray,                               # (spatial_dims..., channels_in, batch)
    params::NamedTuple,
    states::NamedTuple
) where {D}
    # apply discrete Fourier transform: spatial_dims -> modes
    # (freq_dims..., channels_in, batch), (modes..., channels_in, batch)
    (ω, pad) = transform_and_truncate(layer, x)
    # (rank_out, rank_in, rank_modes...) -> (rank_out, rank_in, modes...)
    S = expand_tucker_core_tensor(params.core, params.U_modes)
    # perform tensor contractions in truncated frequency space:
    # (modes..., channels_in, batch) -> (modes..., channels_out, batch)
    y = compute_tensor_contractions(ω, params.U_in, params.U_out, S)
    # pad truncated frequencies with zeros to restore original frequency dimensions: modes -> freq_dims
    y_padded = pad_constant(y, pad, zero(eltype(y)); dims=1:D) # (freq_dims..., channels_out, batch)
    # apply inverse discrete Fourier transform: freq_dims -> spatial_dims
    output = inverse(layer, y_padded, x)                       # (spatial_dims..., channels_out, batch)
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

function left_slice(k::Int)
    stop = 2k + 1
    # 0th mode followed by 2k positive modes = (2k + 1) elements
    return 1:stop
end

function center_slice(len::Int, k::Int)
    center = len ÷ 2 + 1
    start = center - k
    stop = center + k
    # 0th mode at the center + k negative modes before + k positive modes after = (2k + 1) elements
    return start:stop
end

function compute_padding(shape::NTuple{D,Int}, slices::NTuple{D,UnitRange{Int}}) where {D}
    pad = NTuple{2D,Int}(ntuple(Val(2D)) do n
        d = (n + 1) ÷ 2
        slice = slices[d]
        isodd(n) ? (slice.start - 1) : (shape[d] - slice.stop)
    end)
    return pad
end

# complex-valued version
function transform_and_truncate(
    layer::FactorizedSpectralConv{D},
    x::AbstractArray{C}                              # (spatial_dims..., channels, batch)
) where {D,C<:Complex}
    dims = 1:D
    # compute discrete Fourier transform: spatial_dims -> freq_dims
    ω = fft(x, dims)                                 # (freq_dims..., channels, batch)
    # shift along every dimension
    ω_shifted = fftshift(ω, dims)                    # (freq_dims..., channels, batch)
    # take center crops in all dimensions
    shape_ω = size(ω_shifted)[dims]
    modes = layer.modes
    slices = NTuple{D,UnitRange{Int}}(ntuple(d -> center_slice(shape_ω[d], modes[d]), Val(D)))
    # truncate higher frequencies: freq_dims -> modes
    ω_truncated = view(ω_shifted, slices..., :, :)   # (modes..., channels, batch)
    pad = compute_padding(shape_ω, slices)
    return (ω_truncated, pad)
end

# real-valued version
function transform_and_truncate(
    layer::FactorizedSpectralConv{D},
    x::AbstractArray{R}                              # (spatial_dims..., channels, batch)
) where {D,R<:Real}
    dims = 1:D
    # since input is real-valued use rfft to take advantage of skew-symmetry
    ω = rfft(x, dims)                                # (freq_dims..., channels, batch)
    # shift along every dimension except the 1st
    ω_shifted = fftshift(ω, 2:D)
    # take 1:k₁ in dim 1 and center crops in the rest
    shape_ω = size(ω_shifted)[dims]
    modes = layer.modes
    slices = NTuple{D,UnitRange{Int}}(ntuple(
        d -> (d == 1) ? left_slice(modes[d]) : center_slice(shape_ω[d], modes[d]), Val(D)
    ))
    # truncate higher frequencies: freq_dims -> modes
    ω_truncated = view(ω_shifted, slices..., :, :)   # (modes..., channels, batch)
    pad = compute_padding(shape_ω, slices)
    return (ω_truncated, pad)
end

# real-valued 1D version
function transform_and_truncate(
    layer::FactorizedSpectralConv{1},
    x::AbstractArray{R,3}                            # (spatial_dim, channels, batch)
) where {R<:Real}
    # since input is real-valued, use rfft, to take advantage of skew-symmetry
    ω = rfft(x, 1)                                   # (freq_dim, channels, batch)
    # take 1:(2k + 1)
    shape_ω = size(ω)[1:1]
    slice = left_slice(layer.modes[1])
    # truncate higher frequencies: freq_dim -> modes
    ω_truncated = view(ω, slice, :, :)               # (modes, channels, batch)
    pad = compute_padding(shape_ω, (slice,))
    return (ω_truncated, pad)
end

function inverse(
    layer::FactorizedSpectralConv{D},
    ω_shifted::AbstractArray{C,N},                  # (freq_dims..., channels_out, batch)
    x::AbstractArray{C,N}                           # (spatial_dims..., channels_in, batch)
) where {D,C<:Complex,N}
    ω = ifftshift(ω_shifted, 1:D)
    y = ifft(ω, 1:D)                                # (spatial_dims..., channels_out, batch)
    return y
end

function inverse(
    layer::FactorizedSpectralConv{D},
    ω_shifted::AbstractArray{C,N},                  # (freq_dims..., channels_out, batch)
    x::AbstractArray{R,N}                           # (spatial_dims..., channels_in, batch)
) where {D,C<:Complex,R<:Real,N}
    ω = ifftshift(ω_shifted, 2:D)
    y_c = irfft(ω, size(x, 1), 1:D)                 # (spatial_dims..., channels_out, batch)
    y = real(y_c)
    return y
end

function inverse(
    layer::FactorizedSpectralConv{1},
    ω::AbstractArray{C,3},                           # (freq_dim, channels_out, batch)
    x::AbstractArray{R,3}                            # (spatial_dim, channels_in, batch)
) where {C<:Complex,R<:Real}
    y_c = irfft(ω, size(x, 1), 1)                    # (spatial_dim, channels_out, batch)
    y = real(y_c)
    return y
end

# (modes..., ch_in, b) -> (modes..., ch_out, b)
function compute_tensor_contractions(
    x::AbstractArray{C,N},                                    # (modes..., ch_in, b)
    U_in::DenseMatrix{C},                                     # (ch_in, r_in)
    U_out::DenseMatrix{C},                                    # (ch_out, r_out)
    S::AbstractArray{C,N}                                     # (r_out, r_in, modes...)
) where {C<:Complex,N}
    (ch_in, r_in) = size(U_in)
    (ch_out, r_out) = size(U_out)
    dims = size(x)
    b = dims[end]
    modes = NTuple{N-2,Int}(ntuple(i -> dims[i], Val(N - 2)))

    # project input: contract ch_in -> r_in (batching over batch)
    x_flat = reshape(x, :, ch_in, b)                          # (prod(modes), ch_in, b)
    y_flat = batched_mul(x_flat, U_in)                        # (prod(modes), r_in, b)
    y = reshape(y_flat, modes..., r_in, b)                    # (modes..., r_in, b)
    y_perm = permute_mode_dims(y)                             # (r_in, b, modes...)
    y_perm_flat = reshape(y_perm, r_in, b, :)                 # (r_in, b, prod(modes))

    # spectral convolution: contract r_in -> r_out (batching over modes)
    S_flat = reshape(S, r_out, r_in, :)                       # (r_out, r_in, prod(modes))
    z = batched_mul(S_flat, y_perm_flat)                      # (r_out, b, prod(modes))

    # project output: contract r_out -> ch_out (batching over modes)
    z_flat = reshape(z, r_out, :)                             # (r_out, b⋅prod(modes))
    output_flat = U_out * z_flat                              # (ch_out, b⋅prod(modes))
    output_perm = reshape(output_flat, ch_out, b, modes...)   # (ch_out, b, modes...)
    output = unpermute_mode_dims(output_perm)                 # (modes..., ch_out, b)
    return output
end

function permute_mode_dims(x::AbstractArray{T,3}) where {T}
    # (m₁, r_in, b) -> (r_in, b, m₁)
    x_perm = permutedims(x, (2, 3, 1))
    return x_perm
end
function permute_mode_dims(x::AbstractArray{T,4}) where {T}
    # (m₁, m₂, r_in, b) -> (r_in, b, m₁, m₂)
    x_perm = permutedims(x, (3, 4, 1, 2))
    return x_perm
end
function permute_mode_dims(x::AbstractArray{T,5}) where {T}
    # (m₁, m₂, m₃, r_in, b) -> (r_in, b, m₁, m₂, m₃)
    x_perm = permutedims(x, (4, 5, 1, 2, 3))
    return x_perm
end

function unpermute_mode_dims(x_perm::AbstractArray{T,3}) where {T}
    # (r_out, b, m₁) -> (m₁, r_out, b)
    x = permutedims(x_perm, (3, 1, 2))
    return x
end
function unpermute_mode_dims(x_perm::AbstractArray{T,4}) where {T}
    # (r_out, b, m₁, m₂) -> (m₁, m₂, r_out, b)
    x = permutedims(x_perm, (3, 4, 1, 2))
    return x
end
function unpermute_mode_dims(x_perm::AbstractArray{T,5}) where {T}
    # (r_out, b, m₁, m₂, m₃) -> (m₁, m₂, m₃, r_out, b)
    x = permutedims(x_perm, (3, 4, 5, 1, 2))
    return x
end

struct ModeKProduct{K} end

# mode-k tensor-matrix product via matricization followed by batched multiplication
function (::ModeKProduct{K})(
    tensor::AbstractArray{T,D}, matrix::AbstractMatrix{T}
) where {K,T<:Number,D}
    dims = size(tensor)
    dims_before = NTuple{K-1,Int}(ntuple(i -> dims[i],   Val(K-1)))
    dims_after  = NTuple{D-K,Int}(ntuple(i -> dims[K+i], Val(D-K)))
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
#     core::AbstractArray{T,D},           # (r_out, r_in, dims...)
#     U_modes::NTuple{D,DenseMatrix{T}}   # (rₖ × mₖ)
# ) where {T<:Number,D}
#     for d in 1:D
#         mode_k_product = ModeKProduct{d+2}()
#         core = mode_k_product(core, U_modes[d])
#     end
#     return core
# end
