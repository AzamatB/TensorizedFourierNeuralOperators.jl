# Fourier Neural Operator Block with Tucker–Factorized Spectral Convolution
struct FourierNeuralOperatorBlock{
    D,
    ChannelMLP <: Lux.AbstractLuxLayer,
    Skip₁ <: Lux.AbstractLuxLayer,
    Norm₁ <: Lux.AbstractLuxLayer,
    Norm₂ <: Lux.AbstractLuxLayer
} <: Lux.AbstractLuxContainerLayer{(:spectral_conv, :channel_mlp, :skip₁, :skip₂, :norm₁, :norm₂)}
    spectral_conv::FactorizedSpectralConv{D}
    channel_mlp::ChannelMLP
    skip₁::Skip₁
    skip₂::SoftGating{D}
    norm₁::Norm₁
    norm₂::Norm₂
end

function FourierNeuralOperatorBlock(
    channels::Pair{Int,Int}, modes::NTuple{D,Int}; rank_ratio::Float32=0.5f0
) where {D}
    (channels_in, channels_out) = channels
    spectral_conv = FactorizedSpectralConv(channels, modes; rank_ratio)
    # 2-layer channel MLP with GeLU activation in between
    pointwise_kernel = ntuple(_ -> 1, Val(D))
    channels = (channels_out => channels_out)
    channel_mlp = Chain(
        Conv(pointwise_kernel, channels, gelu),
        Conv(pointwise_kernel, channels)
    )
    # channelwise linear skip layer as a pointwise convolutional layer
    skip₁ = Conv(pointwise_kernel, channels)
    skip₂ = SoftGating{D}(channels_out)
    norm₁ = GroupNorm(channels_out, 1)
    norm₂ = GroupNorm(channels_out, 1)

    ChannelMLP = typeof(channel_mlp)
    Skip₁ = typeof(skip₁)
    Norm₁ = typeof(norm₁)
    Norm₂ = typeof(norm₂)
    return FourierNeuralOperatorBlock{D, ChannelMLP, Skip₁, Norm₁, Norm₂}(
        spectral_conv, channel_mlp, skip₁, skip₂, norm₁, norm₂
    )
end

function Lux.initialparameters(rng::AbstractRNG, layer::FourierNeuralOperatorBlock)
    spectral_conv = Lux.initialparameters(rng, layer.spectral_conv)
    channel_mlp = Lux.initialparameters(rng, layer.channel_mlp)
    skip₁ = Lux.initialparameters(rng, layer.skip₁)
    skip₂ = Lux.initialparameters(rng, layer.skip₂)
    norm₁ = Lux.initialparameters(rng, layer.norm₁)
    norm₂ = Lux.initialparameters(rng, layer.norm₂)
    params = (; spectral_conv, channel_mlp, skip₁, skip₂, norm₁, norm₂)
    return params
end

function Lux.initialstates(rng::AbstractRNG, layer::FourierNeuralOperatorBlock)
    return (;) # stateless
end

function Lux.parameterlength(layer::FourierNeuralOperatorBlock)
    len  = Lux.parameterlength(layer.spectral_conv)
    len += Lux.parameterlength(layer.channel_mlp)
    len += Lux.parameterlength(layer.skip₁)
    len += Lux.parameterlength(layer.skip₂)
    len += Lux.parameterlength(layer.norm₁)
    len += Lux.parameterlength(layer.norm₂)
    return len
end

function Lux.statelength(::FourierNeuralOperatorBlock)
    return 0
end

function (layer::FourierNeuralOperatorBlock)(
    x::AbstractArray, params::NamedTuple, states::NamedTuple
)
    # first skip connection
    (x_skip₁, _) = layer.skip₁(x, params.skip₁, states)
    # second skip connection
    (x_skip₂, _) = layer.skip₂(x, params.skip₂, states)
    # spectral convolution
    (x_conv, _) = layer.spectral_conv(x, params.spectral_conv, states)
    # first normalization
    (x_norm₁, _) = layer.norm₁(x_conv, params.norm₁, states)
    # first residual addition followed by first activation
    x_act = gelu.(x_norm₁ .+ x_skip₁)
    # 2-layer channel MLP
    (x_mlp, _) = layer.channel_mlp(x_act, params.channel_mlp, states)
    # second residual addition
    x_res = x_mlp .+ x_skip₂
    # second normalization
    (x_norm₂, _) = layer.norm₂(x_res, params.norm₂, states)
    # final output: second activation
    output = gelu.(x_norm₂)
    # update normalization states
    return (output, states)
end
