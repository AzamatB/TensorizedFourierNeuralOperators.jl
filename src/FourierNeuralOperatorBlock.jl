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
    pointwise_kernel = ntuple(_ -> 1, static(D))
    # channelwise linear skip layer as a pointwise convolutional layer
    skip₁ = Conv(pointwise_kernel, channels)
    skip₂ = SoftGating{D}(channels_in)
    spectral_conv = FactorizedSpectralConv(channels, modes; rank_ratio)
    # 2-layer channel MLP with GeLU activation in between
    channels = (channels_out => channels_out)
    channel_mlp = Chain(
        Conv(pointwise_kernel, channels, gelu),
        Conv(pointwise_kernel, channels)
    )
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

function (layer::FourierNeuralOperatorBlock)(
    x::DenseArray{<:Number}, params::NamedTuple, states::NamedTuple
)
    # first skip connection
    (x_skip₁, state_skip₁) = layer.skip₁(x, params.skip₁, states.skip₁)
    # second skip connection
    (x_skip₂, state_skip₂) = layer.skip₂(x, params.skip₂, states.skip₂)
    # spectral convolution
    (x_conv, state_conv) = layer.spectral_conv(x, params.spectral_conv, states.spectral_conv)
    # first normalization
    (x_norm₁, state_norm₁) = layer.norm₁(x_conv, params.norm₁, states.norm₁)
    # first residual addition followed by first activation
    x_act = gelu.(x_norm₁ .+ x_skip₁)
    # 2-layer channel MLP
    (x_mlp, state_mlp) = layer.channel_mlp(x_act, params.channel_mlp, states.channel_mlp)
    # second residual addition
    # Note: this will fail with dimensionality mismatch unless channels_in == channels_out.
    # This is the case for FNO blocks inside FNO, but for generality, consider replacing the
    # SoftGating with a pointwise Conv(channels_in => channels_out) to allow channel
    # dimension changes.
    x_res = x_mlp .+ x_skip₂
    # second normalization
    (x_norm₂, state_norm₂) = layer.norm₂(x_res, params.norm₂, states.norm₂)
    # final output: second activation
    output = gelu.(x_norm₂)
    states_out = (;
        spectral_conv=state_conv,
        channel_mlp=state_mlp,
        skip₁=state_skip₁,
        skip₂=state_skip₂,
        norm₁=state_norm₁,
        norm₂=state_norm₂
    )
    return (output, states_out)
end
