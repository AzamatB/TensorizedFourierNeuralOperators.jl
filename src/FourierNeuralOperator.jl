struct FourierNeuralOperator{
    D,L,Lift,FNOBlocks,Project
} <: Lux.AbstractLuxContainerLayer{(:lift, :fno_blocks, :project)}
    lift::Lift
    fno_blocks::FNOBlocks
    project::Project
end

function FourierNeuralOperator{D}(
    channels_in::Int, channels_hidden::Int, channels_out::Int;
    modes::NTuple{L,Int}=(8, 16, 16, 16), rank_ratio::Float32=0.5f0
) where {D,L}
    channels = (channels_hidden => channels_hidden)
    pointwise_kernel = ntuple(_ -> 1, Val(D))
    # lift layer: 2-layer channel MLP, i.e. pointwise Conv with GeLU activation in between
    lift = Chain(
        Conv(pointwise_kernel, channels_in => channels_hidden, gelu),
        Conv(pointwise_kernel, channels)
    )
    # stack of L FNO blocks
    fno_blocks_tuple = ntuple(Val(L)) do l
        m = modes[l]
        block_modes = ntuple(_ -> m, Val(D))
        FourierNeuralOperatorBlock(channels, block_modes; rank_ratio)
    end
    fno_blocks = Chain(fno_blocks_tuple...)
    # projection layer: pointwise Conv to output_channels
    project = Chain(
        Conv(pointwise_kernel, channels, gelu),
        Conv(pointwise_kernel, channels_hidden => channels_out)
    )
    Lift = typeof(lift)
    FNOBlocks = typeof(fno_blocks)
    Project = typeof(project)
    return FourierNeuralOperator{D,L,Lift,FNOBlocks,Project}(lift, fno_blocks, project)
end

function Lux.initialparameters(rng::AbstractRNG, layer::FourierNeuralOperator)
    lift = Lux.initialparameters(rng, layer.lift)
    fno_blocks = Lux.initialparameters(rng, layer.fno_blocks)
    project = Lux.initialparameters(rng, layer.project)
    params = (; lift, fno_blocks, project)
    return params
end

function Lux.parameterlength(layer::FourierNeuralOperator)
    len  = Lux.parameterlength(layer.lift)
    len += Lux.parameterlength(layer.fno_blocks)
    len += Lux.parameterlength(layer.project)
    return len
end

function (layer::FourierNeuralOperator)(
    x::AbstractArray, params::NamedTuple, states::NamedTuple
)
    state = (;)
    # lift
    (x_lift, _) = layer.lift(x, params.lift, state)
    # apply FNO blocks
    (x_fno, states_fno_blocks) = layer.fno_blocks(x_lift, params.fno_blocks, states.fno_blocks)
    # project
    (output, _) = layer.project(x_fno, params.project, state)
    states_out = (; fno_blocks=states_fno_blocks)
    return (output, states_out)
end
