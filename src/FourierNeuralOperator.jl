struct FourierNeuralOperator{
    D,L,Lift,FNOBlocks,Project
} <: Lux.AbstractLuxContainerLayer{(:lift, :fno_blocks, :project)}
    lift::Lift
    fno_blocks::FNOBlocks
    project::Project
end

function FourierNeuralOperator{D}(
    channels_in::Int,
    channels_hidden::Int,
    channels_out::Int;
    modes::NTuple{L,Int}=(16, 16, 16, 16),
    rank_ratio::Float32=0.5f0
) where {D,L}
    dim = static(D)
    dimm1 = static(D - 1)
    pointwise_kernel = ntuple(_ -> 1, dim)
    channels = (channels_hidden => channels_hidden)
    # lift layer: 2-layer channel MLP, i.e. pointwise Conv with GeLU activation in between
    lift = Chain(
        Conv(pointwise_kernel, channels_in => channels_hidden, gelu),
        Conv(pointwise_kernel, channels)
    )
    # stack of L FNO blocks
    fno_blocks_tuple = ntuple(static(L)) do l
        mode = modes[l]
        mode₁ = ceil(Int, mode / 2)
        block_modes = (mode₁, ntuple(_ -> mode, dimm1)...)
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

function (layer::FourierNeuralOperator)(
    x::AbstractArray, params::NamedTuple, states::NamedTuple
)
    # lift
    (x_lift, _) = layer.lift(x, params.lift, states.lift)
    # apply FNO blocks
    (x_fno, states_fno_blocks) = layer.fno_blocks(x_lift, params.fno_blocks, states.fno_blocks)
    # project
    (output, _) = layer.project(x_fno, params.project, states.project)
    states_out = (; fno_blocks=states_fno_blocks)
    return (output, states_out)
end
