struct OptimalTransportNeuralOperator{D,FNO} <: Lux.AbstractLuxContainerLayer{(:fno,)}
    fno::FNO
end

function OptimalTransportNeuralOperator{D}(
    channels_in::Int,
    channels_hidden::Int,
    channels_out::Int;
    modes::NTuple{L,Int}=(16, 16, 16, 16),
    rank_ratio::Float32=0.5f0
) where {D,L}
    fno = FourierNeuralOperator{D}(channels_in, channels_hidden, channels_out; modes, rank_ratio)
    FNO = typeof(fno)
    return OptimalTransportNeuralOperator{D,FNO}(fno)
end

function (model::OptimalTransportNeuralOperator)(
    (x, decoding_indices)::Tuple{DenseArray{R},DenseVector{I}},
    params::NamedTuple,
    states::NamedTuple
) where {R<:RNumber{Float32},I<:RNumber{Int32}}
    (y, states_out) = model.fno(x, params.fno, states.fno)
    y_vec = reshape(y, :)
    # pullback to physical space using decoding indices
    y_phys = y_vec[decoding_indices]
    return (y_phys, states_out)
end
