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
    (x, decoding_indices)::Tuple{DenseArray{Float32},DenseVector{Int32}},
    params::NamedTuple,
    states::NamedTuple
)
    (y, states_out) = model.fno(x, params.fno, states.fno)
    y_vec = reshape(y, :)
    # pullback to physical space using decoding indices
    y_phys = y_vec[decoding_indices]
    return (y_phys, states_out)
end

function compute_dataset_loss(
    model::OptimalTransportNeuralOperator,
    params::NamedTuple,
    states::NamedTuple,
    (xs, ys)::Tuple{Vector{<:Tuple{DenseArray{Float32},DenseVector{Int32}}},Vector{<:DenseVector{Float32}}}
)
    loss = 0.0f0
    for (x, y) in zip(xs, ys)
        (ŷ, _) = model(x, params, states)
        num_points = length(y)
        Δ = @. abs2(y - ŷ)
        loss += sum(Δ) / num_points
    end
    # a mean over all samples
    num_samples = length(ys)
    mse = loss / num_samples
    return mse
end
