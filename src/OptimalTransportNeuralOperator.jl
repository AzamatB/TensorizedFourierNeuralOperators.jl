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
    (y, state_fno) = model.fno(x, params.fno, states.fno)
    y_vec = reshape(y, :)
    # pullback to physical space using decoding indices
    y_phys = y_vec[decoding_indices]
    states_out = (; fno=state_fno)
    return (y_phys, states_out)
end

# mean squared error (MSE) over a dataset
function evaluate_dataset_mse(
    model::OptimalTransportNeuralOperator,
    params::NamedTuple,
    states::NamedTuple,
    (xs, ys)::Tuple{Vector{<:Tuple{DenseArray{R},DenseVector{I}}},Vector{<:DenseVector{R}}}
) where {R<:RNumber{Float32},I<:RNumber{Int32}}
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

# mean relative L² (%) error over a dataset
function evaluate_dataset_mrl2e(
    model::OptimalTransportNeuralOperator,
    params::NamedTuple,
    states::NamedTuple,
    (xs, ys)::Tuple{Vector{<:Tuple{DenseArray{R},DenseVector{I}}},Vector{<:DenseVector{R}}}
) where {R<:RNumber{Float32},I<:RNumber{Int32}}
    loss = 0.0f0
    for (x, y) in zip(xs, ys)
        (ŷ, _) = model(x, params, states)
        numΔ = @. abs2(y - ŷ)
        den = @. abs2(y)
        loss += √(sum(numΔ)) / √(sum(den))
    end
    # a mean over all samples
    num_samples = length(ys)
    mrl2e = 100 * loss / num_samples
    return mrl2e
end

# mean absolute percentage (%) error (MAPE) over a dataset
function evaluate_dataset_mape(
    model::OptimalTransportNeuralOperator,
    params::NamedTuple,
    states::NamedTuple,
    (xs, ys)::Tuple{Vector{<:Tuple{DenseArray{R},DenseVector{I}}},Vector{<:DenseVector{R}}}
) where {R<:RNumber{Float32},I<:RNumber{I}}
    loss = 0.0f0
    for (x, y) in zip(xs, ys)
        (ŷ, _) = model(x, params, states)
        num_points = length(y)
        Δ = @. abs((y - ŷ) / y)
        loss += sum(Δ) / num_points
    end
    # a mean over all samples
    num_samples = length(ys)
    mape = 100 * loss / num_samples
    return mape
end
