# main script to train the Optimal Transport Neural Operator (OTNO) model on ShapeNet-Car dataset

import FourierNeuralOperators as FNO
import OptimalTransportEncoding as OTE

using FourierNeuralOperators
using FourierNeuralOperators: OptimalTransportNeuralOperator, compute_dataset_loss
using OptimalTransportEncoding
using OptimalTransportEncoding: OTEDataSample
using Printf
using Static
using Random
using Serialization

using CUDA
using CUDA: DeviceMemory
using Zygote
using Lux
using LuxCUDA
using Optimisers
using LinearAlgebra

CUDA.allowscalar(false)  # disallow slow scalar operations on GPU

const device = gpu_device(; force=true)   # error if no functional GPU device
const cpu = cpu_device()                  # move results back to host for inspection

# set random seed for reproducibility
const rng = Random.default_rng()
Random.seed!(rng, 42)

function load_datasets(
    dataset_dir::String,
    extension::String;
    split::NamedTuple=(; train=0.93, val=0.02, test=0.05)
)
    @assert sum(split) == 1.0
    data_sample_paths = readdir(dataset_dir; join=true)
    filter!(endswith(extension), data_sample_paths)

    num_samples = length(data_sample_paths)
    train_idx_last = ceil(Int, split.train * num_samples)
    val_idx_last = floor(Int, (split.train + split.val) * num_samples)

    train_slice = 1:train_idx_last
    val_slice = (train_idx_last+1):val_idx_last
    test_slice = (val_idx_last+1):num_samples
    ote_samples = OTE.load_sample.(data_sample_paths)

    dataset_train = form_dataset(ote_samples, train_slice)
    dataset_val = form_dataset(ote_samples, val_slice)
    dataset_test = form_dataset(ote_samples, test_slice)
    return (dataset_train, dataset_val, dataset_test)
end

function form_dataset(ote_samples::Vector{OTEDataSample{N}}, slice::UnitRange{Int}) where {N}
    xs = Tuple{Array{Float32,N},Vector{Int32}}[
        get_model_inputs(ote_samples[i]) for i in slice
    ]
    ys = Vector{Float32}[ote_samples[i].target for i in slice]
    dataset = (; xs, ys)
    return dataset
end

function get_model_inputs(sample::OTEDataSample)
    return (sample.features, sample.decoding_indices)
end

function save_checkpoint(train_state::Training.TrainState, save_dir::String, epoch::Int)
    params = train_state.parameters |> cpu
    states = Lux.testmode(train_state.states) |> cpu
    otno_weights = (; params, states)
    weights_path = joinpath(save_dir, "otno_weights_epoch_$(epoch).jls")
    # delete previously saved model parameters
    rm(save_dir; recursive=true, force=true)
    mkpath(save_dir)
    # save the current model parameters
    open(weights_path, "w") do io
        serialize(io, otno_weights)
    end
    return weights_path
end

function train_model(
    rng::AbstractRNG,
    dataset_dir::String;
    weights_save_dir = "pretrained_otno_weights",
    # set model hyperparameters
    fno_modes::NTuple{L,Int}=(16, 16, 16, 16), # L = 4 FNO blocks in the FNO
    rank_ratio::Float32=0.5f0,
    # set training hyperparameters
    num_epochs::Integer,
    learning_rate::Float32=1f-3,
    weight_decay::Float32=1f-4
) where {L}
    # load dataset into CPU memory
    (dataset_train, dataset_val, _) = load_datasets(dataset_dir, ".jls")

    # set model hyperparameters
    features₁ = first(first(dataset_train.xs))
    D = ndims(features₁) - 2
    channels_in = size(features₁, D + 1)
    channels_hidden = channels_in
    channels_out = 1

    # instantiate FNO model
    model = FNO.OptimalTransportNeuralOperator{D}(
        channels_in, channels_hidden, channels_out; fno_modes, rank_ratio
    )
    display(model)

    # setup model parameters and states
    (ps, st) = Lux.setup(rng, model)
    params = ps |> device
    states = st |> device

    # move training data to the GPU device
    xs_train = device.(dataset_train.xs)
    ys_train = device.(dataset_train.ys)
    num_samples_train = length(ys_train)

    # move validation data to the GPU device
    xs_val = device.(dataset_val.xs)
    ys_val = device.(dataset_val.ys)

    # instantiate optimiser
    optimiser = AdamW(eta=learning_rate, lambda=weight_decay)
    # instantiate training state
    train_state = Training.TrainState(model, params, states, optimiser)
    loss_func = MSELoss()
    ad_backend = AutoZygote()

    # precompile model for validation evaluation
    states_val = Lux.testmode(train_state.states)
    loss_val_min = compute_dataset_loss(
        model, train_state.parameters, states_val, (xs_val, ys_val)
    )
    @printf "Validation loss before training:  %4.6f\n" loss_val_min

    @info "Training..."
    for epoch in 1:num_epochs
        loss_train = 0.0f0
        for (xᵢ, yᵢ) in zip(xs_train, ys_train)
            _, loss, _, train_state = Training.single_train_step!(
                ad_backend, loss_func, (xᵢ, yᵢ), train_state
            )
            loss_train += loss
        end
        loss_train /= num_samples_train
        @printf "Epoch [%3d]: Training Loss  %4.6f\n" epoch loss_train

        # evaluate the model on validation set
        states_val = Lux.testmode(train_state.states)
        loss_val = compute_dataset_loss(
            model, train_state.parameters, states_val, (xs_val, ys_val)
        )
        @printf "Epoch [%3d]: Validation loss  %4.6f\n" epoch loss_val
        if loss_val < loss_val_min
            loss_val_min = loss_val
            @info "Saving pretrained model weights with validation loss  $loss_val_min"
            save_checkpoint(train_state, weights_save_dir, epoch)
        end
    end
    @info "Training completed."
    return (; model, params=train_state.parameters, states=Lux.testmode(train_state.states))
end

############################################################################################

const num_epochs = 100
const dataset_dir = normpath(joinpath(@__DIR__, "..", "datasets/ShapeNet-Car"))
const weights_save_dir = normpath(joinpath(@__DIR__, "pretrained_otno_weights"))

(model, params_opt, states_val) = train_model(rng, dataset_dir; num_epochs, weights_save_dir)
