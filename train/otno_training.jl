# main script to train the Optimal Transport Neural Operator (OTNO) model on ShapeNet-Car dataset

using Pkg
Pkg.activate(@__DIR__)

import FourierNeuralOperators as FNO
import OptimalTransportEncoding as OTE

using FourierNeuralOperators
using FourierNeuralOperators: OptimalTransportNeuralOperator, evaluate_dataset_mse
using OptimalTransportEncoding
using OptimalTransportEncoding: OTEDataSample
using Printf
using Static
using Random
using Serialization

using Reactant
using Enzyme
using Lux
using Optimisers
using LinearAlgebra

Reactant.set_default_backend("gpu")            # "gpu" = CUDA backend (via XLA/PJRT)
const device = reactant_device(; force=true)   # error if no functional Reactant GPU device
const cpu = cpu_device()                       # move results back to host for inspection

# set random seed for reproducibility
const rng = Random.default_rng()
Random.seed!(rng, 42)

include("utils.jl")

function save_checkpoint(train_state::Training.TrainState, save_dir::String, epoch::Int)
    model = train_state.model
    params = train_state.parameters |> cpu
    states = Lux.testmode(train_state.states) |> cpu
    otno_model = (; model, params, states)
    otno_model_path = joinpath(save_dir, "otno_model_epoch_$(epoch).jls")
    # delete previously saved model parameters
    rm(save_dir; recursive=true, force=true)
    mkpath(save_dir)
    # save the current model parameters
    open(otno_model_path, "w") do io
        serialize(io, otno_model)
    end
    return otno_model_path
end

function train_model(
    rng::AbstractRNG,
    dataset_dir::String;
    otno_model_save_dir="trained_otno_model",
    # set model hyperparameters
    modes::NTuple{L,Int}=(16, 16, 16, 16), # L = 4 FNO blocks in the FNO
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
    channels_hidden = 128
    channels_out = 1

    # instantiate FNO model
    model = FNO.OptimalTransportNeuralOperator{D}(
        channels_in, channels_hidden, channels_out; modes, rank_ratio
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

    # precompile model for validation evaluation
    x₁ = first(xs_val)
    states_val = Lux.testmode(train_state.states)
    compiled_model = @compile model(x₁, train_state.parameters, states_val)
    loss_val_min = evaluate_dataset_mse(
        compiled_model, train_state.parameters, states_val, (xs_val, ys_val)
    )
    @printf "Validation loss before training:  %4.6f\n" loss_val_min

    @info "Training..."
    for epoch in 1:num_epochs
        loss_train = 0.0f0
        for (xᵢ, yᵢ) in zip(xs_train, ys_train)
            _, loss, _, train_state = Training.single_train_step!(
                AutoEnzyme(), loss_func, (xᵢ, yᵢ), train_state
            )
            loss_train += loss
        end
        loss_train /= num_samples_train
        @printf "Epoch [%3d]: Training Loss  %4.6f\n" epoch loss_train

        # evaluate the model on validation set
        states_val = Lux.testmode(train_state.states)
        loss_val = evaluate_dataset_mse(
            compiled_model, train_state.parameters, states_val, (xs_val, ys_val)
        )
        @printf "Epoch [%3d]: Validation loss  %4.6f\n" epoch loss_val
        if loss_val < loss_val_min
            loss_val_min = loss_val
            @info "Saving pretrained model weights with validation loss  $loss_val_min"
            save_checkpoint(train_state, otno_model_save_dir, epoch)
        end
    end
    @info "Training completed."
    output = (;
        model=compiled_model,
        params=train_state.parameters,
        states=Lux.testmode(train_state.states)
    )
    return output
end

############################################################################################

const num_epochs = 300
const dataset_dir = normpath(joinpath(@__DIR__, "..", "datasets/ShapeNet-Car"))
const otno_model_save_dir = normpath(joinpath(@__DIR__, "trained_otno_model"))

@time (model, params, states) = train_model(rng, dataset_dir; num_epochs, otno_model_save_dir)
