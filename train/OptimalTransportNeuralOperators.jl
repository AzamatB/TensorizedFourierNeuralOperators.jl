import FourierNeuralOperators as FNO
import OptimalTransportEncoding as OTE

using FourierNeuralOperators
using FourierNeuralOperators: OptimalTransportNeuralOperator
using OptimalTransportEncoding
using OptimalTransportEncoding: OTEDataSample
using Printf
using Static
using Random

using CUDA
using CUDA: DeviceMemory
using Zygote
using Lux
using LuxCUDA
using Optimisers
using LinearAlgebra

CUDA.allowscalar(false)  # disallow slow scalar operations on GPU


const device = gpu_device(; force=true)   # error if no functional GPU device
const cpu = cpu_device()                       # move results back to host for inspection

# set random seed for reproducibility
const rng = Random.default_rng()
Random.seed!(rng, 42)

function compute_dataset_loss(
    model::OptimalTransportNeuralOperator,
    params::NamedTuple,
    states::NamedTuple,
    (xs, ys)
)
    mse = 0.0f0
    for (x, y) in zip(xs, ys)
        (ŷ, _) = model(x, params, states)
        len = length(y)
        mse += sum(abs.(y .- ŷ)) / len
    end
    # a mean over all samples
    count = length(ys)
    loss = mse / count
    return loss
end

function load_datasets(
    dataset_dir::String,
    extension::String;
    split::NamedTuple=(; train=0.93, val=0.02, test=0.05)
)
    @assert sum(split) == 1.0
    data_sample_paths = readdir(dataset_dir; join=true)
    filter!(endswith(extension), data_sample_paths)

    len = length(data_sample_paths)
    train_idx_last = ceil(Int, split.train * len)
    val_idx_last = floor(Int, (split.train + split.val) * len)

    train_slice = 1:train_idx_last
    val_slice = (train_idx_last+1):val_idx_last
    test_slice = (val_idx_last+1):len
    ote_samples = OTE.load_sample.(data_sample_paths)

    xs = Tuple{Array{Float32,4},Vector{Int32}}[get_model_inputs(ote_samples[i]) for i in train_slice]
    ys = Vector{Float32}[ote_samples[i].target for i in train_slice]
    dataset_train = (; xs, ys)

    xs = Tuple{Array{Float32,4},Vector{Int32}}[get_model_inputs(ote_samples[i]) for i in val_slice]
    ys = Vector{Float32}[ote_samples[i].target for i in val_slice]
    dataset_val = (; xs, ys)

    xs = Tuple{Array{Float32,4},Vector{Int32}}[get_model_inputs(ote_samples[i]) for i in test_slice]
    ys = Vector{Float32}[ote_samples[i].target for i in test_slice]
    dataset_test = (; xs, ys)

    return (dataset_train, dataset_val, dataset_test)
end

function get_model_inputs(sample::OTEDataSample)
    return (sample.features, sample.decoding_indices)
end

# load dataset into CPU memory
const dataset_dir = normpath(joinpath(@__DIR__, "..", "datasets/ShapeNet-Car"))
(dataset_train, dataset_val, _) = load_datasets(dataset_dir, ".jls")

# set training hyperparameters
const num_epochs = 100
const learning_rate = 1f-3
const weight_decay = 1f-4

# set model hyperparameters
x₁ = first(dataset_train.xs)
(features₁, decoding_indices) = x₁
const D = ndims(features₁) - 2
const channels_in = size(features₁, 3)
const channels_hidden = channels_in
const channels_out = 1
const modes = (16, 16, 16, 16) # L = 4 FNO blocks in the FNO
const rank_ratio = 0.5f0

# instantiate FNO model
model = FNO.OptimalTransportNeuralOperator{D}(
    channels_in, channels_hidden, channels_out; modes, rank_ratio
)
display(model)

# setup model parameters and states
(ps, st) = Lux.setup(rng, model)
params = ps |> device
states = st |> device

# move training data to Reactant device
xs_train = device.(dataset_train.xs)
ys_train = device.(dataset_train.ys)

# move validation data to Reactant device
xs_val = device.(dataset_val.xs)
ys_val = device.(dataset_val.ys)

# instantiate optimiser
optimiser = AdamW(eta=learning_rate, lambda=weight_decay)

# instantiate training state
train_state = Training.TrainState(model, params, states, optimiser)
loss_func = MSELoss()

# precompile model for validation evaluation
states_val = Lux.testmode(train_state.states)
loss_val = compute_dataset_loss(
    model, train_state.parameters, states_val, (xs_val, ys_val)
)
@printf "Validation loss before training:  %4.6f\n" loss_val

@info "Training..."
for epoch in 1:num_epochs
    loss_train = 0.0f0
    for (xᵢ, yᵢ) in zip(xs_train, ys_train)
        _, loss, _, train_state = Training.single_train_step!(
            AutoZygote(), loss_func, (xᵢ, yᵢ), train_state
        )
        loss_train += loss
    end
    loss_train /= length(ys_train)
    @printf "Epoch [%3d]: Training Loss  %4.6f\n" epoch loss_train
    # evaluate the model on validation set
    states_val = Lux.testmode(train_state.states)
    loss_val = compute_dataset_loss(
        compiled_model, train_state.parameters, states_val, (xs_val, ys_val)
    )
    @printf "Epoch [%3d]: Validation loss  %4.6f\n" epoch loss_val
end

@info "Training completed."
params_opt = train_state.parameters
states_val = Lux.testmode(train_state.states)
# return (compiled_model, params_opt, states_val)
# end
