import FourierNeuralOperators as FNO
import OptimalTransportEncoding as OTE

using FourierNeuralOperators
using OptimalTransportEncoding
using OptimalTransportEncoding: OTEDataSample
using Printf
using Static
using Random

using CUDA
using Lux
using Reactant
using Enzyme
using Optimisers
using LinearAlgebra

CUDA.allowscalar(false)

# select Reactant CUDA backend and a Reactant device
Reactant.set_default_backend("gpu")            # "gpu" = CUDA backend (via XLA/PJRT)
const device = reactant_device(; force=true)   # error if no functional Reactant GPU device
const cpu = cpu_device()                       # move results back to host for inspection

# set random seed for reproducibility
const rng = Random.default_rng()
Random.seed!(rng, 42)

# load dataset into CPU memory
const dataset_dir = normpath(joinpath(@__DIR__, "..", "datasets/ShapeNet-Car"))
(dataset_train, dataset_val, _) = load_datasets(dataset_dir, ".jls")

# set model parameters
features₁ = first(dataset_train.features)
const D = ndims(features₁) - 2
const channels_in = size(features₁, 3)
const channels_hidden = channels_in
const channels_out = 1
const modes = (12, ntuple(_ -> 16, static(D - 1))...)
const rank_ratio = 0.5f0

# set optimiser parameters
const learning_rate = 1f-3
const weight_decay = 1f-4

# instantiate FNO model
model = FNO.FourierNeuralOperator{D}(
    channels_in, channels_hidden, channels_out; modes, rank_ratio
)
display(model)

# setup model parameters and states
(ps, st) = Lux.setup(rng, model)
st_val = Lux.testmode(st)
params = ps |> device
states = st |> device
states_val = st_val |> device

# move training data to Reactant device
xs_train = device.(dataset_train.features)
ys_train = device.(dataset_train.targets)
decoding_indices_train = device.(dataset_train.decoding_indices)

# move validation data to Reactant device
xs_val = device.(dataset_val.features)
ys_val = device.(dataset_val.targets)
decoding_indices_val = device.(dataset_val.decoding_indices)

# instantiate optimiser
optimiser = AdamW(eta=learning_rate, lambda=weight_decay)

# instantiate training state
train_state = Training.TrainState(model, params, states, optimiser)

params = ps
states = st
target = dataset_train.targets[1]
decoding_indices = dataset_train.decoding_indices[1]

MSELoss()

(ys, st_out) = model(features₁, params, states)
ys_vec = reshape(ys, :)
ys_phys = ys_vec[decoding_indices]
ys_phys - target

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

    xs = Tuple{Array{Float32,4},Vector{Int}}[get_model_inputs(ote_samples[i]) for i in train_slice]
    ys = Vector{Float32}[ote_samples[i].target for i in train_slice]
    dataset_train = (; xs, ys)

    xs = Tuple{Array{Float32,4},Vector{Int}}[get_model_inputs(ote_samples[i]) for i in val_slice]
    ys = Vector{Float32}[ote_samples[i].target for i in val_slice]
    dataset_val = (; xs, ys)

    xs = Tuple{Array{Float32,4},Vector{Int}}[get_model_inputs(ote_samples[i]) for i in test_slice]
    ys = Vector{Float32}[ote_samples[i].target for i in test_slice]
    dataset_test = (; xs, ys)

    return (dataset_train, dataset_val, dataset_test)
end

function get_model_inputs(sample::OTEDataSample)
    return (sample.features, sample.decoding_indices)
end
