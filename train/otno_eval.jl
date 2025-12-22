import FourierNeuralOperators as FNO
import OptimalTransportEncoding as OTE

using FourierNeuralOperators: OptimalTransportNeuralOperator, evaluate_dataset_mrl2e
using OptimalTransportEncoding: OTEDataSample
using Lux
using Serialization

include("utils.jl")

function load(path::AbstractString)
    object = open(path, "r") do io
        deserialize(io)
    end
    return object
end

otno_model_path = normpath(joinpath(@__DIR__, "trained_otno_model/otno_model_epoch_293.jls"))
dataset_dir = normpath(joinpath(@__DIR__, "..", "datasets/ShapeNet-Car"))

(model, params, st) = load(otno_model_path)
states = Lux.testmode(st)
display(model)

(dataset_train, dataset_val, dataset_test) = load_datasets(dataset_dir, ".jls")
(xs_test, ys_test) = dataset_test

mrl2e = evaluate_dataset_mrl2e(model, params, states, (xs_test, ys_test))
mrl2e = evaluate_dataset_mrl2e(model, params, states, (dataset_val.xs, dataset_val.ys))
mrl2e = evaluate_dataset_mrl2e(model, params, states, (dataset_train.xs, dataset_train.ys))
