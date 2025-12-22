function load_datasets(
    dataset_dir::String,
    extension::String;
    split::NamedTuple=(; train=0.9, val=0.05, test=0.05)
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
