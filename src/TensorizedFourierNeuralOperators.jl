module TensorizedFourierNeuralOperators

########   Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs   ########
#
# See arxiv.org/abs/2310.00120 for details.

export FactorizedSpectralConv

using Lux
using FFTW
using Random
using NNlib: batched_mul, pad_constant
using NeuralOperators: FourierTransform

include("FactorizedSpectralConv.jl")
include("SoftGating.jl")
include("FourierNeuralOperatorBlock.jl")

end # module TensorizedFourierNeuralOperators
