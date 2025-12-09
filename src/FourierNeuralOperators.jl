module FourierNeuralOperators

########   Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs   ########
#
# See arxiv.org/abs/2310.00120 for details.

export FactorizedSpectralConv, FourierNeuralOperator, FourierNeuralOperatorBlock

using Lux
using FFTW
using Random
using NNlib: batched_mul, pad_constant

include("FactorizedSpectralConv.jl")
include("SoftGating.jl")
include("FourierNeuralOperatorBlock.jl")
include("FourierNeuralOperator.jl")

end # module FourierNeuralOperators
