module FourierNeuralOperators

########   Multi-Grid Tensorized Fourier Neural Operator for High-Resolution PDEs   ########
#
# See arxiv.org/abs/2310.00120 for details.

export FactorizedSpectralConv, FourierNeuralOperator, FourierNeuralOperatorBlock

using Reactant: TracedRNumber
using AbstractFFTs: fft, rfft, ifft, irfft, fftshift, ifftshift
using Lux
using Random
using NNlib: batched_mul, pad_constant
using Static

const RNumber{N} = Union{TracedRNumber{N},N} where {N<:Number}

include("FactorizedSpectralConv.jl")

include("SoftGating.jl")

include("FourierNeuralOperatorBlock.jl")

include("FourierNeuralOperator.jl")

include("OptimalTransportNeuralOperator.jl")

end # module FourierNeuralOperators
