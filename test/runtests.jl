using TensorizedFourierNeuralOperators
using ReTest
using Lux
using Random
using Enzyme
using FFTW
using NNlib

# Define custom Enzyme rules to skip differentiation of FFTW plan creation
# We use Any for config to avoid version specific type names
function Enzyme.EnzymeRules.augmented_primal(config, func::Enzyme.Const{typeof(FFTW.plan_rfft)}, ::Type{<:Enzyme.Const}, args::Vararg{Any, N}) where N
    primal = func.val(map(x->x.val, args)...)
    return Enzyme.EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function Enzyme.EnzymeRules.reverse(config, func::Enzyme.Const{typeof(FFTW.plan_rfft)}, ::Type{<:Enzyme.Const}, tape, args::Vararg{Any, N}) where N
    return (nothing, map(_->nothing, args)...)
end

function Enzyme.EnzymeRules.augmented_primal(config, func::Enzyme.Const{typeof(FFTW.plan_fft)}, ::Type{<:Enzyme.Const}, args::Vararg{Any, N}) where N
    primal = func.val(map(x->x.val, args)...)
    return Enzyme.EnzymeRules.AugmentedReturn(primal, nothing, nothing)
end

function Enzyme.EnzymeRules.reverse(config, func::Enzyme.Const{typeof(FFTW.plan_fft)}, ::Type{<:Enzyme.Const}, tape, args::Vararg{Any, N}) where N
    return (nothing, map(_->nothing, args)...)
end

# Helper to check gradients
function check_gradients(layer, x, ps, st)
    function loss_fn(p, x_in)
        out, _ = layer(x_in, p, st)
        return sum(abs2, out)
    end

    dx = Enzyme.make_zero(x)
    dp = Enzyme.make_zero(ps)

    Enzyme.autodiff(Reverse, loss_fn, Active, Duplicated(ps, dp), Duplicated(x, dx))

    # Verify gradients are computed (non-zero)
    @test !all(iszero, dp.core)
    @test !all(iszero, dp.U_in)
    @test !all(iszero, dp.U_out)
end

# Set seed for reproducibility
Random.seed!(1234)

@testset "TensorizedFourierNeuralOperators.jl" begin
    @testset "1D TuckerSpectralConv" begin
        # 1D configuration
        ch_in = 4
        ch_out = 8
        modes = (16,)
        rank_in = 2
        rank_out = 4
        rank_modes = (8,)

        layer = TuckerSpectralConv(
            ch_in => ch_out, modes;
            rank_in=rank_in, rank_out=rank_out, rank_modes=rank_modes
        )

        # Test initialization
        rng = Random.default_rng()
        ps, st = Lux.setup(rng, layer)

        @test hasproperty(ps, :core)
        @test hasproperty(ps, :U_in)
        @test hasproperty(ps, :U_out)
        @test hasproperty(ps, :U_modes)

        # Test forward pass
        batch_size = 3
        x = randn(Float32, 32, ch_in, batch_size)

        y, st_out = layer(x, ps, st)

        @test size(y) == (32, ch_out, batch_size)

        # Test Gradients
        check_gradients(layer, x, ps, st)
    end

    @testset "2D TuckerSpectralConv" begin
        # 2D configuration
        ch_in = 4
        ch_out = 8
        modes = (12, 12)
        rank_in = 2
        rank_out = 4
        rank_modes = (6, 6)

        layer = TuckerSpectralConv(
            ch_in => ch_out, modes;
            rank_in=rank_in, rank_out=rank_out, rank_modes=rank_modes
        )

        rng = Random.default_rng()
        ps, st = Lux.setup(rng, layer)

        # Input: (d1, d2, ch_in, batch)
        batch_size = 2
        x = randn(Float32, 24, 24, ch_in, batch_size)

        y, _ = layer(x, ps, st)

        @test size(y) == (24, 24, ch_out, batch_size)

        # Test Gradients
        check_gradients(layer, x, ps, st)
    end

    @testset "3D TuckerSpectralConv" begin
        # 3D configuration
        ch_in = 2
        ch_out = 4
        modes = (8, 8, 8)
        rank_in = 2
        rank_out = 2
        rank_modes = (4, 4, 4)

        layer = TuckerSpectralConv(
            ch_in => ch_out, modes;
            rank_in=rank_in, rank_out=rank_out, rank_modes=rank_modes
        )

        rng = Random.default_rng()
        ps, st = Lux.setup(rng, layer)

        # Input: (d1, d2, d3, ch_in, batch)
        batch_size = 2
        x = randn(Float32, 16, 16, 16, ch_in, batch_size)

        y, _ = layer(x, ps, st)

        @test size(y) == (16, 16, 16, ch_out, batch_size)

        # Test Gradients
        check_gradients(layer, x, ps, st)
    end
end

retest()
