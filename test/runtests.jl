using Test
using ChangePointDetection
using Random
using LinearAlgebra

@testset "ChangePointDetection" begin

    @testset "squared_distance" begin
        x = [1.0, 2.0, 3.0]
        c = [1.0, 4.0]
        d = squared_distance(x, c)
        @test size(d) == (3, 2)
        @test d[1, 1] == 0.0   # (1-1)^2
        @test d[1, 2] == 9.0   # (1-4)^2
        @test d[2, 1] == 1.0   # (2-1)^2
        @test d[3, 2] == 1.0   # (3-4)^2

        # Float32 support
        x32 = Float32[1.0, 2.0, 3.0]
        c32 = Float32[1.0, 4.0]
        d32 = squared_distance(x32, c32)
        @test eltype(d32) == Float32
        @test d32 ≈ Float32.(d)
    end

    @testset "lsdd basic" begin
        a = Float64.(repeat([1, 2, 3], 14))  # 42 elements
        b = Float64.(repeat([2, 3, 4], 14))  # 42 elements

        # With seeded RNG, lsdd should be deterministic and positive
        rng = Xoshiro(42)
        result = lsdd(a, b; rng=rng)
        @test result isa Float64
        @test result > 0.0  # different distributions should have positive LSDD

        # Determinism: same seed should give same result
        rng2 = Xoshiro(42)
        result2 = lsdd(a, b; rng=rng2)
        @test result == result2

        # Identical distributions should have LSDD close to 0
        rng3 = Xoshiro(42)
        result_same = lsdd(a, a; rng=rng3)
        @test abs(result_same) < 0.1
    end

    @testset "lsdd Float32" begin
        a32 = Float32.(repeat([1, 2, 3], 14))
        b32 = Float32.(repeat([2, 3, 4], 14))

        rng = Xoshiro(42)
        result = lsdd(a32, b32; rng=rng)
        @test result isa Float32
        @test result > 0.0f0
    end

    @testset "lsdd_profile" begin
        # Create a time series with a known change point at index 50
        Random.seed!(123)
        ts = vcat(rand(Float64, 50), rand(Float64, 50) .+ 3.0)

        profile = lsdd_profile(ts; window=15)
        @test length(profile) == length(ts) - 2 * 15
        @test eltype(profile) == Float64

        # The profile should have a peak near the change point
        # Change point is at index 50, so the peak in the profile
        # should be near index 50 - 15 = 35 (offset by window)
        peak_idx = argmax(profile)
        @test 25 <= peak_idx <= 45  # should be near the change point

        # Empty result for too-large window
        @test isempty(lsdd_profile(ts; window=51))
    end

    @testset "lsdd_profile Float32" begin
        ts32 = Float32.(vcat(ones(30), 5.0 .* ones(30)))
        profile = lsdd_profile(ts32; window=10)
        @test eltype(profile) == Float32
        @test length(profile) == 60 - 20
    end

    @testset "changepoints" begin
        # Time series with a clear shift
        ts = vcat(zeros(40), 5.0 .* ones(40))
        points = changepoints(ts; threshold=0.3, window=10)
        @test length(points) >= 1
        # The changepoint should be detected near index 30 (40 - window)
        @test any(25 .<= points .<= 35)
    end

    @testset "getpoints" begin
        profile = [0.1, 0.2, 0.8, 0.9, 0.95, 0.3, 0.1, 0.85, 0.9, 0.2]
        points = getpoints(profile; threshold=0.5)
        @test points == [3, 8]  # two exceedances with hysteresis
    end

    @testset "thread safety" begin
        # lsdd_profile uses threading; verify it produces consistent-length output
        ts = Float32.(vcat(rand(64), rand(64) .+ 2.0))
        profile = lsdd_profile(ts; window=20)
        @test length(profile) == 128 - 40
        @test all(isfinite, profile)
    end

end
