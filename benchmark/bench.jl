using ChangePointDetection
using Random
using LinearAlgebra

println("Julia version: ", VERSION)
println("Threads: ", Threads.nthreads())
println("BLAS threads: ", BLAS.get_num_threads())
println()

# --- Single lsdd call ---

a = Float64.(repeat([1,2,3], 14))
b = Float64.(repeat([2,3,4], 14))

# Warmup
lsdd(a, b; rng=Xoshiro(1))

# Benchmark lsdd (Float64)
t_lsdd_f64 = let times = Float64[]
    for i in 1:5
        push!(times, @elapsed lsdd(a, b; rng=Xoshiro(i)))
    end
    minimum(times)
end
println("lsdd Float64 (42-element): $(round(t_lsdd_f64 * 1000; digits=2)) ms")

# Benchmark lsdd (Float32)
a32 = Float32.(a)
b32 = Float32.(b)
lsdd(a32, b32; rng=Xoshiro(1))  # warmup
t_lsdd_f32 = let times = Float64[]
    for i in 1:5
        push!(times, @elapsed lsdd(a32, b32; rng=Xoshiro(i)))
    end
    minimum(times)
end
println("lsdd Float32 (42-element): $(round(t_lsdd_f32 * 1000; digits=2)) ms")

# --- lsdd_profile ---

Random.seed!(1)
ts_f64 = vcat(rand(Float64, 128), rand(Float64, 128) .+ 1.5)

# Warmup
lsdd_profile(ts_f64; window=50)

t_profile_f64 = let times = Float64[]
    for _ in 1:3
        push!(times, @elapsed lsdd_profile(ts_f64; window=50))
    end
    minimum(times)
end
println("lsdd_profile Float64 (256 timesteps, window=50): $(round(t_profile_f64; digits=3)) s")

ts_f32 = Float32.(ts_f64)
lsdd_profile(ts_f32; window=50)  # warmup
t_profile_f32 = let times = Float64[]
    for _ in 1:3
        push!(times, @elapsed lsdd_profile(ts_f32; window=50))
    end
    minimum(times)
end
println("lsdd_profile Float32 (256 timesteps, window=50): $(round(t_profile_f32; digits=3)) s")

# --- Small window (window=4) ---

lsdd_profile(ts_f32; window=4)  # warmup
t_profile_small = let times = Float64[]
    for _ in 1:3
        push!(times, @elapsed lsdd_profile(ts_f32; window=4))
    end
    minimum(times)
end
println("lsdd_profile Float32 (256 timesteps, window=4): $(round(t_profile_small; digits=3)) s")

println("\nDone.")
