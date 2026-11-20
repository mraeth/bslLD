# Device results of the implicit-E×B pieces equal the CPU results to round-off, on the backend named by
# BSLLD_BACKEND (cpu, amdgpu, cuda; default cpu). The CPU reference and the device run share one process, so
# the workspace caches (keyed by shape and eltype, not by array type) are emptied between them.
using Test
using LinearAlgebra: norm

const _backend_name = lowercase(get(ENV, "BSLLD_BACKEND", "cpu"))
_backend_name in ("cpu", "amdgpu", "cuda") || error("BSLLD_BACKEND must be cpu, amdgpu or cuda, got $_backend_name")

# The GPU package is loaded at top level, so that the functions below see it (world age).
const _gpu_pkg = Dict("amdgpu" => :AMDGPU, "cuda" => :CUDA)
if haskey(_gpu_pkg, _backend_name)
    try
        Core.eval(Main, :(using $(_gpu_pkg[_backend_name])))
    catch err
        @info "backend $_backend_name: $(_gpu_pkg[_backend_name]) could not be loaded" err
    end
end

function _clear_workspace_caches!()
    empty!(parentmodule(bslLD.ScalarField)._spectral_ws_cache)   # PlasmaCore
    empty!(bslLD._solver_workspace_cache)
    empty!(bslLD._plan_cache)
    empty!(bslLD._exb_plan_cache)
    return nothing
end

# Returns true if the device backend is active.
function _use_device!(name)
    name == "cpu" && (bslLD.use_cpu!(); return true)
    isdefined(Main, _gpu_pkg[name]) || return false
    if name == "amdgpu"
        bslLD.amdgpu_available() || return false
        bslLD.use_amdgpu!()
        Main.AMDGPU.allowscalar(false)
    else
        bslLD.cuda_available() || return false
        bslLD.use_cuda!()
        Main.CUDA.allowscalar(false)
    end
    return true
end

_host(a) = Array(a)
_reldiff(a, b) = norm(a .- b) / max(norm(b), floatmin())

# Spatial (x, y, z) grid with one v∥ axis; φ and n0 with a profile in x and Fourier structure in y, z.
function _backend_dk_setup(; L = 100.0, N = (8, 8, 4), a = 0.05)
    grid = bslLD.Grid([0.0, 0.0, 0.0, -1.0], [L, L, L, 1.0], [N..., 2], 3, 1.0, 3)
    x, y, z = collect.(grid.xaxes)
    φ = [a * sin(2π * x[i] / L + 0.3) * cos(2π * (y[j] / L + z[l] / L)) for i in 1:N[1], j in 1:N[2], l in 1:N[3]]
    n0 = [exp(0.05L / 2π * sin(2π * x[i] / L)) * (1 + 0.1 * cos(2π * y[j] / L + 0.2)) * (1 + 0.1 * sin(2π * z[l] / L))
          for i in 1:N[1], j in 1:N[2], l in 1:N[3]]
    return grid, φ, n0
end

function _backend_cases()
    μ = 1 / 3670
    out = Dict{String,Array{Float64}}()
    F(sp) = getfield(sp, :model).dist.data
    # Pushes are compared by their increment f − f0, which is small against f.
    Δ(f, f0) = _host(f) .- f0

    # E×B RK4 push with and without 2/3 dealiasing; frozen φ.
    grid, φa, n0 = _backend_dk_setup()
    for dealias in (false, true)
        el = bslLD.Species(bslLD.FullDK(grid, n0; T_e = 1.0, Nv = 16, exb = :rk4, dealias); q = -1.0, m = μ)
        f0 = _host(F(el))
        bslLD.push_drift!(el, bslLD.ScalarField(φa), grid, 0.5)
        out["exb_rk4 dealias=$dealias Δf"] = Δ(F(el), f0)
    end

    # FullDK parallel streaming (k_z ≠ 0) and the symmetric C step (kick, E×B, kick).
    el = bslLD.Species(bslLD.FullDK(grid, n0; T_e = 1.0, Nv = 16, exb = :rk4, dealias = true); q = -1.0, m = μ)
    f0 = _host(F(el))
    bslLD.push_free!(el, grid, 0.3, 1)
    out["FullDK push_free Δf"] = Δ(F(el), f0)
    Ef = bslLD.es_field(φa, grid)
    E = bslLD.VectorField([bslLD.ScalarField(Ef[d].data) for d in 1:3])
    f0 = _host(F(el))
    bslLD._push_stage!(el, E, bslLD.ScalarField(φa), grid, 0.3; phase = 0)
    out["FullDK _push_stage Δf"] = Δ(F(el), f0)
    ρ, J = bslLD.moments_at(el, grid, 0.0)
    out["FullDK moments ρ"] = _host(ρ)
    out["FullDK moments J∥"] = _host(J[3])

    # Kinetic ions (2x3v) with FullDK electrons: quasineutral φ and one charge-form step with variable
    # coefficients (GMRES) and the dealiased RK4 E×B push.
    Nx, Ny, Nv, Lx, vm = 16, 8, 8, 100.0, 5.0
    grid = bslLD.Grid([0.0, 0.0, -vm, -vm, -vm], [Lx, 100.0, vm, vm, vm], [Nx, Ny, Nv, Nv, Nv], 2, 1.0, 3)
    x, y, v = collect(grid.xaxes[1]), collect(grid.xaxes[2]), collect(grid.vaxes[1])
    M = exp.(-v .^ 2 ./ 2)
    M ./= sum(M) * grid.delta[3]
    Nprof(X) = exp(0.05Lx / 2π * sin(2π * X / Lx))
    f = [Nprof(x[i] + v[b]) * M[a_] * M[b] * M[c] for i in 1:Nx, j in 1:Ny, a_ in eachindex(v), b in eachindex(v), c in eachindex(v)]
    sf = bslLD.ScalarField(f)
    ions = bslLD.Species(1.0, 1.0, bslLD.DistributionGridImpl{2,3,bslLD.Cart,typeof(sf)}(sf))
    ni = _host(bslLD.compute_density(ions.dist, grid).data)
    ne = ni .* [1 + 1e-3 * cos(2π * y[j] / 100 + 0.3) * sin(4π * x[i] / Lx) for i in 1:Nx, j in 1:Ny]
    el = bslLD.Species(bslLD.FullDK(grid, ne; T_e = 1.0, Nv = 4, exb = :rk4, dealias = true); q = -1.0, m = μ)
    z3() = bslLD.VectorField([bslLD.ScalarField(zeros(Nx, Ny)) for _ in 1:3])
    sol = bslLD.FieldSolution(z3(), z3())
    solver = bslLD.ESChargeForm(; variable_coefficients = true)
    bslLD.es_quasineutral_phi!(sol, (ions, el), grid, solver)
    out["es_quasineutral_phi φ"] = _host(sol.phi.data)
    fi0, fe0 = _host(ions.dist.data), _host(F(el))
    φm = bslLD.step_es_charge_form!((ions, el), sol, grid, solver, 0.05; damping_rate = 1.0, corrections = 2)
    out["step_es_charge_form φm"] = _host(φm)
    out["step_es_charge_form φ"] = _host(sol.phi.data)
    out["step_es_charge_form Δf_i"] = Δ(ions.dist.data, fi0)
    out["step_es_charge_form Δf_e"] = Δ(F(el), fe0)
    return out
end

@testset "backend: $_backend_name results equal the CPU" begin
    _clear_workspace_caches!()
    bslLD.use_cpu!()
    ref = _backend_cases()
    _clear_workspace_caches!()
    if _use_device!(_backend_name)
        dev = try
            _backend_cases()
        finally
            _clear_workspace_caches!()
            bslLD.use_cpu!()
        end
        for k in sort(collect(keys(ref)))
            d = _reldiff(dev[k], ref[k])
            @info "backend $_backend_name: $k" reldiff = d norm = norm(ref[k])
            @test norm(ref[k]) > 0
            @test d < (startswith(k, "step") || startswith(k, "es_") ? 1e-9 : 1e-12)
        end
    else
        @info "backend $_backend_name not available: device comparison skipped"
    end
end
