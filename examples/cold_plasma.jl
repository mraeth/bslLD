
# Ion units:
# w = ω/Ωᵢ, K = kρᵢ, mu = mₑ/mᵢ, betai = vthᵢ²/vA²
# Darwin models return K² = (kρᵢ)²

function rootsSDP(S, D, P, θ)
    s2, c2 = sin(θ)^2, cos(θ)^2
    A = S*s2 + P*c2
    B = (S^2 - D^2)*s2 + P*S*(1+c2)
    C = P*(S^2 - D^2)
    q = sqrt(complex(B^2 - 4A*C))
    ((B+q)/(2A), (B-q)/(2A))
end

# ion terms
sI(w) = -1/(w^2 - 1)
dI(w) = 1/(w*(w^2 - 1))
pI(w) = -1/w^2

# electron parallel term
pE(w, mu) = -1/(mu*w^2)

# generic Darwin model
function K2Darwin(w, θ, mu, betai, sE, dE)
    N2 = rootsSDP(sI(w) + sE(w, mu), dI(w) + dE(w, mu), pI(w) + pE(w, mu), θ)
    (betai/2) * w^2 .* N2
end

# three Darwin closures
K2Full(w, θ, mu, betai) =
    K2Darwin(w, θ, mu, betai, (w, mu) -> mu/(1 - mu^2*w^2), (w, mu) -> 1/(w*(1 - mu^2*w^2)))

K2Drift(w, θ, mu, betai) = K2Darwin(w, θ, mu, betai, (w, mu) -> 0, (w, mu) -> 1/w)

K2DriftPol(w, θ, mu, betai) = K2Darwin(w, θ, mu, betai, (w, mu) -> mu, (w, mu) -> 1/w)

# plot K² versus ω
function plotK2(F; mu = 0.1, betai = 1.0)
    ws = range(0.01, 10, length = 2000)

    fig = Figure(size = (900, 350))
    Label(fig[0, 1:2], "μ=$mu, βᵢ=$betai", fontsize = 18)

    for (col, θ, ttl) in [(1, 0.0, "θ = 0"), (2, π/2, "θ = π/2")]
        ax = Axis(
            fig[1, col],
            xlabel = "ω/Ωᵢ",
            ylabel = "K² = (kρᵢ)²",
            title = ttl,
            limits = ((0.01, 10), (-3, 3)),
        )

        y1 = [real(F(w, θ, mu, betai)[1]) for w in ws]
        y2 = [real(F(w, θ, mu, betai)[2]) for w in ws]

        lines!(ax, ws, y1, label = "root 1")
        lines!(ax, ws, y2, label = "root 2")
        axislegend(ax, position = :rt)
    end

    fig
end


# plot ω versus K
function plotOmega(F; mu = 0.1, betai = 1.0)
    ws = range(1e-4, 2/mu, length = 4000)

    k(w, θ, j) = begin
        z = F(w, θ, mu, betai)[j]
        abs(imag(z)) < 1e-8 && real(z) > 0 ? sqrt(real(z)) : NaN
    end

    fig = Figure(size = (900, 350))
    Label(fig[0, 1:2], "μ=$mu, βᵢ=$betai", fontsize = 18)

    for (col, θ, ttl) in [(1, π/2, "θ = π/2"), (2, 0.0, "θ = 0")]
        ax = Axis(
            fig[1, col],
            xlabel = "K = kρᵢ",
            ylabel = "ω/Ωᵢ",
            title = ttl,
            limits = ((0, 5), (0, 20)),
            aspect = 1,
        )

        K1 = [k(w, θ, 1) for w in ws]
        K2 = [k(w, θ, 2) for w in ws]

        lines!(ax, K1, ws, label = "root 1")
        lines!(ax, K2, ws, label = "root 2")
        axislegend(ax, position = :rt)
    end

    fig
end


using CairoMakie

for model in [K2Full, K2Drift, K2DriftPol]
    println(model)
    plotK2(model; mu = 0.1, betai = 0.1)
    plotOmega(model; mu = 0.1, betai = 0.1)
end
