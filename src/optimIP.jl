"""
    OptimIP(Z, opt)

Inducing point object containing its own optimizer
"""
struct OptimIP{S, TZ<:AbstractVector{S}, IP<:AIP{S,TZ}, O} <: AIP{S, TZ}
    Z::IP
    opt::O
end
