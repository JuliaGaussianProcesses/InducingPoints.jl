struct Optim_IP{T, M<:AbstractMatrix{T}, IP<:AIP, O} <: AIP{T, M}
    Z::IP
    opt::O
end


init!(ip::Optim_IP, args...) = init!(ip.Z, args...)
add_point!(ip::Optim_IP, args...) = add_point!(ip.Z, args...)
remove_point!(ip::Optim_IP, args...) = remove_point!(ip.Z, args...)
