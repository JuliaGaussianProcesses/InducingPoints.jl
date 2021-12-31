using InducingPoints, KernelFunctions
using BenchmarkTools
using Plots
plotlyjs()

nv = [2, 4, 8, 16]
nnv = length(nv)
ker = SqExponentialKernel()

sizeug = zeros(nnv)
sizeexpgrid = zeros(nnv)

kermtimeug =  zeros(nnv)
kermtimegrid = zeros(nnv)
kermmemug =  zeros(nnv)
kermmemgrid = zeros(nnv)

sumtimeug =  zeros(nnv)
sumtimegrid = zeros(nnv)
summemug =  zeros(nnv)
summemgrid = zeros(nnv)


for (i, n) in enumerate(nv)
    iter = LinRange(0., 10., n)
    proditer = Iterators.product(iter, iter)
    ug = UniformGrid(proditer)
    expgrid = ug[:]
    println(length(expgrid))

    sizeug[i] = sizeof(ug)
    sizeexpgrid[i] = sizeof(expgrid)

    r1 = @benchmark kernelmatrix($ker, $ug)
    kermtimeug[i] = median(r1.times)
    kermmemug[i] = r1.memory
    r2 = @benchmark kernelmatrix($ker, $expgrid)
    kermtimegrid[i] = median(r2.times)
    kermmemgrid[i] = r2.memory

    rs1 = @benchmark sum.($ug)
    sumtimeug[i] = median(rs1.times)
    summemug[i] = rs1.memory
    rs2 = @benchmark sum.($expgrid)
    sumtimegrid[i] = median(rs2.times)
    summemgrid[i] = rs2.memory
end

elbs = ["" ""]
p1 = plot()
plot!(p1, nv, [sizeug, sizeexpgrid]./1e3, label = ["UniformGrid" "Explicit Grid"],
    xticks = nv, linewidth = 3, legend = :topleft,
    xlabel = "Nr Grid Points", ylabel = "Object Size [kB]")

p2 = plot()
plot!(nv, [kermtimeug, kermtimegrid]./1e3, yaxis = :log, xaxis = :log, label = elbs,
    title = "Kernelmatrix", ylabel = "Compute Time [μs]",
    left_margin = 5.0Plots.Measures.mm,
    xticks = (nv, nv), linewidth = 3,)

p3 = plot()
plot!(nv, [sumtimeug, sumtimegrid]./1e3, yaxis = :log, xaxis = :log, label = elbs,
    title = "Broadcast", xticks = (nv, nv), linewidth = 3,
    left_margin = -2.0Plots.Measures.mm,)

p4 = plot()
plot!(p4, nv, [kermmemug, kermmemgrid]./1e3, yaxis = :log, xaxis = :log, label = elbs,
    xlabel = "Nr Grid Points", ylabel = "Memory [kB]",
    left_margin = 5.0Plots.Measures.mm,
    xticks = (nv, nv), linewidth = 3,)

p5 = plot()
plot!(p5, nv, [summemug, summemgrid]./1e3, yaxis = :log, xaxis = :log, label = elbs,
    xlabel = "Nr of Grid Points",
    left_margin = -2.0Plots.Measures.mm,
    xticks = (nv, nv), linewidth = 3,)

pt = plot(p2,p3)
pm = plot(p4, p5)
pR = plot(pt, pm, layout = (2,1))

p = plot(p1, pR,
    size = (700, 400), margin = 3Plots.Measures.mm,
    layout = grid(1, 2, widths=[0.3 ,0.7]))


savefig(p, "../src/assets/UniformGrid_bench.svg")






