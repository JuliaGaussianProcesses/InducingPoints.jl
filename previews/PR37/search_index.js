var documenterSearchIndex = {"docs":
[{"location":"api/#API","page":"API","title":"API","text":"","category":"section"},{"location":"api/","page":"API","title":"API","text":"Modules = [InducingPoints]","category":"page"},{"location":"api/#InducingPoints.CoverTree","page":"API","title":"InducingPoints.CoverTree","text":"CoverTree(ϵ::Real, lloyds::Bool=true, voronoi::Bool=true, metric::SemiMetric=Euclidean())\n\nThe CoverTree algorithm [1], recursively builds a tree for which the node will optimally cover the given dataset according to the metric distance.\n\nArguments:\n\nϵ::Real: Spatial resolution. Higher ϵ will result in less points.\nlloyds::Bool: Use the centroid of the ball created around the sampled datapoint instead of the datapoint itself if no other inducing point is close\nvoronoi::Bool: Reattributes samples to each node at the end of the proecssing of each layer\nmetric::SemiMetric: Distance metric used to determine distance between points. \n\n[1] Alexander Terenin, David R. Burt, Artem Artemev, Seth Flaxman, Mark van der Wilk, Carl Edward Rasmussen, Hong Ge: Numerically Stable Sparse Gaussian Processes via Minimum Separation using Cover Trees: https://arxiv.org/abs/2210.07893\n\n\n\n\n\n","category":"type"},{"location":"api/#InducingPoints.Greedy","page":"API","title":"InducingPoints.Greedy","text":"Greedy(m::Int, s::Int)\n\nm is the desired number of inducing points\ns is the minibatch size on which to select a new inducing point\n\nGreedy approach first proposed by Titsias[1]. Algorithm loops over minibatches of data and select the best ELBO improvement. Requires passing outputs y, the kernel and the noise as keyword arguments to inducingpoints.\n\n[1] Titsias, M. Variational Learning of Inducing Variables in Sparse Gaussian Processes. Aistats 5, 567–574 (2009).\n\n\n\n\n\n","category":"type"},{"location":"api/#InducingPoints.KmeansAlg","page":"API","title":"InducingPoints.KmeansAlg","text":"KmeansAlg(m::Int, metric::SemiMetric=SqEuclidean(); nMarkov = 10, tol = 1e-3)\n\nk-Means [1] initialization on the data X taking m inducing points. The seeding is computed via [2], nMarkov gives the number of MCMC steps for the seeding.\n\nArguments\n\nk::Int : Number of inducing points\nmetric::SemiMetric : Metric used to compute the distances for the k-means algorithm\n\nKeyword Arguments\n\nnMarkov::Int : Number of random steps for the seeding\ntol::Real : Tolerance for the kmeans algorithm\n\n[1] Arthur, D. & Vassilvitskii, S. k-means++: The advantages of careful seeding. in Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms 1027–1035 (Society for Industrial and Applied Mathematics, 2007). [2] Bachem, O., Lucic, M., Hassani, S. H. & Krause, A. Fast and Provably Good Seedings for k-Means. Advances in Neural Information Processing Systems 29 55–63 (2016) doi:10.1109/tmtt.2005.863818.\n\n\n\n\n\n","category":"type"},{"location":"api/#InducingPoints.OnlineIPSelection","page":"API","title":"InducingPoints.OnlineIPSelection","text":"OIPS(ρ_accept=0.8; η=0.95, kmax=Inf, kmin=10, ρ_remove=Inf)\nOIPS(kmax, η=0.98, kmin=10)\n\nOnline Inducing Points Selection. Method from [1]. Requires passing the kernel as keyword argument to inducingpoints.\n\n[1] Galy-Fajou, T. & Opper, M Adaptive Inducing Points Selection for Gaussian Processes. arXiv:2107.10066v1 (2021).\n\n\n\n\n\n","category":"type"},{"location":"api/#InducingPoints.RandomSubset","page":"API","title":"InducingPoints.RandomSubset","text":"RandomSubset(m::Int)\n\nArguments\n\nm::Int: Number of inducing points\n\nUniform sampling of a subset of m points of the data.\n\n\n\n\n\n","category":"type"},{"location":"api/#InducingPoints.SeqDPP","page":"API","title":"InducingPoints.SeqDPP","text":"SeqDPP()\n\nSequential sampling via Determinantal Point Processes. Requires passing the kernel as keyword argument to inducingpoints.\n\n\n\n\n\n","category":"type"},{"location":"api/#InducingPoints.StdDPP","page":"API","title":"InducingPoints.StdDPP","text":"StdDPP(kernel::Kernel)\n\nStandard DPP (Determinantal Point Process) sampling given kernel. The size of the returned Z is not fixed (but is not allowed to be empty unlike in a classical DPP).\n\n\n\n\n\n","category":"type"},{"location":"api/#InducingPoints.StreamKmeans","page":"API","title":"InducingPoints.StreamKmeans","text":"StreamKmeans(m_target::Int)\n\nOnline clustering algorithm [1] to select inducing points in a streaming setting. Reference : [1] Liberty, E., Sriharsha, R. & Sviridenko, M. An Algorithm for Online K-Means Clustering. arXiv:1412.5721 (2015).\n\n\n\n\n\n","category":"type"},{"location":"api/#InducingPoints.UniGrid","page":"API","title":"InducingPoints.UniGrid","text":"UniGrid(m::Int)\n\nwhere m is the number of points on each dimension. Adaptive uniform grid based on [1]. The resulting inducing points are stored in the memory-efficient custom type  UniformGrid. \n\n[1] Moreno-Muñoz, P., Artés-Rodríguez, A. & Álvarez, M. A. Continual Multi-task Gaussian Processes. (2019).\n\n\n\n\n\n","category":"type"},{"location":"api/#InducingPoints.UniformGrid","page":"API","title":"InducingPoints.UniformGrid","text":"UniformGrid{T,Titer} <: AbstractVector{T}\n\nA memory-efficient custom object representing a wrapper around a Iterators.ProductIterator. Supports broadcasting and other relevant array methods, and avoids explicitly computing all points on the grid.  \n\n\n\n\n\n","category":"type"},{"location":"api/#InducingPoints.Webscale","page":"API","title":"InducingPoints.Webscale","text":"Webscale(m::Int)\n\nOnline k-means algorithm based on [1].\n\n[1] Sculley, D. Web-scale k-means clustering. in Proceedings of the 19th international conference on World wide web - WWW ’10 1177 (ACM Press, 2010). doi:10.1145/1772690.1772862.\n\n\n\n\n\n","category":"type"},{"location":"api/#InducingPoints.kDPP","page":"API","title":"InducingPoints.kDPP","text":"kDPP(m::Int, kernel::Kernel)\n\nk-DPP (Determinantal Point Process) will return a subset of X of size m, according to DPP probability\n\n\n\n\n\n","category":"type"},{"location":"api/#InducingPoints.find_nearest_center","page":"API","title":"InducingPoints.find_nearest_center","text":"Find the closest center to X among Z, return the index and the distance\n\n\n\n\n\n","category":"function"},{"location":"api/#InducingPoints.inducingpoints","page":"API","title":"InducingPoints.inducingpoints","text":" inducingpoints([rng::AbstractRNG], alg::AIPSA, X::AbstractVector; [kwargs...])\n inducingpoints([rng::AbstractRNG], alg::AIPSA, X::AbstractMatrix; obsdim=1, [kwargs...])\n\nSelect inducing points according to the algorithm alg. For some algorithms, additional keyword arguments are required. \n\n\n\n\n\n","category":"function"},{"location":"api/#InducingPoints.inducingpoints-Tuple{Random.AbstractRNG, Greedy, AbstractVector}","page":"API","title":"InducingPoints.inducingpoints","text":" inducingpoints([rng::AbstractRNG], alg::Greedy, X::AbstractVector; \n    y::AbstractVector, kernel::Kernel, noise::Real)\n inducingpoints([rng::AbstractRNG], alg::Greedy, X::AbstractMatrix; \n    obsdim=1, y::AbstractVector, kernel::Kernel, noise::Real)\n\nSelect inducing points according using the Greedy algorithm. Requires as additional keyword arguments the outputs y, the kernel and the noise.\n\n\n\n\n\n","category":"method"},{"location":"api/#InducingPoints.inducingpoints-Tuple{Random.AbstractRNG, InducingPoints.OnlineIPSelection, AbstractVector}","page":"API","title":"InducingPoints.inducingpoints","text":" inducingpoints([rng::AbstractRNG], alg::OIPS, X::AbstractVector; kernel::Kernel)\n inducingpoints([rng::AbstractRNG], alg::OIPS, X::AbstractMatrix; obsdim=1, kernel::Kernel)\n\nSelect inducing points according using Online Inducing Points Selection. Requires as additional keyword argument the kernel.\n\n\n\n\n\n","category":"method"},{"location":"api/#InducingPoints.inducingpoints-Tuple{Random.AbstractRNG, RandomSubset, AbstractVector}","page":"API","title":"InducingPoints.inducingpoints","text":" inducingpoints([rng::AbstractRNG], alg::RandomSubset, X::AbstractVector; [weights::Vector=nothing])\n inducingpoints([rng::AbstractRNG], alg::RandomSubset, X::AbstractMatrix; obsdim=1, [weights::Vector=nothing])\n\nSelect inducing points by taking a Random Subset. Optionally accepts a weight vector for the inputs X.\n\n\n\n\n\n","category":"method"},{"location":"api/#InducingPoints.inducingpoints-Tuple{Random.AbstractRNG, SeqDPP, AbstractVector}","page":"API","title":"InducingPoints.inducingpoints","text":" inducingpoints([rng::AbstractRNG], alg::SeqDPP, X::AbstractVector; kernel::Kernel)\n inducingpoints([rng::AbstractRNG], alg::SeqDPP, X::AbstractMatrix; obsdim=1, kernel::Kernel)\n\nSelect inducing points according using Sequential Determinantal Point Processes. Requires as additional keyword argument the kernel.\n\n\n\n\n\n","category":"method"},{"location":"api/#InducingPoints.kmeans_seeding-Tuple{Random.AbstractRNG, AbstractVector, Integer, Distances.SemiMetric, Integer}","page":"API","title":"InducingPoints.kmeans_seeding","text":"Fast and efficient seeding for KMeans based on Fast and Provably Good Seeding for k-Means\n\n\n\n\n\n","category":"method"},{"location":"api/#InducingPoints.updateZ","page":"API","title":"InducingPoints.updateZ","text":"updateZ([rng::AbstractRNG], Z::AbstractVector, alg::OnIPSA, X::AbstractVector; kwargs...)\n\nReturn new vector of inducing points Z with data X and algorithm alg without changing the original one\n\n\n\n\n\n","category":"function"},{"location":"api/#InducingPoints.updateZ!","page":"API","title":"InducingPoints.updateZ!","text":"updateZ!([rng::AbstractRNG], Z::AbstractVector, alg::OnIPSA, X::AbstractVector; [kwargs...])\n\nUpdate inducing points Z with data X and algorithm alg. Requires additional keyword arguments for some algorithms. Also see InducingPoints.\n\n\n\n\n\n","category":"function"},{"location":"api/#InducingPoints.updateZ-Tuple{Random.AbstractRNG, AbstractVector, InducingPoints.OnlineIPSelection, AbstractVector}","page":"API","title":"InducingPoints.updateZ","text":"updateZ!([rng::AbstractRNG], Z::AbstractVector, alg::OIPS, X::AbstractVector; kernel::Kernel)\n\nUpdate inducing points Z with data X and the OnlineIPSelection algorithm. Requires the kernel as an  additional keyword argument. \n\n\n\n\n\n","category":"method"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = InducingPoints","category":"page"},{"location":"#Intro","page":"Home","title":"Intro","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"InducingPoints.jl aims at providing an easy way to select inducing points locations for Sparse Gaussian Processes both in an online and offline setting.  These are used most prominently in sparse GP regression (see e.g. `ApproximateGPs.jl)","category":"page"},{"location":"#Quickstart","page":"Home","title":"Quickstart","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"InducingPoints.jl provides the following list of algorithms. For details on the specific usage see the algorithms section.","category":"page"},{"location":"","page":"Home","title":"Home","text":"All algorithms inherit from AbstractInducingPointsSelection or AIPSA which can be passed to the different APIs.","category":"page"},{"location":"#Offline-Inducing-Points-Selection","page":"Home","title":"Offline Inducing Points Selection","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"These algorithms are designed to compute inducing points for a data set that is likely to remain unchanged.  If the data set changes, the algorithms have to be rerun from scratch.","category":"page"},{"location":"","page":"Home","title":"Home","text":"alg = KMeansAlg(10)\nZ = inducingpoints(alg, X; kwargs...)","category":"page"},{"location":"","page":"Home","title":"Home","text":"The Offline options are:","category":"page"},{"location":"","page":"Home","title":"Home","text":"KmeansAlg: Use the k-means algorithm to select centroids minimizing the square distance with the dataset. The seeding is done via k-means++. Note that the inducing points are not going to be a subset of the data.\nkDPP: Sample from a k-Determinantal Point Process to select k points. Z will be a subset of X.\nStdDPP: Sample from a standard Determinantal Point Process. The number of inducing points is not fixed here. Z will be a subset of X.\nRandomSubset : Sample randomly k points from the data set uniformly.\nGreedy: Will select a subset of X which maximizes the ELBO (in a stochastic way).\nCoverTree: Will build a tree to select the optimal nodes covering the data.","category":"page"},{"location":"#Online-Inducing-Points-Selection","page":"Home","title":"Online Inducing Points Selection","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Online selection algorithms compute an initial set similarly to the offline methods via inducingpoints. For successive changes of the data sets, InducingPoints.jl allows for efficient updating via updateZ!.","category":"page"},{"location":"","page":"Home","title":"Home","text":"alg = OIPS()\nZ = inducingpoints(alg, x_1; kwargs...)\nfor x in eachbatch(X)\n    updateZ!(Z, alg, x; kwargs...)\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"The Online options are:","category":"page"},{"location":"","page":"Home","title":"Home","text":"OnlineIPSelection: A method based on distance between inducing points and data\nUniGrid: A regularly-spaced grid whom edges are adapted given the data. Uses memory efficient custom type UniformGrid.\nSeqDPP: Sequential Determinantal Point Processes, subsets are regularly sampled from the new data batches conditioned on the existing inducing points.\nStreamKmeans: An online version of k-means.\nWebscale: Another online version of k-means","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"using Random: seed!\nseed!(42)\nusing InducingPoints\nusing CairoMakie\nusing CairoMakie.Colors\nusing CairoMakie.Colors.FixedPointNumbers: N0f8\nusing KernelFunctions\nD = 2\nN = 50\nM = 10\nx = [rand(D) .* [0.8, 1.0] for _ in 1:N]\nN₂ = 25\nx₂ = [rand(D) .* [0.2, 1.0] + [0.8, 0.0] for _ in 1:N₂]\ncolor_x = RGB{N0f8}(253 / 255, 132 / 255, 31 / 255)\ncolor_Z = RGB{N0f8}(225 / 255, 77 / 255, 42 / 255)\ncolor_Z2 = RGB{N0f8}(62 / 255, 109 / 255, 156 / 255)\ncolor_x2 = RGB{N0f8}(0 / 255, 18 / 255, 83 / 255)\nmarkersize = 15.0\nstrokewidth = 5.0\n\nfunction plot_inducing_points(x, Z, x₂ = nothing, Z₂ = nothing)\n    fig, ax, plt = scatter(\n        getindex.(x, 1),\n        getindex.(x, 2);\n        label = \"Original Data\",\n        color = color_x,\n        markersize,\n        markerstrokewidth = 2.0\n    )\n    xlims!(ax, [-0.05, 1.05])\n    ylims!(ax, [-0.05, 1.05])\n    scatter!(\n        ax,\n        getindex.(Z,1),\n        getindex.(Z, 2);\n        marker = :circle,\n        markersize = markersize + strokewidth,\n        color = RGBA(color_Z, 0.0),\n        strokewidth,\n        strokecolor = color_Z,\n        label = \"Inducing Points Z\"\n    )\n        \n    if !isnothing(Z₂)\n        scatter!(\n            ax,\n            getindex.(x₂, 1),\n            getindex.(x₂, 2);\n            markersize,\n            color = color_x2,\n            label = \"Additional Data\",\n        )\n        scatter!(\n            ax,\n            getindex.(Z₂,1),\n            getindex.(Z₂, 2); \n            # marker = :xcross, \n            markersize = markersize + 4 * strokewidth,\n            strokewidth,\n            color = RGBA(color_Z2, 0.0),\n            strokecolor = color_Z2,\n            label = \"Updated Z\",\n        )\n    end\n    fig[0, 1] = Legend(fig, ax; framevisible=false, tellwidth=false, tellheight=true, orientation=:horizontal)\n    return fig\nend","category":"page"},{"location":"algorithms/#available_algorithms","page":"Algorithms","title":"Available Algorithms","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"The algorithms available through InducingPoints.jl can be split into offline and online use. While all algorithms can be used to create one-off sets of inducing points, the online algorithms are designed in a way that allows for cheap updating.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"    Pages = [\"algorithms.md\"]\n    Depth = 3","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"We start with a set of N data points of dimension D, which we would like to reduce to only M < N points.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"D = 2\nN = 50\nM = 10\nx = [rand(D) .* [0.8, 1.0] for _ in 1:N]\nnothing # hide","category":"page"},{"location":"algorithms/#Offline-Algorithms","page":"Algorithms","title":"Offline Algorithms","text":"","category":"section"},{"location":"algorithms/#[KmeansAlg](@ref)","page":"Algorithms","title":"KmeansAlg","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"Uses the k-means algorithm to select centroids minimizing the square distance with the dataset. The seeding is done via k-means++. Note that the inducing points are not going to be a subset of the data.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"alg = KmeansAlg(M)\nZ = inducingpoints(alg, x)\nfig = plot_inducing_points(x, Z) # hide\nsave(\"kmeans.svg\", fig); nothing # hide","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"(Image: k-means plot)","category":"page"},{"location":"algorithms/#[kDPP](@ref)","page":"Algorithms","title":"kDPP","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"Sample from a k-Determinantal Point Process to select k points. Z will be a subset of X. Requires a kernel from KernelFunctions.jl","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"kernel = SqExponentialKernel()\nalg = kDPP(M, kernel)\nZ = inducingpoints(alg, x)\nfig = plot_inducing_points(x, Z) # hide\nsave(\"kdpp.svg\", fig); nothing # hide","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"(Image: k-DPP plot)","category":"page"},{"location":"algorithms/#[StdDPP](@ref)","page":"Algorithms","title":"StdDPP","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"Samples from a standard Determinantal Point Process. The number of inducing points is not fixed here. Z will be a subset of X. Requires a kernel from KernelFunctions.jl","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"kernel = with_lengthscale(SqExponentialKernel(), 0.2)\nalg = StdDPP(kernel)\nZ = inducingpoints(alg, x)\nfig = plot_inducing_points(x, Z) # hide\nsave(\"StdDPP.svg\", fig); nothing # hide","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"(Image: Standard DPP plot)","category":"page"},{"location":"algorithms/#[RandomSubset](@ref)","page":"Algorithms","title":"RandomSubset","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"Sample randomly k points from the data set uniformly.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"alg = RandomSubset(M)\nZ = inducingpoints(alg, x)\nfig = plot_inducing_points(x, Z) # hide\nsave(\"RandomSubset.svg\", fig); nothing # hide","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"(Image: Random subset plot)","category":"page"},{"location":"algorithms/#[Greedy](@ref)","page":"Algorithms","title":"Greedy","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"This algorithm will select a subset of X which maximizes the ELBO (Evidence Lower BOund), which is done in a stochastic way via minibatches of size s. This also requires passing the output data, the kernel and the noise level as additional arguments to inducingpoints.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"y = rand(N)\ns = 5\nkernel = with_lengthscale(SqExponentialKernel(), 0.2)\nnoise = 0.1\nalg = Greedy(M, s)\nZ = inducingpoints(alg, x; y, kernel, noise)\nfig = plot_inducing_points(x, Z) # hide\nsave(\"Greedy.svg\", fig); nothing # hide","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"(Image: Greedy algorithm plot)","category":"page"},{"location":"algorithms/#[CoverTree](@ref)","page":"Algorithms","title":"CoverTree","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"The CoverTree algorithm is a recent algorithm presented in Numerically Stable Sparse Gaussian Processes via Minimum Separation using Cover Trees . It relies on building a covering tree with the nodes representing the inducing points.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"alg = CoverTree(0.2)\nZ = inducingpoints(alg, x)\nfig = plot_inducing_points(x, Z) # hide\nsave(\"CoverTree.svg\", fig); nothing # hide","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"(Image: CoverTree algorithm plot)","category":"page"},{"location":"algorithms/#Online-Algorithms","page":"Algorithms","title":"Online Algorithms","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"These algorithms are useful if we assume that we will have another set of data points that we would like to incorporate into an existing inducing point set.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"D = 2 # hide\nN₂ = 25\nx₂ = [rand(D) .* [0.2, 1.0] + [0.8, 0.0] for _ in 1:N₂]\nnothing # hide","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"We can then update the inital set of inducing points Z via updateZ (or inplace via updateZ!).","category":"page"},{"location":"algorithms/#[OnlineIPSelection](@ref-InducingPoints.OnlineIPSelection)","page":"Algorithms","title":"OnlineIPSelection","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"A method based on distance between inducing points and data. This algorithm has several parameters to tune the result. It also requires the kernel to be passed as a keyword argument to inducingpoints and updateZ.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"kernel = with_lengthscale(SqExponentialKernel(), 0.2)\nalg = OIPS()\nZ = inducingpoints(alg, x; kernel)\nZ₂ = updateZ(Z, alg, x₂; kernel)\nfig = plot_inducing_points(x, Z, x₂, Z₂) # hide\nsave(\"OIPS.svg\", fig); nothing # hide","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"(Image: Online inducing point selection plot)","category":"page"},{"location":"algorithms/#[UniGrid](@ref)","page":"Algorithms","title":"UniGrid","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"A regularly-spaced grid whose edges are adapted given the data. The inducing points Z are returned as the UniformGrid custom type (see below).","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"alg = UniGrid(5)\nZ = inducingpoints(alg, x)\nZ₂ = updateZ(Z, alg, x₂)\nfig = plot_inducing_points(x, Z, x₂, Z₂) #hide\nsave(\"UniGrid.svg\", fig); nothing # hide","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"(Image: Unigrid plot)","category":"page"},{"location":"algorithms/#[UniformGrid](@ref)","page":"Algorithms","title":"UniformGrid","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"When using the UniGrid algorithm, InducingPoints.jl provides the memory-efficient custom type UniformGrid, which is essentially a wrapper around a Iterators.product. It functions in many ways like an AbstractVector, but does not explicitly store all elements of the grid. Therefore, shown via the example of a two-dimensional grid, the object size only depends on the dimension, not on the number of grid points.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"It is optimized to be very efficient with kernelmatrix function provided by Kernelfunctions.jl. However, compared to an explicitly stored Vector of grid points, it incurs additional overhead when used with other vector operations (illustrated below for the example of broadcasting sum).","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"(Image: Uniform grid bench plot)","category":"page"},{"location":"algorithms/#[SeqDPP](@ref)","page":"Algorithms","title":"SeqDPP","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"Sequential Determinantal Point Processes, subsets are regularly sampled from the new data batches conditioned on the existing inducing points.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"kernel = with_lengthscale(SqExponentialKernel(), 0.2)\nalg = SeqDPP()\nZ = inducingpoints(alg, x; kernel)\nZ₂ = updateZ(Z, alg, x₂; kernel)\nfig = plot_inducing_points(x, Z, x₂, Z₂) # hide\nsave(\"SeqDPP.svg\", fig); nothing # hide","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"(Image: Sequential DPP plot)","category":"page"},{"location":"algorithms/#[StreamKmeans](@ref)","page":"Algorithms","title":"StreamKmeans","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"An online version of k-means.","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"alg = StreamKmeans(M)\nZ = inducingpoints(alg, x)\nZ₂ = updateZ(Z, alg, x₂)\nfig = plot_inducing_points(x, Z, x₂, Z₂) # hide\nsave(\"StreamKmeans.svg\", fig); nothing # hide","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"(Image: Stream k-means plot)","category":"page"},{"location":"algorithms/#[Webscale](@ref)","page":"Algorithms","title":"Webscale","text":"","category":"section"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"Another online version of k-means","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"alg = Webscale(M)\nZ = inducingpoints(alg, x)\nZ₂ = updateZ(Z, alg, x₂)\nfig = plot_inducing_points(x, Z, x₂, Z₂) # hide\nsave(\"Webscale.svg\", fig); nothing # hide","category":"page"},{"location":"algorithms/","page":"Algorithms","title":"Algorithms","text":"(Image: Webscale plot)","category":"page"}]
}