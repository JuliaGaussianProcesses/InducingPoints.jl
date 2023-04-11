var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = InducingPoints","category":"page"},{"location":"#InducingPoints","page":"Home","title":"InducingPoints","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"InducingPoints.jl aims at providing an easy way to select inducing points locations for Sparse Gaussian Processes both in an online and offline setting.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The point selection is splitted in the online (OnIPSA) and offline settings.","category":"page"},{"location":"","page":"Home","title":"Home","text":"All algorithms inherit from AbstractInducingPointsSelection or AIPSA which can be passed to the different APIs","category":"page"},{"location":"#Offline-Inducing-Points-Selection","page":"Home","title":"Offline Inducing Points Selection","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"alg = KMeansAlg(10)\nZ = inducingpoints(alg, X; kwargs...)","category":"page"},{"location":"","page":"Home","title":"Home","text":"The Offline options are:","category":"page"},{"location":"","page":"Home","title":"Home","text":"KmeansAlg : use the k-means algorithm to select centroids minimizing the square distance with the dataset. The seeding is done via k-means++. Note that the inducing points are not going to be a subset of the data\nkDPP : sample from a k-Determinantal Point Process to select k points. Z will be a subset of X\nStdDPP : sample from a standard Determinantal Point Process. The number of inducing points is not fixed here. Z will be a subset of X\nRandomSubset : sample randomly k points from the data set uniformly.\nGreedy : Will select a subset of X which maximizes the ELBO (in a stochastic way)","category":"page"},{"location":"#Online-Inducing-Points-Selection","page":"Home","title":"Online Inducing Points Selection","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Online selection is a bit more involved.","category":"page"},{"location":"","page":"Home","title":"Home","text":"alg = OIPS()\nZ = initZ(alg, x_1; kwargs...)\nfor x in eachbatch(X)\n    updateZ!(Z, alg, x; kwargs...)\nend","category":"page"},{"location":"","page":"Home","title":"Home","text":"With init, a first instance of Z is created. update! will then update the vectors in place.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The Online options are:","category":"page"},{"location":"","page":"Home","title":"Home","text":"OnlineIPSelection : A method based on distance between inducing points and data\nUniGrid : A regularly-spaced grid whom edges are adapted given the data\nSeqDPP : Sequential Determinantal Point Processes, subsets are regularly sampled from the new data batches conditionned on the existing inducing points.\nStreamKmeans : An online version of k-means.\nWebscale : Another online version of k-means","category":"page"},{"location":"#Index","page":"Home","title":"Index","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"#API","page":"Home","title":"API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Modules = [InducingPoints]","category":"page"},{"location":"#InducingPoints.Greedy","page":"Home","title":"InducingPoints.Greedy","text":"GreedyIP(m::Int, s::Int)\n\nm is the desired number of inducing points\ns is the minibatch size on which to select a new inducing point\n\nGreedy approach first proposed by Titsias[1]. Algorithm loops over minibatches of data and select the best ELBO improvement.\n\n[1] Titsias, M. Variational Learning of Inducing Variables in Sparse Gaussian Processes. Aistats 5, 567–574 (2009).\n\n\n\n\n\n","category":"type"},{"location":"#InducingPoints.KmeansAlg","page":"Home","title":"InducingPoints.KmeansAlg","text":"KmeansAlg(m::Int, metric::SemiMetric=SqEuclidean(); nMarkov = 10, tol = 1e-3)\n\nk-Means [1] initialization on the data X taking m inducing points. The seeding is computed via [2], nMarkov gives the number of MCMC steps for the seeding.\n\nArguments\n\nk::Int : Number of inducing points\nmetric::SemiMetric : Metric used to compute the distances for the k-means algorithm\n\nKeyword Arguments\n\nnMarkov::Int : Number of random steps for the seeding\ntol::Real : Tolerance for the kmeans algorithm\n\n[1] Arthur, D. & Vassilvitskii, S. k-means++: The advantages of careful seeding. in Proceedings of the eighteenth annual ACM-SIAM symposium on Discrete algorithms 1027–1035 (Society for Industrial and Applied Mathematics, 2007). [2] Bachem, O., Lucic, M., Hassani, S. H. & Krause, A. Fast and Provably Good Seedings for k-Means. Advances in Neural Information Processing Systems 29 55–63 (2016) doi:10.1109/tmtt.2005.863818.\n\n\n\n\n\n","category":"type"},{"location":"#InducingPoints.OnlineIPSelection","page":"Home","title":"InducingPoints.OnlineIPSelection","text":"OIPS(ρ_accept=0.8; η=0.95, kmax=Inf, kmin=10, ρ_remove=Inf)\nOIPS(kmax, η=0.98, kmin=10)\n\nOnline Inducing Points Selection. Method from the paper include reference here.\n\n\n\n\n\n","category":"type"},{"location":"#InducingPoints.RandomSubset","page":"Home","title":"InducingPoints.RandomSubset","text":"RandomSubset(m::Int)\n\nArguments\n\nm::Int: Number of inducing points\n\nUniform sampling of a subset of m points ofthe data.\n\n\n\n\n\n","category":"type"},{"location":"#InducingPoints.SeqDPP","page":"Home","title":"InducingPoints.SeqDPP","text":"SeqDPP()\n\nSequential sampling via Determinantal Point Processes\n\n\n\n\n\n","category":"type"},{"location":"#InducingPoints.StdDPP","page":"Home","title":"InducingPoints.StdDPP","text":"StdDPP(kernel::Kernel)\n\nStandard DPP (Determinantal Point Process) sampling given kernel. The size of the returned Z is not fixed (but cannot be empty unlike in a classical DPP)\n\n\n\n\n\n","category":"type"},{"location":"#InducingPoints.StreamKmeans","page":"Home","title":"InducingPoints.StreamKmeans","text":"StreamKmeans(m_target::Int)\n\nOnline clustering algorithm [1] to select inducing points in a streaming setting. Reference : [1] Liberty, E., Sriharsha, R. & Sviridenko, M. An Algorithm for Online K-Means Clustering. arXiv:1412.5721 cs.\n\n\n\n\n\n","category":"type"},{"location":"#InducingPoints.UniGrid","page":"Home","title":"InducingPoints.UniGrid","text":"UniGrid(m::Int)\n\nwhere m is the number of points on each dimension Adaptive uniform grid based on [1]\n\n[1] Moreno-Muñoz, P., Artés-Rodríguez, A. & Álvarez, M. A. Continual Multi-task Gaussian Processes. (2019).\n\n\n\n\n\n","category":"type"},{"location":"#InducingPoints.Webscale","page":"Home","title":"InducingPoints.Webscale","text":"Webscale(m::Int)\n\nOnline k-means algorithm based on [1].\n\n[1] Sculley, D. Web-scale k-means clustering. in Proceedings of the 19th international conference on World wide web - WWW ’10 1177 (ACM Press, 2010). doi:10.1145/1772690.1772862.\n\n\n\n\n\n","category":"type"},{"location":"#InducingPoints.kDPP","page":"Home","title":"InducingPoints.kDPP","text":"kDPP(m::Int, kernel::Kernel)\n\nk-DPP (Determinantal Point Process) will return a subset of X of size m, according to DPP probability\n\n\n\n\n\n","category":"type"},{"location":"#InducingPoints.distance-Tuple{Any, Any, Nothing}","page":"Home","title":"InducingPoints.distance","text":"Compute the distance (kernel if included) between a point and a findnearestcenter\n\n\n\n\n\n","category":"method"},{"location":"#InducingPoints.find_nearest_center","page":"Home","title":"InducingPoints.find_nearest_center","text":"Find the closest center to X among Z, return the index and the distance\n\n\n\n\n\n","category":"function"},{"location":"#InducingPoints.inducingpoints","page":"Home","title":"InducingPoints.inducingpoints","text":" inducingpoints([rng::AbstractRNG], alg::OffIPSA, X::AbstractVector; kwargs...)\n inducingpoints([rng::AbstractRNG], alg::OffIPSA, X::AbstractMatrix; obsdim=1, kwargs...)\n\nSelect inducing points according to the algorithm alg.\n\n\n\n\n\n","category":"function"},{"location":"#InducingPoints.initZ","page":"Home","title":"InducingPoints.initZ","text":" initZ([rng::AbstractRNG], alg::OnIPSA, X::AbstractVector; kwargs...)\n initZ([rng::AbstractRNG], alg::OnIPSA, X::AbstractMatrix; obsdim=1, kwargs...)\n\nSelect inducing points according to the algorithm alg and return a Vector of Vector.\n\n\n\n\n\n","category":"function"},{"location":"#InducingPoints.kmeans_seeding-Tuple{Random.AbstractRNG, AbstractVector{T} where T, Integer, Distances.SemiMetric, Integer}","page":"Home","title":"InducingPoints.kmeans_seeding","text":"Fast and efficient seeding for KMeans based on `Fast and Provably Good Seeding for k-Means\n\n\n\n\n\n","category":"method"},{"location":"#InducingPoints.updateZ","page":"Home","title":"InducingPoints.updateZ","text":"updateZ([rng::AbstractRNG], Z::AbstractVector, alg::OnIPSA, X::AbstractVector; kwargs...)\n\nReturn new vector of inducing points Z with data X and algorithm alg without changing the original one\n\n\n\n\n\n","category":"function"},{"location":"#InducingPoints.updateZ!","page":"Home","title":"InducingPoints.updateZ!","text":"updateZ!([rng::AbstractRNG], Z::AbstractVector, alg::OnIPSA, X::AbstractVector; kwargs...)\n\nUpdate inducing points Z with data X and algorithm alg\n\n\n\n\n\n","category":"function"}]
}
