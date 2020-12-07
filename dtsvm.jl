using DataFrames, Combinatorics, LinearAlgebra, Distributions, Random, StatsBase, CSV, DelimitedFiles, LIBSVM


mutable struct Xk
    id::Int64
    x::Array{Float64, 2}
end

mutable struct Samples
    partitions::Vector{Xk}
    labels::Dict
end

mutable struct Chromosome
    alleles::Vector{Int}
    fitness::Float64
end

mutable struct Node
	"""
	cpidx & cnidx are used to train binary SVM labeling Xp as 1 and Xn as 0
		if either is Xp or Xn are unique
			they are a leaf = predict this label if SVM chose this on testing data
		else
			create another node, with new Xp and Xn repartition and another SVM classifier
		end
	m - 1 inner nodes
	m leaf nodes
	"""
	id::Int64
	leafOutput::Union{Bool, Int64} # false || 0 || 1
	cpidx::Vector{Int64} # labels in Xp, if unique, leafOutput = 1
	cnidx::Vector{Int64} # labels in Xn, if unique, leafOutput = 0
	p::Union{Node, Int64} # node || unique label: if result of svm is p, pred is Int64 in samples dict
	n::Union{Node, Int64} # node || unique label: if result of svm is n, pred is Int64 in samples dict
	svm::Union{BinaryModel, LIBSVM.SVM{Int64}, Int64}
end


bernoulli(p) = rand() < p


"""
###################################################
GENETIC ALGORITHM
###################################################
"""

function clusteringCenter(xk::Xk)
    sum(sum(xk.x, dims = 2)) / size(xk.x, 1)
end

function clusteringCenter(xks::Vector{Xk})
    sum(clusteringCenter.(xks)) / length(xks)
end

function clusteringCenter(cpidx::Vector{Int64}, cnidx::Vector{Int64}, xks::Vector{Xk})
    abs(clusteringCenter(xks[cpidx]) - clusteringCenter(xks[cnidx]))
end

function ga_fitness!(chromosomes::Vector{Chromosome}, xks::Vector{Xk})
    for c in chromosomes
        cpidx = findall(x -> x == 1, c.alleles)
        cnidx = findall(x -> x == 0, c.alleles)
        c.fitness = clusteringCenter(cpidx, cnidx, xks)
    end
end

function initChromosomes(m::Int64, pop::Int64)
    out = Vector{Chromosome}(undef, pop)
    for i = 1:pop
        tmp = Vector{Int64}(undef, m)
        while true
            tmp = collect((rand([0,1]) for _ in 1:m))
            if length(findall(x -> x == 0, tmp)) != 0 && length(findall(x -> x == 1, tmp)) != 0
                break
            end
        end
        out[i] = Chromosome(tmp, 0.0)
    end
    return out
end

function selection(chromosomes::Vector{Chromosome}, sel::Float64)
    sort(chromosomes, by = x -> x.fitness, rev = true)[1:floor(Int64, length(chromosomes) * sel)]
end

function selection!(chromosomes::Vector{Chromosome})
    sort!(chromosomes, by = x -> x.fitness, rev = true)
end

function initChildren!(chromosomes::Vector{Chromosome}, pop::Int64, selidx::Int64)
    n = pop-length(chromosomes[1:selidx])
    mod(n, 2) != 0 && return print("\nError: mod(pop*sel, 2) != 0 => can't generate children")
    pool1 = rand()
    for i = n+1:2:pop
        p1 = rand(chromosomes)
        tmppool = filter(x -> x != p1, chromosomes)
        p2 = rand(tmppool)
        c = rand(1:length(p1.alleles))
        chromosomes[i] = Chromosome(vcat(p1.alleles[1:c], p2.alleles[c+1:end]) , 0.0)
        chromosomes[i+1] = Chromosome(vcat(p2.alleles[1:c], p1.alleles[c+1:end]) , 0.0)
    end
end

function mutate!(chromosome::Chromosome, γ::Float64)
    for i = 1:length(chromosome.alleles)
        if bernoulli(γ)
            if chromosome.alleles[i] == 0
                chromosome.alleles[i] = 1
            elseif chromosome.alleles[i] == 1
                chromosome.alleles[i] = 0
            end
        end
    end
end

function mutate!(chromosomes::Vector{Chromosome}, α::Float64, γ::Float64)
    for i = 1:length(chromosomes)
        if bernoulli(α)
            mutate!(chromosomes[i], γ)
        end
    end
end

function repair!(chromosomes::Vector{Chromosome})
    m = length(chromosomes[1].alleles)
    e1 = findall(x->x.alleles == ones(Int, m), chromosomes)
    e2 = findall(x->x.alleles == zeros(Int, m), chromosomes)
    e = vcat(e1, e2)
    if length(e) != 0
        for idx in e
            i = rand(1:length(chromosomes[idx].alleles))
            if chromosomes[idx].alleles[i] == 0
                chromosomes[idx].alleles[i] = 1
            elseif chromosomes[idx].alleles[i] == 1
                chromosomes[idx].alleles[i] = 0
            end
        end
    end
end

function ga_optimizeClustering(xks::Vector{Xk}, gen::Int64, stop::Int64, pop::Int64, sel::Float64, α::Float64, γ::Float64)
    m = length(xks)
    selidx = floor(Int64, pop * sel)

    chromosomes = initChromosomes(m , pop)
    ga_fitness!(chromosomes, xks)
    hercules = Chromosome
    """
    to do
        add stopping mechanism at fitness plateau
    """
    for g in 1:gen
        selection!(chromosomes)
        initChildren!(chromosomes, pop, selidx)
        mutate!(chromosomes, α, γ)
        repair!(chromosomes)
        ga_fitness!(chromosomes, xks)
        hercules = sort(chromosomes, by = x -> x.fitness, rev = true)[1]
    end

    return hercules
end


"""
###################################################
DECISION TREE
###################################################
"""

function buildSVMTrainingSet(rawsamples::Samples, cpidx::Vector{Int}, cnidx::Vector{Int})
    # px = rawsamples.partitions[cpidx]
    # nx = rawsamples.partitions[cnidx]

    #pxvec = [collect(i.x) for i in px]
    pxflt = reduce(vcat, [collect(i.x) for i in rawsamples.partitions[cpidx]])
    nxflt = reduce(vcat, [collect(i.x) for i in rawsamples.partitions[cnidx]])

    instances = vcat(pxflt, nxflt)'

    labels = vcat(ones(Int, size(pxflt,1)), zeros(Int, size(nxflt,1)))
    return instances, labels
end

function initSplit(rawsamples::Samples, id::Int64, xks::Vector{Xk}, gen::Int64, stop::Int64, pop::Int64, sel::Float64, α::Float64, γ::Float64)

	hercules = ga_optimizeClustering(xks, gen, stop, pop, sel, α, γ)
	cpidx = findall(x -> x ∈ xks[findall(x -> x == 1, hercules.alleles)], rawsamples.partitions)
	cnidx = findall(x -> x ∈ xks[findall(x -> x == 0, hercules.alleles)], rawsamples.partitions)
	isLeafOut = false
	p = 0
	n = 0
	length(cpidx) == 1  && (isLeafOut = 1; p = rawsamples.partitions[cpidx[1]].id)
	length(cnidx) == 1  && (isLeafOut = 0; n = rawsamples.partitions[cnidx[1]].id)

    (instances, labels) = buildSVMTrainingSet(rawsamples, cpidx, cnidx)
    model = svmtrain(instances[:, 1:2:end], labels[1:2:end])

	return Node(id, isLeafOut, cpidx, cnidx, p, n, model)
end

function splitting!(rawsamples::Samples, xks::Vector{Xk}, node::Node, gen::Int64, stop::Int64, pop::Int64, sel::Float64, α::Float64, γ::Float64)

	if node.p == 0

		tmp1 = Vector{Xk}(undef, length(node.cpidx))
		s1 = 1
		[(tmp1[s1] = rawsamples.partitions[i]; s1 += 1) for i in node.cpidx]
		node.p = initSplit(samples, node.id + 1, tmp1, gen, stop, pop, sel, α, γ)

		tmp2 = Vector{Xk}(undef, length(node.p.cpidx))
		s2 = 1
		[(tmp2[s2] = rawsamples.partitions[i]; s2 += 1) for i in node.p.cpidx]
		splitting!(rawsamples, tmp2, node.p, gen, stop, pop, sel, α, γ)
	end

	if node.n == 0

		tmp1 = Vector{Xk}(undef, length(node.cnidx))
		s1 = 1
		[(tmp1[s1] = rawsamples.partitions[i]; s1 += 1) for i in node.cnidx]
		node.n = initSplit(samples, node.id +1, tmp1, gen, stop, pop, sel, α, γ)

		tmp2 = Vector{Xk}(undef, length(node.n.cnidx))
		s2 = 1
		[(tmp2[s2] = rawsamples.partitions[i]; s2 += 1) for i in node.n.cnidx]
		splitting!(rawsamples, tmp2, node.n, gen, stop, pop, sel, α, γ)
	end
end

function buildTree(rawsamples::Samples, gen::Int64, stop::Int64, pop::Int64, sel::Float64, α::Float64, γ::Float64)
	root = initSplit(rawsamples, 1, rawsamples.partitions, gen, stop, pop, sel, α, γ)
	splitting!(rawsamples, rawsamples.partitions, root, gen, stop, pop, sel, α, γ)
	return root
end

function predict(node::Node, labels::Dict, sample)
	#p = predict(sample, labels, node.svm)
    (p, _) = svmpredict(node.svm, sample)	#p = 1
	if p[1] == 1
		if typeof(node.p) == Node
			return predict(node.p, labels, sample)
		else
			return labels[node.p]
		end
	elseif p[1] == 0
		if typeof(node.n) == Node
			return predict(node.n, labels, sample)
		else
			return labels[node.n]
		end
	end
end


"""

CYBER DATA

"""

featLabels = [
    "duration"
    "protocol_type"
    "service"
    "flag"
    "src_bytes"
    "dst_bytes"
    "land"
    "wrong_fragment"
    "urgent"
    "hot"
    "num_failed_logins"
    "logged_in"
    "num_compromised"
    "root_shell"
    "su_attempted"
    "num_root"
    "num_file_creations"
    "num_shells"
    "num_access_files"
    "num_outbound_cmds"
    "is_host_login"
    "is_guest_login"
    "count"
    "srv_count"
    "serror_rate"
    "srv_serror_rate"
    "rerror_rate"
    "srv_rerror_rate"
    "same_srv_rate"
    "diff_srv_rate"
    "srv_diff_host_rate"
    "dst_host_count"
    "dst_host_srv_count"
    "dst_host_same_srv_rate"
    "dst_host_diff_srv_rate"
    "dst_host_same_src_port_rate"
    "dst_host_srv_diff_host_rate"
    "dst_host_serror_rate"
    "dst_host_srv_serror_rate"
    "dst_host_rerror_rate"
    "dst_host_srv_rerror_rate"
    "class"
]

choice = ["src_bytes",
     "dst_bytes",
     "hot",
     "count",
     "srv_count",
     "diff_srv_rate",
     "dst_host_count",
     "dst_host_srv_count",
     "dst_host_same_srv_rate",
     "dst_host_diff_srv_rate",
     "dst_host_same_src_port_rate",
     "dst_host_serror_rate",
     "protocol_type",
     "service",
     "flag",
     "class"
]

 """
 ###################################################
 DATA UTILITIES
 ###################################################
 """

 function partitionSamples(data::Array{Any,2})
     labels = unique(data[:,end])
 	features = data[:,1:end-1]

     m = length(labels)
     xks = Vector{Xk}(undef, m)

 	labeldct = Dict()

     for i in 1:m
         yiidx = findall(x -> x == labels[i], data[:,end])
         xks[i] = Xk(i, features[yiidx,:])
 		#labeldct[i] = String(labels[i])
 		l = labels[i]
 		labeldct[i] = "$l"
     end
     return Samples(xks, labeldct)
 end

function splitLines(data::Array{Any,2}, keep::Vector{Int})
    l = length(data)
    out = Array{SubString{String},2}(undef, l, length(keep))
    for i = 1:l
        out[i,:] = split(data[i], ",")[keep]
    end
    return out
end

function encodeFeatures(data::Any, nominalidx::Vector{Int})
	nsamples = size(data, 1)
	nfeatures = size(data, 2)

	numericalidx = filter(x -> x ∉ nominalidx, collect(1:nfeatures))

	num = Array{Float64, 2}(undef, nsamples, length(numericalidx))
	for x in 1:length(numericalidx)
		num[:,x] = p(numericalidx[x], data)
	end
	numn = min_max_scaling(num)

	len = 0
	nom = Array{Float64, 2}(undef, nsamples, length(nominalidx))
	for k = 1:length(nominalidx)
		nom[:,k] = col2dct(data[:,nominalidx[k]])
		len += length(unique(data[:,nominalidx[k]]))
		print("\n ", unique(data[:,nominalidx[k]]))
	end
	# print("\nlen ", len)



	onehot = Array{Float64}(undef, nsamples, len)

	for i = 1:size(nom, 1)
		# print("\nrow ", i)
		vec = zeros()
		print("\n")
		if mod(size(nom, 1), 1000) == 0
			print("\n => ", i)
		end
		for j = 1:size(nom, 2)
			l = length(unique(nom[:,j]))
			# print("\nsubl ", l)
			#bloc = Vector{Float64}(undef, nsamples, l)

			subhot = zeros(Float64, l)
			subhot[floor(Int, nom[i,j])] = 1.0
			vec = vcat(vec, subhot)

			#onehot = vcat(onehot, subhot)
		end
		onehot[i,:] = vec[2:end]
	end

	return num, numn, nom, onehot
	# return num, numn
end

p(k, in) = parse.(Float64, in[collect(1:size(in,1)), k])

function col2dct(col)
    v = unique(col)
    w = Float64.(collect(1:length(v)))
    d = Dict(v[i] => w[i] for i ∈ 1:length(v))
    c = getindex.(Ref(d), col)
    return c
end

standardisation(x::Float64, μ::Float64, σ::Float64) = (x - μ) / σ
function standardisation(col::Vector{Float64})
    μ = mean(col)
    σ = std(col)
    return standardisation.(col, μ, σ)
end

mean_normalisation(x::Float64, μ::Float64, min::Float64, max::Float64) = (x - μ) / (max - min)
function mean_normalisation(col::Vector{Float64})
    μ = mean(col)
    min = minimum(col)
    max = maximum(col)
    return mean_normalisation.(col, μ, min, max)
end

min_max_scaling(x::Float64, min::Float64, max::Float64) = (x - min) / (max - min)
function min_max_scaling(col::Vector{Float64})
    min = minimum(col)
    max = maximum(col)
    return min_max_scaling.(col, min, max)
end
function min_max_scaling(array::Array{Float64, 2})
	out = Array{Float64, 2}(undef, size(array))
	for i = 1:size(array, 2)
		out[:,i] = min_max_scaling(array[:,i])
	end
	return out
end

function sampleFeatures(features::Array{Float64,2}, n::Int64)
    s = size(features, 1)
    out = Array{Float64,2}(undef, n, size(features, 2))
    for i=1:n
        p = rand(1:s)
        out[i,:] = features[p,:]
    end
    return out
end

pwd()
cd()
cd("Documents/julia/data/cyber/")


data = readdlm("kddcup.data.corrected")


"""
let's not analyse 5 millions rows on my laptop
    data janitoring shenanigans
"""
keep =  findall(x -> x ∈ choice, featLabels)
lines = splitLines(data, keep)

ys = col2dct(lines[:,end])

c1 = findall(x->x==1, ys)
c1idx = rand(c1, 1000)
unique(c1idx)

c5 = findall(x->x==5, ys)
c5idx = rand(c5, 1000)

c6 = findall(x->x==6, ys)
c6idx = rand(c6, 1000)

c10 = findall(x->x==10, ys)
c10idx = rand(c10, 1000)
unique(c10idx)


c11 = findall(x->x==11, ys)
c11idx = rand(c11, 1000)
unique(c11idx)


loadidx = unique(vcat(c1idx, c5idx, c6idx, c10idx, c11idx))
load = lines[loadidx,:]

num, numn, nom, onehot = encodeFeatures(load[:,1:end-1], [1,2,3])

xx = hcat(numn, onehot)
yy = col2dct(load[:,end])


normalidx = findall(x -> x == 1, yy)
class5idx = findall(x -> x == 5, yy)
class2idx = findall(x -> x == 2, yy)
class4idx = findall(x -> x == 4, yy)
class3idx = findall(x -> x == 3, yy)
nmapidx = findall(x -> x == 18, yy)
bufferoverflowidx = findall(x -> x == 2, yy)
teardropidx = findall(x -> x == 9, yy)
satanidx = findall(x -> x == 16, yy)


snormal = sampleFeatures(xx[normalidx,:], 500)
class2 = sampleFeatures(xx[class2idx,:], 500)
class3 = sampleFeatures(xx[class3idx,:], 500)
class4 = sampleFeatures(xx[class4idx,:], 500)
class5 = sampleFeatures(xx[class5idx,:], 500)

#x = vcat(steardrop, snmap,  steardrop, ssatanix, sbufferoverflow)

x = vcat(snormal, class2, class3, class4, class5)
y = vcat(fill(1, 500), fill(2, 500), fill(3, 500), fill(4, 500), fill(5, 500))
samples = partitionSamples(convert(Array{Any}, hcat(x, y)))
tree = buildTree(samples, 30, 10, 10, 0.4, 0.2, 0.25)
prediction = predict(tree, samples.labels, x'[:,1502:1502])
labels[1502]
