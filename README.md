# DTSVM_IDS
Intrusion Detection System using DTSVM classifier.


Builds a decision tree containing binary SVM classifiers at each inner node, with a genetic algorithm to optimise node segmentation.

Followed this publication for model architecture

  -> https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6077641

Using KDD Cup 1999 Data

  -> http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

It works, although many speed improvements are to look out for.

# How to use:

features = x :: Matrix 2 Float64
labels = y :: Vector Int

    x = vcat(snormal, class2, class3, class4, class5)
    y = vcat(fill(1, 500), fill(2, 500), fill(3, 500), fill(4, 500), fill(5, 500))
    
Initialise our Partition structure containing features and labels

    samples = partitionSamples(convert(Array{Any}, hcat(x, y)))
    
Build the Decision Tree
buildTree(rawsamples::Samples, gen::Int64, stop::Int64, pop::Int64, sel::Float64, α::Float64, γ::Float64)
  GA PARAMETERS:
  - gen = number of generations
  - stop = useless for now but there to stop GA at fitness plateau
  - pop = number of chromosomes
  - sel = selection ration at each generation
  - α = allele selection ratio for mutation
  - γ = mutation strengh ratio
  
         tree = buildTree(samples, 30, 10, 10, 0.4, 0.2, 0.25)
      
Generate prediction on sample

      prediction = predict(tree, samples.labels, x'[:,1502:1502])
      labels[1502] # check prediction
