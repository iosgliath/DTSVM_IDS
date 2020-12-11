# DTSVM_NIDS

# Network Intrusion Detection System using DTSVM classifier.


Builds a decision tree containing binary SVM classifiers at each inner node, with a genetic algorithm to optimise node segmentation.

The intuition is that we want to separate our dataset at each nodes with the right combination of classes (the one that maximise the euclidan distance between the features of the two classes combinations) => it gives more "room" for our binary SVM to draw support vectors later on. 

Genetic Algorithm is used at each node to try find the best solution possible to this combination problem (distribute all classes from node into two subsets Xp and Xn). Binary SVM is then trained on this subdataset. If Xp or Xn are only one class, we got a leaf node, else, an inner node.
Tree is built, with (m) leaf nodes and (m-1) inner nodes (m = number of classes in our dataset).

Followed this publication for model architecture

  -> https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6077641

Using KDD Cup 1999 Data

  -> http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

It works, although many speed improvements are to look out for (on top of hyperparameter tuning).

Decision tree and GA are implemented from scratch.
SVM part is using LIBSVM.jl (was not the focus of this project, you can look out my svm repo to look out for a SVM SMO implmentation)

Any feedback welcome!

# How to use:

features = x :: Array 2 Float64

labels = y :: Vector Any

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
  - sel = chromosome selection ratio at each generation (for breeding next generation)
  - α = allele selection ratio (for mutation)
  - γ = mutation strengh
  
         tree = buildTree(samples, 30, 10, 10, 0.4, 0.2, 0.25)
      
Generate prediction on sample

      prediction = predict(tree, samples.labels, x'[:,1502:1502])
      y[1502] # check prediction
