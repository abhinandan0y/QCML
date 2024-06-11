#### Quantum Computing in HealthCare — Protein Folding Part-1
Fractal AI Research

“Nature isn’t classical, dammit, and if you want to make a simulation of nature, you’d better make it quantum mechanical, and by golly it’s a wonderful problem, because it doesn’t look so easy.” — Richard Feynman

As indicated by Feynman, Chemistry simulation is an area which is more naturally aligned to Quantum simulations and hence quantum computers will invariably have a major advantage over classical computers. Problems of this type have exponential computational need that hinder attempts to understand even modestly large quantum systems with every electron added to the polypeptide molecule. Here in this series we will explore usage of Quantum computing algorithms in healthcare involving intricate study of molecular structures through chemical simulations.
```
Table of Contents
Drug Discovery & Design
What is Protein Folding problem ?
What is Quantum Machine Learning ?
Role of Quantum Machine Learning in Drug Discovery
VQE (Variational Quantum Eigensolver)
Quantum Annealing
QSVM (Quantum Support Vector Machine)
QNN (Quantum Neural Networks)
QGAN (Quantum Generative Adversarial Network)
QRBM (Quantum Restricted Boltzman machine)
```

Drug Discovery & Design: A new drug typically takes 10 to 15 years to progress from discovery to launch with total cost up to $2 billion and the success rate of the drug is less than 10%. For these reasons, biopharma companies count on a few flagship drugs to reap payback of more than $180 billion that the pharma industry collectively spends each year on R&D alone. This research pipeline can be expedited many folds by studying drugs & their interactions at molecular level. Drug Discovery & Design can be broadly categorized into following three key areas:
```
1. Lead generation for small-molecule drugs: Accelerate the identification of chemical compounds that selectively bind a disease-related protein target.

2. Protein structure prediction: Accurately determine the three-dimensional structure of a protein of based on its amino acid sequence.

3. Protein engineering & design: Generate novel, complex biomolecules that selectively activate, inhibit, or modulate certain biological functions.
```
Protein Folding is at core of all these three above areas, Experimental Research done by IBM, Dwave, ProteinQure & others show promising results using Quantum algorithms projecting exponential improvement compared to classical neural network approaches. With relatively slow & inefficient approaches in classical world plus lack of time for extensive live experimentation the progress in drug discovery has been very slow for ex. during the Current Covid Pandemic, there was a critical urgency of rapid progress in this field, hopefully next pandemic if not addressed right away, will at least be handled more efficiently with help of Quantum Computing Machines.

What is Protein Folding problem ? Proteins can fold in many ways because they are made up of large number of chains of amino acids which combinatorically can assume several possible shapes. It is believed that proteins fold themselves in a state where they are stable, and this is related to the lowest energy state. Some diseases are considered to be caused by proteins which do not fold properly and then accumulate in organs causing diseases like Alzheimer’s, Huntington’s, and Parkinson’s and many more.

As demonstrated by Levinthal’s paradox, it would take longer than the age of the known universe to randomly enumerate all possible configurations of a typical protein before reaching the true 3D structure — yet proteins themselves fold spontaneously, within milliseconds. Protein folding problem is categorized as NP-Hard problem and has already inspired countless developments, from IBM’s efforts in supercomputing (BlueGene), to novel citizen science efforts (Folding@Home and FoldIt) to new engineering realms, such as rational protein design.


Image source ProteinQure
```
Attempts to study protein structures using NMR (nuclear magnetic resonance), cryo-electron microscopy and X-ray crystallography have been carried out, but these methods are very expensive and time consuming. Recent advances in machine learning from DeepMind developed AlphaFold, a neural network framework able to predict properties of proteins. Alphafold is also bound by several approximations of the neural network & takes significant amount of compute power & time. The key question is whether there are other ways to predict protein folding more efficiently and faster.

What is Quantum Machine Learning ? Quantum machine learning is the area that explores the interplay of concepts from quantum computing and machine learning. It extends the pool of hardware & software for machine learning by an entirely new type of computing devices & algorithmic approaches. Quantum computers can be used and trained like neural networks. We can systematically use the physical control parameters, such as an electromagnetic field or a laser pulse frequency, to solve problems. For example, a trained circuit can be used to classify the content of images, by encoding the image into the physical state of the device and taking measurements.

Role of Quantum Machine Learning in Drug Discovery: Unfolded polypeptides have an enormous number of potential conformations. The exponential growth of potential conformations with chain length makes the search problem intractable for classical computers. So far there is theoretical and experimental evidence of advantage of solving such optimization problems using Quantum computing approaches like Quantum Annealing, VQE & QAOA..

Protein folding prediction using quantum computing techniques is an ongoing research field most of the approaches still are at, work in progress or published paper, stage and the use of quantum computing is very novel. Thus, there is need of the hour to investigate misfolded proteins & predict protein folds & shapes involving not fully understood processes & algorithms in quantum computing.

In this section we will briefly discuss various quantum machine learning algorithms which can possibly be applicable to study and predict molecular structures and hence the folding of a protein polymer chain.

VQE (Variational Quantum Eigensolver): is an algorithm that can be used to estimate energy levels of molecules, it is a variational algorithm in which a classical computer optimizes parameters of a quantum computation. In the setup of the problem, we take the known structure of a molecule and construct a Hamiltonian for the system. The Hamiltonian of a system provides a description of how that system evolves in time and we want to find its ground state. VQE consists of an optimization loop inside which, a state is prepared based on a set of parameters and then the energy expectation of the state is measured. A classical optimizer for example stochastic gradient descent can then be used to find better parameters of the quantum circuit and restart the loop until desired energy state is reached.
```
Quantum Annealing: is a metaheuristic technique for finding the global minima of a given objective function over a given set of candidate states, by a process using quantum fluctuations (in other words, a meta-procedure for finding a procedure that finds an absolute minimum size/length/cost/distance from within a possibly very large, but nonetheless finite set of possible solutions using quantum fluctuation-based computation instead of classical computation). Quantum annealing is used mainly for problems where the search space is discrete (combinatorial optimization problems) with many local minima such as finding the ground state of a spin glass. Thus Quantum annealing can be used to predict energy states of a protein or any other molecule.

QSVM (Quantum Support Vector Machine): Quantum Support Vector Machine (QSVM) is a quantum version of the Support Vector Machine (SVM) algorithm which uses quantum properties to perform calculations. QSVM uses the power of Quantum technology to improve performance of classical SVM algorithms that run on classical machines. The main benefit is quantum computing enables the use of kernels which are hard to compute classically. Finding an optimal boundary between the possible outputs after Kernel transformations that are used by SVM for classifying the dataset into various classes. Data that seems hard to separate in its original space by a simple hyperplane can be separated by applying a non-linear transformation function (known as feature map) in space known as feature space. The Inner product for each pair of data points in the set is computed to assess the similarity between them and this in turn is used to classify the data points in this new feature space (the higher the value of the inner product more similar they are to each other) and this collection of inner products is called the kernel. The ability to map high dimensionality in the Hilbert space using kernel methods could possibly be used to study molecular structures along with other quantum or classical approach.

QNN (Quantum Neural Networks): A quantum neural network (QNN) is a machine learning model or algorithm that combines concepts from quantum computing and artificial neural networks. The term has been used to describe a variety of ideas, ranging from quantum computers emulating the exact computations of neural nets, to general trainable quantum circuits that bear only little resemblance with the multi-layer perceptron structure. QNN could also be used along with QSVM leveraging the SVM kernels to map features into a Hamiltonian hence could be used to study various characteristics of molecules.

Image source Tensorflow Quantum

QGAN (Quantum Generative Adversarial Network): is a hybrid quantum-classical algorithm used for generative modelling tasks. This adaptive algorithm uses the interplay of a generative network and a discriminative network to learn the probability distribution underlying given training data. These networks are trained in alternating optimization steps, where the discriminator tries to differentiate between training data samples and data samples from the generator and the generator aims at generating samples which the discriminator classifies as training data samples. Eventually, the quantum generator learns the training data’s underlying probability distribution. The trained quantum generator loads a quantum state which is a model of the target distribution.

QRBM (Quantum Restricted Boltzman machine): Boltzmann machines, which are probabilistic graphical models that can be understood as stochastic recurrent neural networks, play an important role in the quantum machine learning literature. For example, it is suggested to use samples from a quantum computer to train classical Boltzmann machines, or to interpret spins as physical units of a quantum Boltzmann machine model.

```

In Part 2 in this series we will use Variational Quantum Eigensolver to study characteristics of a polypeptide protein molecule by simulating its structure on an actual quantum computer
