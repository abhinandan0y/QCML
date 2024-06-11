#### Quantum Computing in HealthCare — Protein Folding Part-2 (Comparison of Quantum & Classical Algorithms for Molecular Simulation)
Study of Protein molecule using VQE (Variational Quantum Eigensolver) in Pennylane Quantum Machine Learning library & Quantum Processor by Rigetti through Amazon Braket Quantum Computing Cloud Service.
Fractal AI Research
```
https://medium.com/@fractal.ai/quantum-computing-in-healthcare-protein-folding-part-2-comparison-of-quantum-classical-d717368629e6
```
code
```
The goal of this case study is to explore whether a small hypothetical anti-retroviral molecule folds and binds with a virus using hybrid classical-quantum chemistry modelling. Successful binding will be determined by a lower total ground-state energy for the molecules when they are close together (forming a single macromolecule). This work is inspired by tutorials from Pennylane & Qiskit. In case you missed please refer to Part 1 in this series for detailed introduction to Quantum computing in healthcare.

Antiretroviral treatment (also known as antiretroviral therapy or ART) are the drugs that can treat diseases like HIV. ART is not a cure, but keeps the disease conditions under control.
To design new anti-retrovirals it is important to perform chemical simulations to confirm that the anti-retroviral binds with the virus protein. Such simulations are very hard to simulate on classical CPUs even supercomputers. Quantum computers promise more accurate simulations allowing for a better drug-design workflow.
Anti-retrovirals are drugs that bind with a virus protein called protease. This protease divides virus proteins into smaller proteins thus multiplying the virus. The Anti-retrovirals will bind to this protease and prevent it from multiplying the virus protein. The anti-retroviral can be thought of as a sticky obstacle that disrupts its ability to multiply. Reference video explaining how anti-retroviral treatment works
Total ground-state energy refers to the sum of the energies concerning the arrangement of the electrons and the nuclei. The nuclear energy is easy to calculate classically. It is the energy of the electron distribution (i.e. molecular spin-orbital occupation) that is extremely difficult.
The ground-state of a molecule in some configuration consists of the locations of the nuclei, together with some distribution of electrons around the nuclei. The nucleus-nucleus, nuclei-electron and electron-electron forces/energy of attraction and repulsion are kept in a matrix called the Hamiltonian. Since the nuclei are relatively Big in size compared to electrons, they move at a slower rate than the electrons. This allows us to split the calculation into two parts, placing the nuclei and calculating the electron distribution, followed by moving the nuclei and recalculating the electron distribution until a minimum total energy distribution is reached.
Algorithm: Find total ground state energy place nuclei repeat until grid completed or no change in total energy then calculate electronic ground state total energy = (nuclei repulsion + electronic energy) move nuclei (either in grid or following gradient) return total energy
Here we will use PennyLane QML library on Amazon Braket Quantum Computing Cloud Service to find ground-state energy of a molecule by implementing the variational quantum eigensolver (VQE) algorithm using Pennylane’s qchem package.

There are two well-known approaches suitable for chemical simulations namely Quantum Phase Estimation (QPE) and Variational Quantum Eigensolver (VQE) . QPE requires fault-tolerant quantum computers that have not yet been built. The second is suitable for current day NISQ era quantum computers hence we will use VQE.

Braket comes with pennylane installed, to learn how to get started with amazon braket and other tutorials please refer to their examples repository. First Import the pennylane packages.

Define Molecular Geometry
To simplify the calculations, let us focus on O=C-N part of the molecule. We also add few hydrogen atoms to try and make the molecule as realistic as possible (HCONH₂, Formamide, is a stable molecule, which also is an ionic solvent, so it actually does “cut” ionic bonds). This peptide bond is one of the most important factors in determining the chemistry of proteins, including protein folding in general and the HIV protease’s replicating ability.

Making O=C-N toy protease molecule is an extreme simplification, but biologically motivated

Imagine that this molecule is responsible to cut the HIV master protein (Gag-Pol polyprotein) to make copies of the virus:


Image created using https://molview.org
Following are the 3D coordinates of the elements of the molecule

“O”: (1.1280, 0.2091, 0.0000)
“N”: (-1.1878, 0.1791, 0.0000)
“C”: (0.0598, -0.3882, 0.0000)
“H”: (-1.3085, 1.1864, 0.0001)
“H”: (-2.0305, -0.3861, -0.0001)
“H”: (-0.0014, -1.4883, -0.0001)

For simplicity and due to limited compute resources on the quantum devices we choose a single carbon atom to represent an anti-retroviral molecule.

We arbitrarily select a carbon atom with given coordinates in a line approaching the nitrogen atom. This “blocking” approach will try to obstruct the scissor from cutting. If it “sticks”, it’s working and successfully disrupts the duplication mechanism of the virus.


“C”: (-0.1805, 1.3955, 0.0000)

The input data is the geometry of this macro-molecule (O,N,C,H,H,H,C) the geometry data can be provided as following list of lists or could be directly read from a geometry xyz file.

The fixed location of each nucleus is specified as a python list, where each nucleus (as a list) contains a string corresponding to the atom name and its 3D co-ordinates (as another list).


Prepare the Hamiltonian
Once nuclei positions are fixed, the only part of the Hamiltonian that needs to be calculated on the quantum computer is the detailed electron-electron interactions.
The nuclei-electron and a rough mean field electron-electron interaction can be pre-computed as allowed molecular orbitals on a classical computer via the Hartree-Fock approximation.
The molecular orbital and overlap pre-calculation are provided by classical package PySCF connected to pennylane via a driver.
With these allowed molecular orbitals and their pre-calculated overlaps, Pennylane automatically produces an interacting electron-electron fermionic molecular-orbital Hamiltonian (called Second Quantization).
Defining an active space is an approximation to keep the simulation within the computational resources at hand. In principle, one should include all the electrons in the molecule.

For our molecule we thus have:

Total number of electrons Ne = (2*6C+7*1N+8*1O+3*1H) = 30
For a minimal basis set (calculated using sto-3g basis set) number of molecular orbitals required is M = 23
Using total number of molecular orbitals M we can determine the number of qubits as Nqubits=2*M
Therefore we would require 46 qubits to simulate the full molecular Hamiltonian. This is very hard to simulate due to the size and depth of the variational circuit, and the number of terms in the decomposed Hamiltonian.
A natural approximation of the required orbitals is to ignore the core orbitals (orbitals with deep energies that are typically not important for chemical bonding). In our case there are 4 core orbitals populated by 8 core electrons. So we would end up with a system of 22 electrons and 19 valence orbitals which still requires significant resources (38 qubits). In order to reduce further the size of our simulation we can decrease the number of active electrons and orbitals. Currently, Note that if we have a chemical intuition about which orbitals we want to include in the active space we can use the decompose function that allows us to choose the active orbitals. We also specify the overall charge, which tells pennylane to automatically calculate the number of needed electrons to produce that charge


Lets first try to estimate Classically the ground state energy for the Hamiltonian of our molecule, then later we will train the Quantum circuit to converge near to this ground state. Use 6 electrons because beyond 6 electrons classical approach goes out of memory and errors out


ground_state_energy: -193.19554924925973

Using the VQE algorithm we compute the energy of the molecule by measuring the expectation value of the Hamiltonian on a variational quantum circuit. We will train the circuit parameters so that the expectation value i.e. the energy of the Hamiltonian is minimized, thereby finding the ground state energy of the molecule.

Here we define the spin function for our molecule using jordan wigner transformation


Group observables to reduce circuit executions
Here we can see that this Hamiltonian is composed of the listed N number of individual observables that are tensor products of Pauli operators


Number of observables(N): 341

A straightforward approach to measuring the expectation value would be to implement the circuit N times, and each time measuring one of the Pauli terms that form part of the Hamiltonian. However we can build this more efficiently by combining and grouping the Pauli terms using PennyLane’s grouping module, these groups can be measured concurrently on a single circuit. Elements of each group are known as qubit-wise commuting observables. The Hamiltonian can be split into G number of groups as listed below:


Number of qubit-wise commuting groups(G): 100

This means that instead of executing N separate circuits, we just need to execute G number of circuits. This saving can become even more useful as the number of Pauli terms in the Hamiltonian increases while switching to a larger molecule which can exponentially increase both the number of qubits and the number of terms. Pennylane has in-built support for pre-grouping the observables in a Hamiltonian to minimize the number of device executions, saving both runtime and simulation time when using remote devices.

Here we can see the sample list of operations which will be done on the Hamiltonian


Measurement [‘Identity’] on wires <Wires = [0]>
Measurement [‘PauliZ’] on wires <Wires = [0]>
Measurement [‘PauliY’, ‘PauliZ’, ‘PauliY’] on wires <Wires = [0, 1, 2]>
Measurement [‘PauliX’, ‘PauliZ’, ‘PauliX’] on wires <Wires = [0, 1, 2]>
Measurement [‘PauliY’, ‘PauliZ’, ‘PauliZ’, ‘PauliZ’, ‘PauliY’] on wires <Wires = [0, 1, 2, 3, 4]>
Measurement [‘PauliX’, ‘PauliZ’, ‘PauliZ’, ‘PauliZ’, ‘PauliX’] on wires <Wires = [0, 1, 2, 3, 4]>
Measurement [‘PauliY’, ‘PauliZ’, ‘PauliZ’, ‘PauliZ’, ‘PauliZ’, ‘PauliZ’, ‘PauliY’] on wires <Wires = [0, 1, 2, 3, 4, 5, 6]>
Measurement [‘PauliX’, ‘PauliZ’, ‘PauliZ’, ‘PauliZ’, ‘PauliZ’, ‘PauliZ’, ‘PauliX’] on wires <Wires = [0, 1, 2, 3, 4, 5, 6]>………………………………

Define the Quantum processor
Define the designated Quantum device or simulator. Here we use amazon braket api to construct a device pointer to rigetti’s Aspen-8 quantum computer which has about 30 qubits. Note this will incur usage cost thus please check AWS pricing to estimate the cost. Alternatively we can also use pennylane or braket simulators without any cost. Replace mybucketID & my_folder_name with your corresponding AWS Bucket & Folder names.


Define the Quantum Circuit Here we use a chemistry inspired circuit called the Unitary Coupled-Cluster Singles and Doubles (UCCSD) ansatz. To use it we must define following additional parameters.


Here we define the quantum circuit using USSCD ansatz


Measuring the energy and total spin
The expectation values or the energy of this Hamiltonian and the total spin S² operator can be defined as:


Total spin S of the prepared state can be obtained from the expectation value as:


therefore lets define the function to calculate total spin as


Now Let’s initialize few random values to evaluate corresponding random energy and spin.


Since the parameters chosen are random, the measured energy is not the ground state energy and the prepared state is not an eigenstate. Therefore We will train the circuit to fine tune the parameters to arrive at the minimum energy of the Hamiltonian. The parameter tuning process is repeated until the energy stops decreasing.

Minimizing the energy
The energy can be minimized by choosing an optimizer, pennylane supports multiple optimizers here we choose GradientDescentOptimizer, it will classically find the gradient and send the updated parameters to the quantum circuit.


Now we train our circuit for the defined number of epochs, after many iterations we determined that the energy wasn’t reducing much more after 50 epochs.


epoch: 5 , Energy: -192.4479635723132 , Total Spin: 0.620895471273827
epoch: 10 , Energy: -192.86997943654941 , Total Spin: 0.46789539218260
epoch: 15 , Energy: -193.02045093733543 , Total Spin: 0.31651570392956
epoch: 20 , Energy: -193.0860096598843 , Total Spin: 0.21350563067516
epoch: 25 , Energy: -193.12102508511012 , Total Spin: 0.14620775705568
epoch: 30 , Energy: -193.14180581148196 , Total Spin: 0.101719666603168
epoch: 35 , Energy: -193.15494171701704 , Total Spin: 0.07186452465624
epoch: 40 , Energy: -193.16362838308729 , Total Spin: 0.0515593639401
epoch: 45 , Energy: -193.16958647581617 , Total Spin: 0.037572014681173
epoch: 50 , Energy: -193.17380474370404 , Total Spin: 0.0278099457412
— — — — — — — —
Optimized energy: -193.17380474370404 Hartfree
Total Spin: 0.0278099457412867

We can see that the optimized energy (measured in Hartfree unit) received is quite near to the energy we estimated classically above Let’s now plot how the energy and spin evolved during optimization



From the Plot we can see that after certain point both the energy & spin have stopped to reduce any further, In future when quantum processors become more powerful we can try the simulation with more electrons.

Results — Both classical & quantum algorithms calculated the ground state energy of the molecule to be approximately -193.19 Ha (Hartree), but as the number of electrons increased classical approach did not work as shown below:


Conclusion — Even in this small simulation we can see that the Quantum circuit was able to perform better than the classical computation. Imagine the scale of chemical simulations that will be possible when we will have quantum computers with thousands, millions of qubits.

In Part 3 in this series we will use Adiabatic quantum computing using D-wave’s Quantum Annealer to study & predict characteristics & folding of a protein

```
