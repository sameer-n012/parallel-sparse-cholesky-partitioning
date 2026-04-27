# Partitioning Strategies for Parallel Sparse Cholesky Factorization

**Sameer Narendran**  
University of Illinois Urbana-Champaign  
sameern3@illinois.edu  

---

## 1. Introduction and Motivation

Sparse symmetric positive definite (SPD) linear systems are fundamental in scientific computing and are commonly solved using Cholesky factorization. In parallel environments, performance depends heavily on the sparsity structure of the matrix and the induced elimination tree. Ordering and partitioning strategies determine fill-in, tree height, frontal matrix size, and communication cost. This project proposes an empirical evaluation of ordering strategies—including natural ordering, approximate minimum degree (AMD), and nested dissection—and their impact on parallel performance. Using CHOLMOD from SuiteSparse and graph partitioning tools such as METIS, we analyze elimination tree geometry, graph coloring–based concurrency, treewidth proxies, and observed runtime on modern CPU and GPU hardware.

Sparse SPD systems often arise in finite element discretizations, graph Laplacians, Gaussian Markov random fields, and large-scale optimization. For a sparse SPD matrix $A$, Cholesky factorization $A = LL^\top$ corresponds to eliminating vertices in its adjacency graph $G = (V, E)$. Eliminating a vertex forms a clique among its neighbors, producing fill edges.

The elimination process induces an elimination tree whose structure encodes column dependencies. This tree governs potential parallelism in the factorization process. Tree height bounds the critical path length and total depth of the process. Level width determines the available concurrency, and the frontal matrix size reflects the separator size and treewidth. Therefore, ordering strategies simultaneously influence total fill-in, arithmetic work, memory footprint, parallel scalability, and communication volume.

Understanding the structural-to-performance mapping on modern heterogeneous hardware motivates this study.

---

## 2. Related Work

Sparse Cholesky factorization has long had deep connections to graph theory. The fill pattern produced by elimination corresponds to a chordal embedding, and the elimination tree formalizes data dependencies (Liu, 1990). As a result, the size of the largest clique in the chordal completion equals the treewidth plus one, directly determining asymptotic work and memory.

Minimum degree orderings aim to greedily reduce fill by eliminating vertices of smallest degree in the evolving graph. Liu (1984) introduced multiple elimination refinements that improved practical performance. Approximate Minimum Degree (AMD) provides scalable implementations and remains widely used in practice.

Nested dissection, introduced by George (1973), takes a global approach by recursively partitioning the graph using vertex separators. For planar and grid-like graphs arising from PDE discretizations, nested dissection achieves asymptotically optimal complexity: $O(n^{3/2})$ in 2D and $O(n^2)$ in 3D. Crucially, nested dissection also produces balanced elimination trees with logarithmic height, exposing substantial parallelism.

Heath, Ng, and Peyton (1991) surveyed early parallel sparse direct solvers and highlighted the elimination tree as the central abstraction for concurrency. Later multifrontal and supernodal solvers exploited tree parallelism at multiple granularities: subtree parallelism, node-level dense linear algebra parallelism, and assembly-tree scheduling.

Modern sparse direct solvers increasingly rely on supernodal and multifrontal formulations to attain high performance on contemporary architectures. In particular, CHOLMOD (Davis and Hager, 2008) employs a supernodal Cholesky factorization in which columns with similar sparsity structure are grouped into supernodes. This aggregation transforms many sparse operations into dense matrix kernels, enabling the use of Level-3 BLAS routines. Because BLAS-3 operations achieve significantly higher arithmetic intensity than sparse scalar updates, overall performance becomes closely tied to the size and shape of the resulting dense frontal matrices.

Graph partitioning tools such as METIS (Karypis and Kumar, 1998) provide practical mechanisms for computing high-quality nested dissection orderings at scale. METIS uses a multilevel paradigm: the graph is successively coarsened into smaller graphs, a partition is computed at the coarsest level, and this partition is then projected back and refined during uncoarsening. This approach efficiently approximates balanced vertex separators while controlling edge cuts and separator size.

---

## 3. Proposed Work

Recent theoretical work connects sparse factorization complexity to communication lower bounds: separator size dictates both arithmetic intensity and minimal data movement in distributed-memory settings. In GPU environments, concurrency is further constrained by irregular memory access and limited fine-grained synchronization, making elimination tree width and frontal matrix shape particularly important.

While prior literature establishes asymptotic fill and complexity bounds, fewer empirical studies systematically correlate elimination tree height, measured parallel speedup, level width, and core utilization. This project aims to experimentally evaluate these relationships using contemporary hardware and solver implementations.

### 3.1 Ordering Strategies

We evaluate the following ordering strategies:

- Natural ordering  
- Minimum Degree  
- Approximate Minimum Degree (AMD)  
- Nested Dissection via METIS  
- Hybrid ND followed by constrained AMD refinement  

Strategies like Minimum Degree and AMD utilize greedy approaches, while Nested Dissection is a global separator-based approach.

### 3.2 Structural Metrics

For each matrix and ordering, we also compute:

- Elimination tree height  
- Level width distribution  
- Total fill-in  
- Largest frontal matrix size  
- Symbolic factorization statistics  

Largest frontal size serves as a practical proxy for treewidth and separator magnitude. Level decomposition approximates available parallelism.

### 3.3 Experimental Setup

Benchmarks will be drawn from the SuiteSparse Matrix Collection, focusing on large SPD systems from PDE and graph applications. Experiments will be conducted on a multi-core CPU and NVIDIA A100 and H200 GPUs.

Using CHOLMOD’s supernodal implementation, we will measure the factorization time, total runtime, memory consumption, parallel speedup, and communication cost. We can then analyze correlations between structural metrics and observed scalability, testing whether separator-balanced orderings consistently yield improved heterogeneous performance.

---

## References

1. M. T. Heath, E. G. Ng, and B. W. Peyton. *Parallel algorithms for sparse linear systems.* SIAM Review, 33(3):420–460, 1991.  
2. J. W. H. Liu. *Modification of the minimum-degree algorithm by multiple elimination.* ACM Transactions on Mathematical Software, 11(2):141–153, 1984.  
3. J. W. H. Liu. *The role of elimination trees in sparse factorization.* SIAM Journal on Matrix Analysis and Applications, 11(1):134–172, 1990.  
4. A. George. *Nested dissection of a regular finite element mesh.* SIAM Journal on Numerical Analysis, 10(2):345–363, 1973.  
5. G. Karypis and V. Kumar. *A fast and high quality multilevel scheme for partitioning irregular graphs.* SIAM Journal on Scientific Computing, 20(1):359–392, 1998.  
6. T. A. Davis and W. W. Hager. *Algorithm 887: CHOLMOD, supernodal sparse Cholesky factorization and update/downdate.* ACM Transactions on Mathematical Software, 35(3), 2008.