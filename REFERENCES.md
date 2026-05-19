# References and Citations

## AI-Assisted Development Tools

- **ChatGPT**: OpenAI. <https://openai.com/chatgpt>.
- **Claude**: Anthropic. <https://www.anthropic.com/claude>.
- **CodeRabbit**: CodeRabbit AI, Inc. <https://coderabbit.ai/>.
- **Codex**: OpenAI. <https://openai.com/codex/>.
- **GitHub Copilot**: GitHub. <https://github.com/features/copilot>.
- **KiloCode AI Engineering Assistant**: KiloCode. <https://kilocode.ai/>.
- **WARP**: Warp Dev, Inc. <https://www.warp.dev/>.

All AI-generated output was reviewed and/or edited by the maintainer.
No generated content was used without human oversight.

## How to Cite This Library

If you use this library in your research or project, please cite it using the information provided in our
[CITATION.cff](CITATION.cff) file. This file contains structured citation metadata that can be automatically
processed by GitHub and other platforms.

**Quick citation:**

```text
Adam Getchell. 2025. delaunay: A d-dimensional Delaunay triangulation library.
Zenodo. DOI: https://doi.org/10.5281/zenodo.16931097
```

For BibTeX, APA, or other citation formats, please refer to the [CITATION.cff](CITATION.cff) file or use
GitHub's "Cite this repository" feature.

## Inline Citation Keys

Some Rust API docs use compact numbered citations. The labels below point to
the corresponding bibliography entries in this file:

- [1] Shewchuk, J. R. (1997), adaptive precision floating-point arithmetic and
  robust geometric predicates.
- [2] Bowyer, A. (1981), incremental Dirichlet/Delaunay tessellation.
- [3] Watson, D. F. (1981), n-dimensional Delaunay tessellation by cavity
  replacement.
- [4] Edelsbrunner, H., and Shah, N. R. (1996), topological flipping for
  regular triangulations.
- [5] Edelsbrunner, H. (2001), mesh generation, cavity-based construction, and
  triangulation repair background.

## Advanced Computational Geometry Topics

These references support specialized features and high-dimensional computations in the library.

### High-Dimensional Computational Geometry

- Avis, D., and Bremner, D. "How Good Are Convex Hull Algorithms?" *Computational Geometry* 7,
  no. 5-6 (1997): 265-301. DOI: [10.1016/S0925-7721(96)00023-5](https://doi.org/10.1016/S0925-7721(96)00023-5)
- Chazelle, B. "An Optimal Convex Hull Algorithm in Any Fixed Dimension."
  *Discrete & Computational Geometry* 10, no. 4 (1993): 377-409. DOI: [10.1007/BF02573985](https://doi.org/10.1007/BF02573985)
- Mulmuley, K. "Computational Geometry: An Introduction Through Randomized Algorithms."
  New Jersey: Prentice Hall, 1994. ISBN: 978-0-13-336363-6
- Seidel, R. "The Upper Bound Theorem for Polytopes: An Easy Proof of Its Asymptotic Version."
  *Computational Geometry* 5, no. 2 (1995): 115-116. DOI: [10.1016/0925-7721(95)00013-Y](https://doi.org/10.1016/0925-7721(95)00013-Y)

### Volume Computation and Gram Matrix Methods

- Coxeter, H.S.M. *Introduction to Geometry*. 2nd ed., 1969. (Chapter 13.)
- Richter-Gebert, Jürgen. *Perspectives on Projective Geometry*. 2011. (Section 14.3.)
- See Edelsbrunner, H. *Geometry and Topology for Mesh Generation* (2001) in
  [Data Structures and Implementation References](#data-structures-and-implementation-references) for a
  Gram-matrix treatment (Chapter 2).

## Combinatorial Algorithms and Enumeration

These references provide algorithmic foundations for systematic enumeration and combinatorial generation used in the library.

### Cube Triangulation (Freudenthal / Kuhn)

These references describe the canonical triangulation of the n-cube into n! simplices via coordinate orderings,
commonly known as the Freudenthal or Kuhn triangulation. This construction underlies grid-to-simplex
decompositions used in topology, numerical methods, and periodic triangulations (e.g., T³).

- Freudenthal, H. "Simplizialzerlegungen von beschränkter Flachheit."
  *Annals of Mathematics* 43, no. 3 (1942): 580–582.
- Hatcher, A. *Algebraic Topology*. Cambridge University Press, 2002.
  (See discussions of simplicial decompositions and product spaces.)
- Munkres, J. R. *Elements of Algebraic Topology*. Addison–Wesley, 1984.
  (Simplicial complexes and triangulations of product spaces.)

### Grid Generation and Mixed-Radix Counters

- Knuth, D. E. "The Art of Computer Programming, Vol. 4A: Combinatorial Algorithms." Boston: Addison-Wesley, 2011.
  ISBN: 978-0-201-03804-0
- Nijenhuis, A., and Wilf, H. S. "Combinatorial Algorithms for Computers and Calculators." 2nd ed. New York: Academic Press, 1978.
  ISBN: 978-0-12-519260-6

## Convex Hull Algorithms

The library includes d-dimensional convex hull functionality, based on these algorithmic foundations.

Some foundational background (lifted paraboloid / Delaunay–convex hull duality and randomized incremental methods)
is listed under [Lifted Paraboloid Method](#lifted-paraboloid-method) and
[Triangulation Construction Algorithms](#triangulation-construction-algorithms).
High-dimensional convex hull bounds are covered under [High-Dimensional Computational Geometry](#high-dimensional-computational-geometry).

### Convex Hull from Delaunay Triangulations

- Brown, K.Q. "Voronoi Diagrams from Convex Hulls." *Information Processing Letters* 9, no. 5 (1979): 223-228.
  DOI: [10.1016/0020-0190(79)90074-7](https://doi.org/10.1016/0020-0190(79)90074-7)

### Incremental Convex Hull Construction

- Barber, C.B., Dobkin, D.P., and Huhdanpaa, H. "The Quickhull Algorithm for Convex Hulls."
  *ACM Transactions on Mathematical Software* 22, no. 4 (1996): 469-483. DOI: [10.1145/235815.235821](https://doi.org/10.1145/235815.235821)

### Point-in-Polytope Testing

- O'Rourke, J. "Computational Geometry in C." 2nd ed. Cambridge: Cambridge University Press, 1998.
  DOI: [10.1017/CBO9780511804120](https://doi.org/10.1017/CBO9780511804120)

## Core Delaunay Triangulation Algorithms and Data Structures

This section contains the foundational algorithms and data structures that form the basis of this library.

### Bistellar (Pachner) Moves and Delaunay Repair

- Edelsbrunner, H., and Shah, N. R. "Incremental Topological Flipping Works for Regular Triangulations."
  *Algorithmica* 15, no. 3 (1996): 223-241. DOI: [10.1007/BF01975867](https://doi.org/10.1007/BF01975867)
- Joe, B. "Construction of Three-Dimensional Delaunay Triangulations Using Local Transformations."
  *Computer Aided Geometric Design* 8, no. 2 (1991): 123-142.
  DOI: [10.1016/0167-8396(91)90038-D](https://doi.org/10.1016/0167-8396(91)90038-D)
- Lawson, C. L. "Software for C1 Surface Interpolation."
  In J. R. Rice (ed.), *Mathematical Software III*, Academic Press, 1977, pp. 161-194.
- Pachner, U. "PL Homeomorphic Manifolds Are Equivalent by Elementary Shellings."
  *European Journal of Combinatorics* 12, no. 2 (1991): 129-145.
  DOI: [10.1016/S0195-6698(13)80080-7](https://doi.org/10.1016/S0195-6698(13)80080-7)

### Data Structures and Implementation References

- Blandford, D.K., Blelloch, G.E., and Kadow, C. "Engineering a Compact Parallel Delaunay Algorithm in 3D."
  *Proceedings of the Twenty-Second Annual Symposium on Computational Geometry* (2006): 292-300.
  DOI: [10.1145/1137856.1137895](https://doi.org/10.1145/1137856.1137895)
- de Berg, M., et al. "Computational Geometry: Algorithms and Applications." 3rd ed. Berlin: Springer-Verlag, 2008.
  DOI: [10.1007/978-3-540-77974-2](https://doi.org/10.1007/978-3-540-77974-2)
- Devillers, O. "The Delaunay Hierarchy." *International Journal of Foundations of Computer Science* 13,
  no. 2 (2002): 163-180. DOI: [10.1142/S0129054102001035](https://doi.org/10.1142/S0129054102001035)
- Edelsbrunner, H. "Geometry and Topology for Mesh Generation." Cambridge: Cambridge University Press, 2001.
  DOI: [10.1017/CBO9780511530067](https://doi.org/10.1017/CBO9780511530067)
  ISBN: 978-0-521-79309-4
- Getchell, Adam. "la-stack: Fast, stack-allocated linear algebra for fixed dimensions in Rust." Zenodo, 2026.
  DOI: [10.5281/zenodo.18444315](https://doi.org/10.5281/zenodo.18444315)
- The CGAL Project. "CGAL User and Reference Manual." CGAL Editorial Board, 6.0.1 edition, 2024.
  Available at: <https://doc.cgal.org/6.0.1/Manual/packages.html>
  - Triangulation_3 reference: <https://doc.cgal.org/6.0.1/Triangulation_3/index.html>

### Foundational Delaunay and Voronoi Theory

- Delaunay, B. "Sur la sphère vide. A la mémoire de Georges Voronoï."
  *Bulletin de l'Académie des Sciences de l'URSS. Classe des Sciences Mathématiques et Naturelles*
  1934, no. 6 (1934): 793-800.
- Voronoi, G. F. "Nouvelles applications des paramètres continus à la théorie des formes quadratiques."
  *Journal für die reine und angewandte Mathematik* 133 (1908): 97-178; 134 (1908): 198-287.
- Aurenhammer, F., Klein, R., and Lee, D.-T. *Voronoi Diagrams and Delaunay Triangulations*.
  World Scientific, 2013. ISBN: 978-981-4447-62-1

### Lifted Paraboloid Method

- Edelsbrunner, Herbert. "Algorithms in Combinatorial Geometry."
  EATCS Monographs on Theoretical Computer Science. Berlin: Springer-Verlag, 1987.
  DOI: [10.1007/978-3-642-61568-9](https://doi.org/10.1007/978-3-642-61568-9)
- Preparata, Franco P., and Michael Ian Shamos. "Computational Geometry: An Introduction."
  Texts and Monographs in Computer Science. New York: Springer-Verlag, 1985.
  DOI: [10.1007/978-1-4612-1098-6](https://doi.org/10.1007/978-1-4612-1098-6)

### Periodic and Toroidal Triangulations

- Caroli, M., and Teillaud, M. "Computing 3D Periodic Triangulations."
  In *Algorithms - ESA 2009*, Lecture Notes in Computer Science 5757,
  Springer, 2009, pp. 37-48. Full version: INRIA Research Report RR-6823,
  <https://inria.hal.science/inria-00356871>
- The CGAL Project. "3D Periodic Triangulations." *CGAL User and Reference Manual*.
  Available at: <https://doc.cgal.org/latest/Periodic_3_triangulation_3/>

### Point Location in Triangulations

- Devillers, O., Pion, S., and Teillaud, M. "Walking in a Triangulation."
  *International Journal of Foundations of Computer Science* 13, no. 2 (2002): 181-199.
  DOI: [10.1142/S0129054102001047](https://doi.org/10.1142/S0129054102001047)
- Mücke, E.P., Saias, I., and Zhu, B. "Fast Randomized Point Location Without Preprocessing in Two- and Three-Dimensional Delaunay Triangulations."
  *Proceedings of the Twelfth Annual Symposium on Computational Geometry* (1996): 274-283.
  DOI: [10.1145/237218.237393](https://doi.org/10.1145/237218.237393)

### Scoped Local Validation and Flips

1. Edelsbrunner, H., and Shah, N. R. "Incremental Topological Flipping Works for Regular Triangulations."
   *Algorithmica* 15, no. 3 (1996): 223-241. DOI: [10.1007/BF01975867](https://doi.org/10.1007/BF01975867)
   Local flip neighborhoods support scoped validation when the mutation frontier is complete; global
   PL vertex-link obligations still require full-link validation.

### Spatial Ordering and Hilbert Curves

- Moon, B., Jagadish, H. V., Faloutsos, C., and Saltz, J. H.
  "Analysis of the Clustering Properties of the Hilbert Space-Filling Curve."
  *IEEE Transactions on Knowledge and Data Engineering* 13, no. 1 (2001): 124-141.
  DOI: [10.1109/69.908985](https://doi.org/10.1109/69.908985)
- Skilling, J. "Programming the Hilbert Curve." *AIP Conference Proceedings*
  707 (2004): 381-387. DOI: [10.1063/1.1751381](https://doi.org/10.1063/1.1751381)

### Triangulation Construction Algorithms

- Bowyer, A. "Computing Dirichlet tessellations." *The Computer Journal* 24, no. 2 (1981): 162-166.
  DOI: [10.1093/comjnl/24.2.162](https://doi.org/10.1093/comjnl/24.2.162)
- Clarkson, K.L., and Shor, P.W. "Applications of Random Sampling in Computational Geometry, II."
  *Discrete & Computational Geometry* 4, no. 1 (1989): 387-421. DOI: [10.1007/BF02187740](https://doi.org/10.1007/BF02187740)
- Fortune, S. "A Sweepline Algorithm for Voronoi Diagrams."
  *Algorithmica* 2, no. 1-4 (1987): 153-174. DOI: [10.1007/BF01840357](https://doi.org/10.1007/BF01840357)
- Guibas, L., Knuth, D., and Sharir, M. "Randomized Incremental Construction of Delaunay and Voronoi Diagrams."
  *Algorithmica* 7, no. 1-6 (1992): 381-413. DOI: [10.1007/BF01758770](https://doi.org/10.1007/BF01758770)
- Guibas, L., and Stolfi, J. "Primitives for the Manipulation of General Subdivisions and the Computation of Voronoi Diagrams."
  *ACM Transactions on Graphics* 4, no. 2 (1985): 74-123. DOI: [10.1145/282918.282923](https://doi.org/10.1145/282918.282923)
- Shewchuk, J.R. "Delaunay Refinement Algorithms for Triangular Mesh Generation."
  *Computational Geometry* 22, no. 1-3 (2002): 21-74. DOI: [10.1016/S0925-7721(01)00047-5](https://doi.org/10.1016/S0925-7721(01)00047-5)
- Watson, D.F. "Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes."
  *The Computer Journal* 24, no. 2 (1981): 167-172. DOI: [10.1093/comjnl/24.2.167](https://doi.org/10.1093/comjnl/24.2.167)

## Geometric Predicates and Numerical Robustness

These references ensure the library's geometric computations are mathematically sound and numerically stable.

### Circumcenter and Circumradius Calculations

- Lévy, Bruno, and Yang Liu. "Lp Centroidal Voronoi Tessellation and Its Applications."
  *ACM Transactions on Graphics* 29, no. 4 (July 26, 2010): 119:1-119:11.
  DOI: [10.1145/1778765.1778856](https://doi.org/10.1145/1778765.1778856)

### Exact Determinant Sign Computation

- Bareiss, E. H. "Sylvester's Identity and Multistep Integer-Preserving Gaussian Elimination."
  *Mathematics of Computation* 22, no. 103 (1968): 565-578.
  DOI: [10.1090/S0025-5718-1968-0226829-0](https://doi.org/10.1090/S0025-5718-1968-0226829-0)

  The Bareiss fraction-free elimination algorithm is used by `la_stack::Matrix::det_sign_exact`
  to compute provably correct determinant signs in exact `BigRational` arithmetic.

### Geometric Tie-Breaking and Deterministic Perturbations

- Burnikel, C., Funke, S., and Mehlhorn, K. "Exact Geometric Computation Made Easy."
  *Proceedings of the Fifteenth Annual Symposium on Computational Geometry* (1999): 341-350.
  DOI: [10.1145/304893.304988](https://doi.org/10.1145/304893.304988)
- Yap, C. K. "Towards Exact Geometric Computation."
  *Computational Geometry* 7, no. 1-2 (1997): 3-23. DOI: [10.1016/0925-7721(95)00040-2](https://doi.org/10.1016/0925-7721(95)00040-2)

### Robust Geometric Predicates

- Fortune, S., and Van Wyk, C.J. "Efficient Exact Arithmetic for Computational Geometry."
  *Proceedings of the Ninth Annual Symposium on Computational Geometry* (1993): 163-172.
  DOI: [10.1145/160985.161140](https://doi.org/10.1145/160985.161140)
- Fortune, S., and Van Wyk, C. J. "Static Analysis Yields Efficient Exact Integer Arithmetic for Computational Geometry."
  *ACM Transactions on Graphics* 15, no. 3 (1996): 223-248. DOI: [10.1145/234535.234691](https://doi.org/10.1145/234535.234691)
- Karasick, M., Lieber, D., and Nackman, L. R. "Efficient Delaunay Triangulation Using Rational Arithmetic."
  *ACM Transactions on Graphics* 10, no. 1 (1991): 71-91. DOI: [10.1145/99902.99917](https://doi.org/10.1145/99902.99917)
- Shewchuk, J. R. "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates."
  *Discrete & Computational Geometry* 18, no. 3 (1997): 305-363. DOI: [10.1007/PL00009321](https://doi.org/10.1007/PL00009321)
- Shewchuk, J. R. "Robust Adaptive Floating-Point Geometric Predicates."
  *Proceedings of the Twelfth Annual Symposium on Computational Geometry* (1996): 141-150.
  DOI: [10.1145/237218.237337](https://doi.org/10.1145/237218.237337)

### Simulation of Simplicity and Degeneracy Handling

- Edelsbrunner, H., and Mücke, E. P. "Simulation of Simplicity: A Technique to Cope with Degenerate Cases in Geometric Algorithms."
  *ACM Transactions on Graphics* 9, no. 1 (1990): 66-104. DOI: [10.1145/77635.77639](https://doi.org/10.1145/77635.77639)
- Seidel, R. "The Nature and Meaning of Perturbations in Geometric Computing."
  *Discrete & Computational Geometry* 19, no. 1 (1998): 1-17. DOI: [10.1007/PL00009336](https://doi.org/10.1007/PL00009336)

## Mesh Quality Metrics

These references inform the geometric quality measures used for evaluating simplex quality
and selecting high-quality simplices during triangulation repair.

### Quality Measures for Simplicial Meshes

- Field, D.A. "Qualitative Measures for Initial Meshes."
  *International Journal for Numerical Methods in Engineering* 47, no. 4 (2000): 887-906.
  DOI: [10.1002/(SICI)1097-0207(20000210)47:4<887::AID-NME804>3.0.CO;2-H](https://doi.org/10.1002/(SICI)1097-0207(20000210)47:4<887::AID-NME804>3.0.CO;2-H)
- Liu, A., and Joe, B. "Relationship between Tetrahedron Shape Measures."
  *BIT Numerical Mathematics* 34, no. 2 (1994): 268-287. DOI: [10.1007/BF01955874](https://doi.org/10.1007/BF01955874)
- Shewchuk, J.R. "What Is a Good Linear Element? Interpolation, Conditioning, Anisotropy,
  and Quality Measures." Eleventh International Meshing Roundtable, Ithaca, New York (2002).
  Available at: <https://people.eecs.berkeley.edu/~jrs/papers/elemj.pdf>

### Radius Ratio and Aspect Ratio Metrics

- Knupp, P.M. "Algebraic Mesh Quality Metrics."
  *SIAM Journal on Scientific Computing* 23, no. 1 (2001): 193-218.
  DOI: [10.1137/S1064827500371499](https://doi.org/10.1137/S1064827500371499)
- Parthasarathy, V.N., Graichen, C.M., and Hathaway, A.F. "A Comparison of Tetrahedron Quality Measures."
  *Finite Elements in Analysis and Design* 15, no. 3 (1994): 255-261.
  DOI: [10.1016/0168-874X(94)90033-7](https://doi.org/10.1016/0168-874X(94)90033-7)

## Performance Optimization and Memory Management

These references inform the library's performance optimization strategies and memory-efficient data structure design.

### Cache-Efficient Algorithms

- Arge, L., Goodrich, M.T., Nelson, M., and Sitchinava, N. "Fundamental Parallel Algorithms for Private-Cache Chip Multiprocessors."
  *Proceedings of the Twentieth Annual Symposium on Parallelism in Algorithms and Architectures* (2008): 197-206.
  DOI: [10.1145/1378533.1378573](https://doi.org/10.1145/1378533.1378573)

### Memory-Efficient Data Structures

- Blandford, D.K., Blelloch, G.E., Dahle, C., and Karp, R. "Compact Representations of Simplicial Meshes in Two and Three Dimensions."
  *International Journal of Computational Geometry & Applications* 15, no. 1 (2005): 3-24.
  DOI: [10.1142/S0218195905001580](https://doi.org/10.1142/S0218195905001580)
- Geuzaine, C., and Remacle, J.-F. "Gmsh: A 3-D Finite Element Mesh Generator
  with Built-In Pre- and Post-Processing Facilities." *International Journal for Numerical Methods in Engineering*
  79, no. 11 (2009): 1309-1331. DOI: [10.1002/nme.2579](https://doi.org/10.1002/nme.2579)
- Shewchuk, J.R. "Triangle: Engineering a 2D Quality Mesh Generator and Delaunay Triangulator."
  *Applied Computational Geometry Towards Geometric Engineering* (1996): 203-222.
  DOI: [10.1007/BFb0014497](https://doi.org/10.1007/BFb0014497)
- Si, H. "TetGen, a Delaunay-Based Quality Tetrahedral Mesh Generator."
  *ACM Transactions on Mathematical Software* 41, no. 2 (2015): Article 11.
  DOI: [10.1145/2629697](https://doi.org/10.1145/2629697)

## Research Software, Citation, and Reproducibility

These references support the library's treatment as a reusable, tested, benchmarked,
and citable research software artifact.

- Smith, A. M., Katz, D. S., Niemeyer, K. E., and the FORCE11 Software Citation Working Group.
  "Software Citation Principles." *PeerJ Computer Science* 2 (2016): e86.
  DOI: [10.7717/peerj-cs.86](https://doi.org/10.7717/peerj-cs.86)
- Wilson, G., Aruliah, D. A., Brown, C. T., Chue Hong, N. P., Davis, M.,
  Guy, R. T., Haddock, S. H. D., et al. "Best Practices for Scientific Computing."
  *PLOS Biology* 12, no. 1 (2014): e1001745.
  DOI: [10.1371/journal.pbio.1001745](https://doi.org/10.1371/journal.pbio.1001745)
- Wilson, G., Bryan, J., Cranston, K., Kitzes, J., Nederbragt, L., and Teal, T. K.
  "Good Enough Practices in Scientific Computing." *PLOS Computational Biology*
  13, no. 6 (2017): e1005510.
  DOI: [10.1371/journal.pcbi.1005510](https://doi.org/10.1371/journal.pcbi.1005510)

## Set Similarity Metrics (Testing and Validation)

These references motivate the Jaccard similarity/distance used in tests to compare
set-based structures (e.g., edge sets) under different construction orders.

- Jaccard, P. "Étude comparative de la distribution florale dans une portion des Alpes et des Jura."
  *Bulletin de la Société Vaudoise des Sciences Naturelles* 37, no. 142 (1901): 547–579.
- Tanimoto, T. T. "An Elementary Mathematical Theory of Classification and Prediction."
  IBM Technical Report, 1958. Often cited for the Tanimoto coefficient (equivalent to the
  Jaccard index for binary vectors).

## Topological Manifolds and PL Topology (Level 3 Validation)

These references support the topology-only manifold / PL-manifold validation logic in this
library (facet degree, closed-boundary checks, and links of simplices).

- Edelsbrunner, H., and Harer, J. *Computational Topology*. AMS, 2010.
- Hatcher, A. *Algebraic Topology*. Cambridge University Press, 2002.
  (Appendix A: PL Manifolds and Links.)
- Munkres, J. R. *Elements of Algebraic Topology*. Addison–Wesley, 1984.
  (Chapter 9: Simplicial Manifolds and Links.)
- Rourke, C. P., and Sanderson, B. J. *Introduction to Piecewise-Linear Topology*. Springer, 1972.
- Stillwell, J. *Euler's Gem: The Polyhedron Formula and the Birth of Topology*. Princeton University Press, 2010.
- Zomorodian, A. *Topology for Computing*. Cambridge University Press, 2005.
