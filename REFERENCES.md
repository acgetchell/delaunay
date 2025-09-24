---
title: "References and Citations"
description: "Academic references and bibliographic citations used throughout the delaunay library"
keywords: ["references", "citations", "computational geometry", "Delaunay triangulation", "bibliography"]
author: "Adam Getchell"
date: "2025-08-28"
category: "Documentation"
tags: ["academic", "research", "citations", "geometry"]
layout: "page"
---

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

## Core Delaunay Triangulation Algorithms and Data Structures

This section contains the foundational algorithms and data structures that form the basis of this library.

### Triangulation Construction Algorithms

- Bowyer, A. "Computing Dirichlet tessellations." *The Computer Journal* 24, no. 2 (1981): 162-166.
  DOI: [10.1093/comjnl/24.2.162](https://doi.org/10.1093/comjnl/24.2.162)
- Watson, D.F. "Computing the n-dimensional Delaunay tessellation with application to Voronoi polytopes."
  *The Computer Journal* 24, no. 2 (1981): 167-172. DOI: [10.1093/comjnl/24.2.167](https://doi.org/10.1093/comjnl/24.2.167)

### Data Structures and Implementation References

- The CGAL Project. "CGAL User and Reference Manual." CGAL Editorial Board, 6.0.1 edition, 2024.
  Available at: <https://doc.cgal.org/6.0.1/Manual/packages.html>
- de Berg, M., et al. "Computational Geometry: Algorithms and Applications." 3rd ed. Berlin: Springer-Verlag, 2008.
  DOI: [10.1007/978-3-540-77974-2](https://doi.org/10.1007/978-3-540-77974-2)

### Lifted Paraboloid Method

- Preparata, Franco P., and Michael Ian Shamos. "Computational Geometry: An Introduction."
  Texts and Monographs in Computer Science. New York: Springer-Verlag, 1985.
  DOI: [10.1007/978-1-4612-1098-6](https://doi.org/10.1007/978-1-4612-1098-6)
- Edelsbrunner, Herbert. "Algorithms in Combinatorial Geometry."
  EATCS Monographs on Theoretical Computer Science. Berlin: Springer-Verlag, 1987.
  DOI: [10.1007/978-3-642-61568-9](https://doi.org/10.1007/978-3-642-61568-9)

## Geometric Predicates and Numerical Robustness

These references ensure the library's geometric computations are mathematically sound and numerically stable.

### Robust Geometric Predicates

- Shewchuk, J. R. "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates."
  *Discrete & Computational Geometry* 18, no. 3 (1997): 305-363. DOI: [10.1007/PL00009321](https://doi.org/10.1007/PL00009321)
- Shewchuk, J. R. "Robust Adaptive Floating-Point Geometric Predicates."
  *Proceedings of the Twelfth Annual Symposium on Computational Geometry* (1996): 141-150.

### Simulation of Simplicity and Degeneracy Handling

- Edelsbrunner, H., and Mücke, E. P. "Simulation of Simplicity: A Technique to Cope with Degenerate Cases in Geometric Algorithms."
  *ACM Transactions on Graphics* 9, no. 1 (1990): 66-104. DOI: [10.1145/77635.77639](https://doi.org/10.1145/77635.77639)
- Seidel, R. "The Nature and Meaning of Perturbations in Geometric Computing."
  *Discrete & Computational Geometry* 19, no. 1 (1998): 1-17. DOI: [10.1007/PL00009336](https://doi.org/10.1007/PL00009336)

### Geometric Tie-Breaking and Deterministic Perturbations

- Burnikel, C., Funke, S., and Mehlhorn, K. "Exact Geometric Computation Made Easy."
  *Proceedings of the Fifteenth Annual Symposium on Computational Geometry* (1999): 341-350.
  DOI: [10.1145/304893.304988](https://doi.org/10.1145/304893.304988)
- Yap, C. K. "Towards Exact Geometric Computation."
  *Computational Geometry* 7, no. 1-2 (1997): 3-23. DOI: [10.1016/0925-7721(95)00040-2](https://doi.org/10.1016/0925-7721(95)00040-2)

### Circumcenter and Circumradius Calculations

- Lévy, Bruno, and Yang Liu. "Lp Centroidal Voronoi Tessellation and Its Applications."
  *ACM Transactions on Graphics* 29, no. 4 (July 26, 2010): 119:1-119:11.
  DOI: [10.1145/1778765.1778856](https://doi.org/10.1145/1778765.1778856)

## Convex Hull Algorithms

The library includes d-dimensional convex hull functionality, based on these algorithmic foundations.

### Convex Hull from Delaunay Triangulations

- Brown, K.Q. "Voronoi Diagrams from Convex Hulls." *Information Processing Letters* 9, no. 5 (1979): 223-228.
  DOI: [10.1016/0020-0190(79)90074-7](https://doi.org/10.1016/0020-0190(79)90074-7)
- Edelsbrunner, H. "Algorithms in Combinatorial Geometry."
  EATCS Monographs on Theoretical Computer Science. Berlin: Springer-Verlag, 1987.
  DOI: [10.1007/978-3-642-61568-9](https://doi.org/10.1007/978-3-642-61568-9)

### Point-in-Polytope Testing

- Preparata, F.P., and Shamos, M.I. "Computational Geometry: An Introduction."
  Texts and Monographs in Computer Science. New York: Springer-Verlag, 1985.
  DOI: [10.1007/978-1-4612-1098-6](https://doi.org/10.1007/978-1-4612-1098-6)
- O'Rourke, J. "Computational Geometry in C." 2nd ed. Cambridge: Cambridge University Press, 1998.
  DOI: [10.1017/CBO9780511804120](https://doi.org/10.1017/CBO9780511804120)

### Incremental Convex Hull Construction

- Clarkson, K.L., and Shor, P.W. "Applications of Random Sampling in Computational Geometry, II."
  *Discrete & Computational Geometry* 4, no. 1 (1989): 387-421. DOI: [10.1007/BF02187740](https://doi.org/10.1007/BF02187740)
- Barber, C.B., Dobkin, D.P., and Huhdanpaa, H. "The Quickhull Algorithm for Convex Hulls."
  *ACM Transactions on Mathematical Software* 22, no. 4 (1996): 469-483. DOI: [10.1145/235815.235821](https://doi.org/10.1145/235815.235821)

## Combinatorial Algorithms and Enumeration

These references provide algorithmic foundations for systematic enumeration and combinatorial generation used in the library.

### Grid Generation and Mixed-Radix Counters

- Knuth, D. E. "The Art of Computer Programming, Vol. 4A: Combinatorial Algorithms." Boston: Addison-Wesley, 2011.
  ISBN: 978-0-201-03804-0
- Nijenhuis, A., and Wilf, H. S. "Combinatorial Algorithms for Computers and Calculators." 2nd ed. New York: Academic Press, 1978.
  ISBN: 978-0-12-519260-6

## Advanced Computational Geometry Topics

These references support specialized features and high-dimensional computations in the library.

### Volume Computation and Gram Matrix Methods

- Coxeter, H.S.M. "Introduction to Geometry" (2nd ed., 1969), Chapter 13
- Richter-Gebert, Jürgen. "Perspectives on Projective Geometry" (2011), Section 14.3
- Edelsbrunner, Herbert. "Geometry and Topology for Mesh Generation" (2001), Chapter 2

### High-Dimensional Computational Geometry

- Chazelle, B. "An Optimal Convex Hull Algorithm in Any Fixed Dimension."
  *Discrete & Computational Geometry* 10, no. 4 (1993): 377-409. DOI: [10.1007/BF02573985](https://doi.org/10.1007/BF02573985)
- Seidel, R. "The Upper Bound Theorem for Polytopes: An Easy Proof of Its Asymptotic Version."
  *Computational Geometry* 5, no. 2 (1995): 115-116. DOI: [10.1016/0925-7721(95)00013-Y](https://doi.org/10.1016/0925-7721(95)00013-Y)
