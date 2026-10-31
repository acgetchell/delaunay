//! Exact revised-simplex support for realized-simplex intersection.
//!
//! This private geometry module owns the provisional floating-point basis search,
//! exact Phase I/II certification, artificial-variable handling, dual certificates,
//! and matrix helpers used by Level 4 realization validation.

#![forbid(unsafe_code)]

use core::{array::from_fn, ops::ControlFlow};

use la_stack::{BigInt, BigRational, FromPrimitive, Signed, column_basis, rational_dot};

use crate::geometry::matrix::{
    ExactF64Conversion, Interval, LaError, LaVector, Matrix, RationalMatrix, RationalVector,
    StackMatrixDispatchError, Tolerance, gram_matrix, solve_exact_runtime_system,
};
use crate::geometry::realization::{
    LabeledSimplexRealization, SimplexIntersectionWitness, SimplexRealizationBuffer,
};

/// Complete LP support for standalone simplex intersections through D=8.
const MAX_LP_BASIS_DIMENSION: usize = 10;
/// Complete active-set enumeration through D=8, with finite private-input work.
const MAX_ACTIVE_SETS: usize = 50_000;

/// Tries cheap candidate separators in the normal space of one shared face.
pub(in crate::geometry) fn shared_face_fast_confinement<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
) -> bool
where
    L: Eq,
{
    if shared_labels.is_empty() || shared_labels.len() >= D {
        return false;
    }
    let shared_coordinates: Vec<_> = shared_labels
        .iter()
        .map(|label| {
            let first_index = first
                .labels()
                .iter()
                .position(|candidate| candidate == label)?;
            let second_index = second
                .labels()
                .iter()
                .position(|candidate| candidate == label)?;
            coordinates_are_identical(
                &first.coordinates()[first_index],
                &second.coordinates()[second_index],
            )
            .then_some(first.coordinates()[first_index])
        })
        .collect::<Option<_>>()
        .unwrap_or_default();
    if shared_coordinates.len() != shared_labels.len() {
        return false;
    }
    let base = shared_coordinates[0];
    let shared_deltas: Result<Vec<_>, _> = shared_coordinates[1..]
        .iter()
        .map(|coordinates| LaVector::try_new(from_fn(|axis| coordinates[axis] - base[axis])))
        .collect();
    let Ok(shared_deltas) = shared_deltas else {
        return false;
    };
    let Some(gram) = shared_face_gram(&shared_deltas) else {
        return false;
    };

    let mut axis = [0.0; D];
    let mut normalized_rays = Vec::with_capacity(2 * D);
    for (label, coordinates) in first.labels().iter().zip(first.coordinates()) {
        if shared_labels.contains(label) {
            continue;
        }
        let raw_ray = from_fn(|coordinate| coordinates[coordinate] - base[coordinate]);
        let Some(ray) = project_ray_orthogonal_to_shared_face(raw_ray, &shared_deltas, &gram)
        else {
            return false;
        };
        for coordinate in 0..D {
            axis[coordinate] += ray[coordinate];
        }
        push_normalized_ray::<D>(&mut normalized_rays, ray);
    }
    for (label, coordinates) in second.labels().iter().zip(second.coordinates()) {
        if shared_labels.contains(label) {
            continue;
        }
        let raw_ray = from_fn(|coordinate| base[coordinate] - coordinates[coordinate]);
        let Some(ray) = project_ray_orthogonal_to_shared_face(raw_ray, &shared_deltas, &gram)
        else {
            return false;
        };
        for coordinate in 0..D {
            axis[coordinate] += ray[coordinate];
        }
        push_normalized_ray::<D>(&mut normalized_rays, ray);
    }
    if axis.iter().all(|value| value.is_finite())
        && provisional_axis_certifies_confinement(first, second, shared_labels, &axis)
    {
        return true;
    }

    let mut refined_axis = [0.0; D];
    for ray in &normalized_rays {
        for (value, component) in refined_axis.iter_mut().zip(ray) {
            *value += component;
        }
    }
    for _ in 0..16 {
        let mut adjusted = false;
        for ray in &normalized_rays {
            let Ok(projection) = LaVector::try_new(refined_axis)
                .and_then(|axis| axis.dot(&LaVector::try_new(*ray)?))
            else {
                return false;
            };
            if projection < 1.0 {
                let correction = 1.0 - projection;
                for (value, component) in refined_axis.iter_mut().zip(ray) {
                    *value = component.mul_add(correction, *value);
                }
                adjusted = true;
            }
        }
        if !adjusted {
            break;
        }
    }
    refined_axis.iter().all(|value| value.is_finite())
        && provisional_axis_certifies_confinement(first, second, shared_labels, &refined_axis)
}

/// Keeps runtime Gram dispatch out of the single-shared-vertex path.
fn shared_face_gram<const D: usize>(deltas: &[LaVector<D>]) -> Option<Vec<Vec<f64>>> {
    if deltas.is_empty() {
        return Some(Vec::new());
    }
    la_stack::try_with_stack_matrix!(deltas.len(), |mut matrix| -> Result<_, LaError> {
        matrix = gram_matrix(&from_fn(|index| deltas[index]))?;
        Ok(matrix
            .into_rows()
            .into_iter()
            .map(Vec::from)
            .collect::<Vec<_>>())
    })
    .ok()
}

/// Removes provisional components tangent to the shared affine face.
fn project_ray_orthogonal_to_shared_face<const D: usize>(
    mut ray: [f64; D],
    shared_deltas: &[LaVector<D>],
    gram: &[Vec<f64>],
) -> Option<[f64; D]> {
    if shared_deltas.is_empty() {
        return Some(ray);
    }
    let vector = LaVector::try_new(ray).ok()?;
    let residuals: Result<Vec<_>, _> = shared_deltas
        .iter()
        .map(|delta| delta.dot(&vector))
        .collect();
    let correction = solve_f64_system(gram, &residuals.ok()?)?;
    for (coordinate, value) in ray.iter_mut().enumerate() {
        for (coefficient, delta) in correction.iter().zip(shared_deltas) {
            *value = (-coefficient).mul_add(delta.as_array()[coordinate], *value);
        }
    }
    ray.iter().all(|value| value.is_finite()).then_some(ray)
}

/// Certifies one provisional shared-face separator through a filtered or exact path.
fn provisional_axis_certifies_confinement<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
    axis: &[f64],
) -> bool
where
    L: Eq,
{
    if shared_labels.len() == 1 {
        filtered_single_shared_vertex_confinement(first, second, shared_labels, axis)
    } else {
        exact_shared_face_confinement_certificate(first, second, shared_labels, axis)
    }
}

/// Appends one finite unit ray for provisional separator refinement.
fn push_normalized_ray<const D: usize>(rays: &mut Vec<[f64; D]>, ray: [f64; D]) {
    let Ok(norm) = LaVector::try_new(ray).and_then(|vector| vector.norm()) else {
        return;
    };
    if norm > 0.0 {
        rays.push(ray.map(|component| component / norm));
    }
}

/// Outcome of the exact linear-program intersection path.
#[derive(Debug)]
pub(in crate::geometry) enum IntersectionLinearProgramResult<L> {
    /// The simplices are disjoint or meet only in their shared face.
    Valid,
    /// A feasible point has positive weight outside the shared face.
    Invalid(SimplexIntersectionWitness<L>),
    /// The first simplex cannot define a barycentric basis.
    SingularBarycentricBasis,
    /// A backend failure must never certify an empty intersection.
    SolverFailed {
        source: Box<StackMatrixDispatchError>,
    },
    /// An incomplete active-set search cannot certify validity.
    ActiveSetBudgetExceeded { limit: usize },
}

/// Keeps proven singularity distinct from an unavailable exact solve.
fn barycentric_failure<L>(source: StackMatrixDispatchError) -> IntersectionLinearProgramResult<L> {
    match source {
        StackMatrixDispatchError::La {
            source: LaError::Singular { .. },
        } => IntersectionLinearProgramResult::SingularBarycentricBasis,
        source => IntersectionLinearProgramResult::SolverFailed {
            source: Box::new(source),
        },
    }
}

/// Floating-point form used only to choose provisional revised-simplex bases.
struct ProvisionalIntersectionProgram {
    matrix: Vec<Vec<f64>>,
    rhs: Vec<f64>,
    objective: Vec<f64>,
    simplex_vertex_count: usize,
}

/// Builds the common-point equality constraints and shared-face objective.
fn provisional_intersection_program<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
) -> Option<ProvisionalIntersectionProgram>
where
    L: Eq,
{
    let first_labels = first.labels();
    let second_labels = second.labels();
    let simplex_vertex_count = first_labels.len();
    if second_labels.len() != simplex_vertex_count
        || first.coordinates().len() != simplex_vertex_count
        || second.coordinates().len() != simplex_vertex_count
    {
        return None;
    }

    let variable_count = simplex_vertex_count * 2;
    let mut matrix = vec![vec![0.0; variable_count]; D + 2];
    let mut rhs = vec![0.0; D + 2];
    rhs[0] = 1.0;
    rhs[1] = 1.0;

    for (alpha_index, coordinates) in first.coordinates().iter().enumerate() {
        matrix[0][alpha_index] = 1.0;
        for (axis, coordinate) in coordinates.iter().enumerate() {
            matrix[axis + 2][alpha_index] = *coordinate;
        }
    }
    for (beta_index, coordinates) in second.coordinates().iter().enumerate() {
        matrix[1][simplex_vertex_count + beta_index] = 1.0;
        for (axis, coordinate) in coordinates.iter().enumerate() {
            matrix[axis + 2][simplex_vertex_count + beta_index] = -*coordinate;
        }
    }

    let mut objective = vec![0.0; variable_count];
    for (index, label) in first_labels.iter().enumerate() {
        if !shared_labels.contains(label) {
            objective[index] = 1.0;
        }
    }
    for (index, label) in second_labels.iter().enumerate() {
        if !shared_labels.contains(label) {
            objective[simplex_vertex_count + index] = 1.0;
        }
    }

    Some(ProvisionalIntersectionProgram {
        matrix,
        rhs,
        objective,
        simplex_vertex_count,
    })
}

/// Uses exact linear programming to avoid enumerating every intersection-polytope basis.
///
/// The variables are the non-negative barycentric weights `alpha` and `beta`
/// of the two simplices. Equality constraints require each weight vector to sum
/// to one and both weighted coordinate sums to describe the same point.
/// Maximizing the total weight on non-shared labels is zero exactly when every
/// feasible intersection point lies in the shared face.
pub(in crate::geometry) fn intersection_via_linear_program<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
    verify_barycentric_basis: bool,
) -> IntersectionLinearProgramResult<L>
where
    L: Clone + Eq,
{
    let precomputed_barycentric = if verify_barycentric_basis {
        let coordinates = match barycentric_coordinates_of_vertices(second, first) {
            Ok(coordinates) => coordinates,
            Err(source) => return barycentric_failure(source),
        };
        Some(coordinates)
    } else {
        None
    };
    if verify_barycentric_basis {
        match barycentric_coordinates(&second.coordinates()[0], second) {
            Ok(_) => {}
            Err(StackMatrixDispatchError::La {
                source: LaError::Singular { .. },
            }) => {
                // Preserve label-level witnesses for collapsed second simplices,
                // just as the lower-dimensional orientation check does.
                return intersection_via_active_sets_impl(
                    first,
                    second,
                    shared_labels,
                    precomputed_barycentric.as_deref(),
                    false,
                );
            }
            Err(source) => return barycentric_failure(source),
        }
    }
    solve_intersection_program(
        first,
        second,
        shared_labels,
        precomputed_barycentric.as_deref(),
    )
}

/// Tries certified provisional separators before exact Phase I/II optimization.
fn solve_intersection_program<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
    precomputed_barycentric: Option<&[Vec<BigRational>]>,
) -> IntersectionLinearProgramResult<L>
where
    L: Clone + Eq,
{
    let Some(program) = provisional_intersection_program(first, second, shared_labels) else {
        return intersection_via_active_sets(first, second, shared_labels, precomputed_barycentric);
    };
    let ProvisionalIntersectionProgram {
        matrix: float_matrix,
        rhs: float_rhs,
        objective: float_objective,
        simplex_vertex_count,
    } = program;
    let variable_count = simplex_vertex_count * 2;
    let first_labels = first.labels();
    let second_labels = second.labels();

    let ProvisionalPhaseDuals {
        infeasibility_axis,
        confinement_axis,
        confinement_basis,
    } = provisional_phase_dual_axes(&float_matrix, &float_rhs, &float_objective, variable_count);
    if infeasibility_axis
        .is_some_and(|axis| simplices_are_strictly_separated_along(first, second, &axis))
    {
        return IntersectionLinearProgramResult::Valid;
    }
    let axis_certifies = confinement_axis.is_some_and(|axis| {
        filtered_single_shared_vertex_confinement(first, second, shared_labels, &axis)
            || exact_shared_face_confinement_certificate(first, second, shared_labels, &axis)
    });
    if axis_certifies {
        return IntersectionLinearProgramResult::Valid;
    }

    let matrix = exact_matrix_from_f64(&float_matrix);
    let rhs: Vec<_> = float_rhs.iter().copied().map(rational_from_f64).collect();
    let objective: Vec<_> = float_objective
        .iter()
        .copied()
        .map(rational_from_f64)
        .collect();
    let dual_certifies = confinement_basis.is_some_and(|basis| {
        phase_two_dual_proves_shared_face_confinement(&matrix, &rhs, &objective, &basis)
    });
    if dual_certifies {
        return IntersectionLinearProgramResult::Valid;
    }

    match maximize_nonnegative_equality_linear_program(&matrix, &rhs, &objective) {
        ExactLinearProgramResult::Infeasible => IntersectionLinearProgramResult::Valid,
        ExactLinearProgramResult::Optimal {
            objective_value, ..
        } if !objective_value.is_positive() => IntersectionLinearProgramResult::Valid,
        ExactLinearProgramResult::Optimal { solution, .. } => {
            let first_only_witness = positive_nonshared_labels(
                &solution[..simplex_vertex_count],
                first_labels,
                shared_labels,
            );
            let second_only_witness = positive_nonshared_labels(
                &solution[simplex_vertex_count..],
                second_labels,
                shared_labels,
            );
            if first_only_witness.is_empty() && second_only_witness.is_empty() {
                return intersection_via_active_sets(
                    first,
                    second,
                    shared_labels,
                    precomputed_barycentric,
                );
            }
            IntersectionLinearProgramResult::Invalid(SimplexIntersectionWitness {
                shared: shared_labels.iter().cloned().collect(),
                first_only_witness,
                second_only_witness,
            })
        }
        ExactLinearProgramResult::Failed => {
            intersection_via_active_sets(first, second, shared_labels, precomputed_barycentric)
        }
    }
}

/// Runs the pre-optimization active-set predicate without newer confinement shortcuts.
///
/// Exactly degenerate second simplices can represent a shared point with positive
/// weight on non-shared labels. The historical predicate reports that witness,
/// whereas a geometric confinement shortcut would erase the label-level overlap.
pub(in crate::geometry) fn intersection_via_legacy_active_sets<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
) -> IntersectionLinearProgramResult<L>
where
    L: Clone + Eq,
{
    intersection_via_active_sets_impl(first, second, shared_labels, None, false)
}

/// Preserves the exact active-set fallback when revised-simplex setup fails.
fn intersection_via_active_sets<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
    precomputed_barycentric: Option<&[Vec<BigRational>]>,
) -> IntersectionLinearProgramResult<L>
where
    L: Clone + Eq,
{
    intersection_via_active_sets_impl(first, second, shared_labels, precomputed_barycentric, true)
}

/// Implements active-set enumeration with an optional nondegenerate confinement certificate.
fn intersection_via_active_sets_impl<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
    precomputed_barycentric: Option<&[Vec<BigRational>]>,
    allow_confinement_certificate: bool,
) -> IntersectionLinearProgramResult<L>
where
    L: Clone + Eq,
{
    let computed_barycentric;
    let second_vertices_in_first = if let Some(coordinates) = precomputed_barycentric {
        coordinates
    } else {
        let coordinates = match barycentric_coordinates_of_vertices(second, first) {
            Ok(coordinates) => coordinates,
            Err(source) => return barycentric_failure(source),
        };
        computed_barycentric = coordinates;
        &computed_barycentric
    };

    if allow_confinement_certificate
        && intersection_is_confined_to_shared_face(
            second_vertices_in_first,
            first.labels(),
            shared_labels,
        )
    {
        return IntersectionLinearProgramResult::Valid;
    }

    let mut witness = None;
    let result = visit_intersection_vertices(second_vertices_in_first, MAX_ACTIVE_SETS, |beta| {
        let alpha = alpha_from_beta(beta, second_vertices_in_first);
        let first_only_witness = positive_nonshared_labels(&alpha, first.labels(), shared_labels);
        let second_only_witness = positive_nonshared_labels(beta, second.labels(), shared_labels);
        if first_only_witness.is_empty() && second_only_witness.is_empty() {
            return ControlFlow::Continue(());
        }
        witness = Some(SimplexIntersectionWitness {
            shared: shared_labels.iter().cloned().collect(),
            first_only_witness,
            second_only_witness,
        });
        ControlFlow::Break(())
    });
    match result {
        Err(ActiveSetFailure::Solver { source }) => IntersectionLinearProgramResult::SolverFailed {
            source: Box::new(source),
        },
        Err(ActiveSetFailure::Budget { limit }) => {
            IntersectionLinearProgramResult::ActiveSetBudgetExceeded { limit }
        }
        Ok(()) => witness.map_or(
            IntersectionLinearProgramResult::Valid,
            IntersectionLinearProgramResult::Invalid,
        ),
    }
}

/// Accepts a pair when simplex-facet signs prove any intersection lies in the shared face.
fn intersection_is_confined_to_shared_face<L>(
    other_vertices_in_basis: &[Vec<BigRational>],
    basis_labels: &[L],
    shared_labels: &[L],
) -> bool
where
    L: Eq,
{
    basis_labels
        .iter()
        .enumerate()
        .filter(|(_, label)| !shared_labels.contains(label))
        .all(|(basis_index, _)| {
            other_vertices_in_basis
                .iter()
                .all(|barycentric| !barycentric[basis_index].is_positive())
        })
}

/// Expresses every vertex of one simplex in the barycentric basis of another.
fn barycentric_coordinates_of_vertices<L, const D: usize>(
    vertices: &LabeledSimplexRealization<L, D>,
    basis: &LabeledSimplexRealization<L, D>,
) -> Result<Vec<Vec<BigRational>>, StackMatrixDispatchError> {
    vertices
        .coordinates()
        .iter()
        .map(|coordinates| barycentric_coordinates(coordinates, basis))
        .collect()
}

/// Computes exact barycentric coordinates of one point in one simplex basis.
fn barycentric_coordinates<L, const D: usize>(
    point: &[f64; D],
    simplex: &LabeledSimplexRealization<L, D>,
) -> Result<Vec<BigRational>, StackMatrixDispatchError> {
    if D == 0 {
        return Ok(vec![rational_one()]);
    }

    let origin = &simplex.coordinates()[0];
    let mut matrix = vec![vec![rational_zero(); D]; D];
    let mut rhs = vec![rational_zero(); D];

    for axis in 0..D {
        let origin_coordinate = rational_from_f64(origin[axis]);
        rhs[axis] = rational_from_f64(point[axis]) - origin_coordinate.clone();
        for (column, matrix_value) in matrix[axis].iter_mut().enumerate() {
            *matrix_value = rational_from_f64(simplex.coordinates()[column + 1][axis])
                - origin_coordinate.clone();
        }
    }

    let lambdas = solve_rational_system(&matrix, &rhs)?;
    let lambda_sum = lambdas
        .iter()
        .fold(rational_zero(), |sum, value| sum + value.clone());
    let mut barycentric = Vec::with_capacity(D + 1);
    barycentric.push(rational_one() - lambda_sum);
    barycentric.extend(lambdas);
    Ok(barycentric)
}

/// Failure to complete the fallback is distinct from an infeasible candidate.
#[derive(Debug)]
enum ActiveSetFailure {
    Solver { source: StackMatrixDispatchError },
    Budget { limit: usize },
}

/// Visits feasible intersection vertices, stopping early on a decisive witness.
/// C(18,8)=43,758 is the complete D=8 envelope. The explicit budget also bounds
/// malformed/private inputs and never turns incomplete enumeration into validity.
fn visit_intersection_vertices<F>(
    second_vertices_in_first: &[Vec<BigRational>],
    limit: usize,
    mut on_vertex: F,
) -> Result<(), ActiveSetFailure>
where
    F: FnMut(&[BigRational]) -> ControlFlow<()>,
{
    let variable_count = second_vertices_in_first.len();
    let active_count = variable_count.saturating_sub(1);
    let constraint_count = variable_count * 2;
    let mut active_set = Vec::with_capacity(active_count);
    let mut completed = 0;
    let _ = enumerate_active_sets(
        constraint_count,
        active_count,
        0,
        &mut active_set,
        &mut |active_constraints| {
            if completed == limit {
                return Err(ActiveSetFailure::Budget { limit });
            }
            completed += 1;
            match intersection_vertex_for_active_set(second_vertices_in_first, active_constraints) {
                Ok(beta) if beta_is_feasible(&beta, second_vertices_in_first) => {
                    Ok(on_vertex(&beta))
                }
                Ok(_)
                | Err(StackMatrixDispatchError::La {
                    source: LaError::Singular { .. },
                }) => Ok(ControlFlow::Continue(())),
                Err(source) => Err(ActiveSetFailure::Solver { source }),
            }
        },
    )?;
    Ok(())
}

/// Enumerates each constraint set once, propagating errors and early witnesses.
fn enumerate_active_sets<F>(
    constraint_count: usize,
    active_count: usize,
    start: usize,
    active_set: &mut Vec<usize>,
    on_active_set: &mut F,
) -> Result<ControlFlow<()>, ActiveSetFailure>
where
    F: FnMut(&[usize]) -> Result<ControlFlow<()>, ActiveSetFailure>,
{
    if active_set.len() == active_count {
        return on_active_set(active_set);
    }
    let remaining = active_count - active_set.len();
    let last_start = constraint_count.saturating_sub(remaining);
    for constraint in start..=last_start {
        active_set.push(constraint);
        let result = enumerate_active_sets(
            constraint_count,
            active_count,
            constraint + 1,
            active_set,
            on_active_set,
        )?;
        active_set.pop();
        if result.is_break() {
            return Ok(result);
        }
    }
    Ok(ControlFlow::Continue(()))
}

/// Solves one active-constraint system and returns the candidate beta weights.
fn intersection_vertex_for_active_set(
    second_vertices_in_first: &[Vec<BigRational>],
    active_constraints: &[usize],
) -> Result<Vec<BigRational>, StackMatrixDispatchError> {
    let variable_count = second_vertices_in_first.len();
    let mut matrix = Vec::with_capacity(variable_count);
    let mut rhs = Vec::with_capacity(variable_count);

    matrix.push(vec![rational_one(); variable_count]);
    rhs.push(rational_one());

    for &constraint in active_constraints {
        matrix.push(constraint_coefficients(
            second_vertices_in_first,
            constraint,
        ));
        rhs.push(rational_zero());
    }

    solve_rational_system(&matrix, &rhs)
}

/// Builds coefficients for either a beta non-negativity or alpha non-negativity constraint.
fn constraint_coefficients(
    second_vertices_in_first: &[Vec<BigRational>],
    constraint: usize,
) -> Vec<BigRational> {
    let variable_count = second_vertices_in_first.len();
    if constraint < variable_count {
        let mut coefficients = vec![rational_zero(); variable_count];
        coefficients[constraint] = rational_one();
        return coefficients;
    }

    let alpha_index = constraint - variable_count;
    second_vertices_in_first
        .iter()
        .map(|barycentric| barycentric[alpha_index].clone())
        .collect()
}

/// Checks whether beta weights and the induced alpha weights are all non-negative.
fn beta_is_feasible(beta: &[BigRational], second_vertices_in_first: &[Vec<BigRational>]) -> bool {
    beta.iter().all(|value| !value.is_negative())
        && alpha_from_beta(beta, second_vertices_in_first)
            .iter()
            .all(|value| !value.is_negative())
}

/// Converts second-simplex beta weights into first-simplex alpha weights.
fn alpha_from_beta(
    beta: &[BigRational],
    second_vertices_in_first: &[Vec<BigRational>],
) -> Vec<BigRational> {
    let variable_count = second_vertices_in_first.len();
    let mut alpha = vec![rational_zero(); variable_count];

    for (beta_index, beta_value) in beta.iter().enumerate() {
        for (alpha_index, alpha_value) in alpha.iter_mut().enumerate() {
            *alpha_value = alpha_value.clone()
                + beta_value.clone() * second_vertices_in_first[beta_index][alpha_index].clone();
        }
    }

    alpha
}

/// Result of an exact non-negative equality-constrained linear program.
#[derive(Debug)]
enum ExactLinearProgramResult {
    /// The objective is maximized at an exact basic feasible solution.
    Optimal {
        /// Values for the original decision variables.
        solution: Vec<BigRational>,
        /// Exact objective value at `solution`.
        objective_value: BigRational,
    },
    /// No non-negative solution satisfies the equality constraints.
    Infeasible,
    /// Tableau construction or a boundedness invariant failed.
    Failed,
}

/// Result of one revised-simplex optimization phase.
#[derive(Debug)]
enum RevisedSimplexResult {
    /// The current basis is optimal.
    Optimal {
        /// Values for every variable in the supplied matrix.
        solution: Vec<BigRational>,
        /// Exact objective value at `solution`.
        objective_value: BigRational,
    },
    /// The objective can increase without bound.
    Unbounded,
    /// The supplied basis or iteration budget failed.
    Failed,
}

/// Maximizes an exact objective subject to `matrix * x = rhs` and `x >= 0`.
///
/// Phase I introduces one artificial variable per equality. [Bland's pivot
/// rule](https://github.com/acgetchell/delaunay/blob/main/REFERENCES.md#point-in-polytope-testing)
/// makes degenerate exact-arithmetic bases deterministic and avoids
/// cycling. Artificial variables are pivoted out at zero before Phase II.
/// The explicit iteration budget can still return `Failed` before termination.
fn maximize_nonnegative_equality_linear_program(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    objective: &[BigRational],
) -> ExactLinearProgramResult {
    let constraint_count = rhs.len();
    let variable_count = objective.len();
    if constraint_count == 0
        || matrix.len() != constraint_count
        || matrix.iter().any(|row| row.len() != variable_count)
        || rhs.iter().any(Signed::is_negative)
    {
        return ExactLinearProgramResult::Failed;
    }

    let total_variable_count = variable_count + constraint_count;
    let mut phase_matrix = Vec::with_capacity(constraint_count);
    for (row_index, row) in matrix.iter().enumerate() {
        let mut phase_row = Vec::with_capacity(total_variable_count);
        phase_row.extend(row.iter().cloned());
        phase_row.extend((0..constraint_count).map(|column| {
            if column == row_index {
                rational_one()
            } else {
                rational_zero()
            }
        }));
        phase_matrix.push(phase_row);
    }

    let mut basis: Vec<usize> = (variable_count..total_variable_count).collect();
    let mut phase_one_objective = vec![rational_zero(); total_variable_count];
    for coefficient in &mut phase_one_objective[variable_count..] {
        *coefficient = -rational_one();
    }

    let phase_one = revised_simplex_maximize(
        &phase_matrix,
        rhs,
        &phase_one_objective,
        total_variable_count,
        &mut basis,
    );
    let RevisedSimplexResult::Optimal {
        objective_value, ..
    } = phase_one
    else {
        return ExactLinearProgramResult::Failed;
    };
    if objective_value.is_negative() {
        return ExactLinearProgramResult::Infeasible;
    }
    if objective_value.is_positive()
        || !pivot_artificial_variables_out(&phase_matrix, rhs, variable_count, &mut basis)
    {
        return ExactLinearProgramResult::Failed;
    }

    let mut phase_two_objective = vec![rational_zero(); total_variable_count];
    phase_two_objective[..variable_count].clone_from_slice(objective);
    match revised_simplex_maximize(
        &phase_matrix,
        rhs,
        &phase_two_objective,
        variable_count,
        &mut basis,
    ) {
        RevisedSimplexResult::Optimal {
            solution,
            objective_value,
        } => ExactLinearProgramResult::Optimal {
            solution: solution[..variable_count].to_vec(),
            objective_value,
        },
        RevisedSimplexResult::Unbounded | RevisedSimplexResult::Failed => {
            ExactLinearProgramResult::Failed
        }
    }
}

/// Removes zero-valued Phase I artificial variables from the active basis.
fn pivot_artificial_variables_out(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    original_variable_count: usize,
    basis: &mut [usize],
) -> bool {
    for basis_position in 0..basis.len() {
        if basis[basis_position] < original_variable_count {
            continue;
        }

        let Some(current_basis_matrix) = basis_matrix(matrix, basis) else {
            return false;
        };
        let Ok(basic_solution) = solve_rational_system(&current_basis_matrix, rhs) else {
            return false;
        };
        if basic_solution[basis_position] != rational_zero() {
            return false;
        }

        let mut replacement = None;
        for candidate in 0..original_variable_count {
            if basis.contains(&candidate) {
                continue;
            }
            let Ok(direction) =
                solve_rational_system(&current_basis_matrix, &matrix_column(matrix, candidate))
            else {
                continue;
            };
            if direction[basis_position] != rational_zero() {
                replacement = Some(candidate);
                break;
            }
        }
        let Some(candidate) = replacement else {
            return false;
        };
        basis[basis_position] = candidate;
    }
    true
}

/// Uses a floating-point basis search followed by an exact optimality certificate.
///
/// The floating-point phase is only a guide: it cannot return a validation
/// result directly. A candidate basis is accepted only after exact rational
/// primal and dual checks. If certification fails, the deterministic exact
/// simplex path restarts from the original basis.
fn revised_simplex_maximize(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    objective: &[BigRational],
    entering_variable_count: usize,
    basis: &mut [usize],
) -> RevisedSimplexResult {
    let starting_basis = basis.to_vec();
    if navigate_simplex_basis_f64(matrix, rhs, objective, entering_variable_count, basis).is_some()
        && let Some(optimal) =
            certify_optimal_basis(matrix, rhs, objective, entering_variable_count, basis)
    {
        return optimal;
    }

    basis.clone_from_slice(&starting_basis);
    revised_simplex_maximize_exact(matrix, rhs, objective, entering_variable_count, basis)
}

/// Runs deterministic exact revised simplex from an existing feasible basis.
fn revised_simplex_maximize_exact(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    objective: &[BigRational],
    entering_variable_count: usize,
    basis: &mut [usize],
) -> RevisedSimplexResult {
    const MAX_SIMPLEX_ITERATIONS: usize = 10_000;

    if matrix.is_empty()
        || matrix.len() != rhs.len()
        || basis.len() != rhs.len()
        || matrix.iter().any(|row| row.len() != objective.len())
        || entering_variable_count > objective.len()
    {
        return RevisedSimplexResult::Failed;
    }

    for _ in 0..MAX_SIMPLEX_ITERATIONS {
        let Some(current_basis_matrix) = basis_matrix(matrix, basis) else {
            return RevisedSimplexResult::Failed;
        };
        let Ok(basic_solution) = solve_rational_system(&current_basis_matrix, rhs) else {
            return RevisedSimplexResult::Failed;
        };
        if basic_solution.iter().any(Signed::is_negative) {
            return RevisedSimplexResult::Failed;
        }

        let basic_objective: Vec<_> = basis
            .iter()
            .map(|&index| objective[index].clone())
            .collect();
        let Ok(dual_solution) = solve_rational_system(
            &transpose_square_matrix(&current_basis_matrix),
            &basic_objective,
        ) else {
            return RevisedSimplexResult::Failed;
        };

        let mut entering = None;
        for candidate in (0..entering_variable_count).filter(|candidate| !basis.contains(candidate))
        {
            let Ok(contribution) = rational_dot(&dual_solution, &matrix_column(matrix, candidate))
            else {
                return RevisedSimplexResult::Failed;
            };
            if (objective[candidate].clone() - contribution).is_positive() {
                entering = Some(candidate);
                break;
            }
        }
        let Some(entering) = entering else {
            let mut solution = vec![rational_zero(); objective.len()];
            for (&variable, value) in basis.iter().zip(&basic_solution) {
                solution[variable] = value.clone();
            }
            let Ok(objective_value) = rational_dot(objective, &solution) else {
                return RevisedSimplexResult::Failed;
            };
            return RevisedSimplexResult::Optimal {
                objective_value,
                solution,
            };
        };

        let Ok(direction) =
            solve_rational_system(&current_basis_matrix, &matrix_column(matrix, entering))
        else {
            return RevisedSimplexResult::Failed;
        };
        let leaving = direction
            .iter()
            .enumerate()
            .filter(|(_, coefficient)| coefficient.is_positive())
            .map(|(position, coefficient)| {
                (
                    position,
                    basic_solution[position].clone() / coefficient.clone(),
                )
            })
            .min_by(
                |(left_position, left_ratio), (right_position, right_ratio)| {
                    left_ratio
                        .cmp(right_ratio)
                        .then_with(|| basis[*left_position].cmp(&basis[*right_position]))
                },
            );
        let Some((leaving_position, _)) = leaving else {
            return RevisedSimplexResult::Unbounded;
        };
        basis[leaving_position] = entering;
    }

    RevisedSimplexResult::Failed
}

/// Searches for an optimal basis quickly using f64 arithmetic.
///
/// Every decision made here is provisional. The caller must pass the resulting
/// basis through [`certify_optimal_basis`] before using it.
fn navigate_simplex_basis_f64(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    objective: &[BigRational],
    entering_variable_count: usize,
    basis: &mut [usize],
) -> Option<Vec<f64>> {
    let float_matrix: Vec<_> = matrix
        .iter()
        .map(|row| rationals_to_f64(row))
        .collect::<Option<_>>()?;
    let float_rhs = rationals_to_f64(rhs)?;
    let float_objective = rationals_to_f64(objective)?;
    navigate_simplex_basis_f64_values(
        &float_matrix,
        &float_rhs,
        &float_objective,
        entering_variable_count,
        basis,
    )
}

/// Runs provisional revised simplex directly on finite floating-point values.
fn navigate_simplex_basis_f64_values(
    matrix: &[Vec<f64>],
    rhs: &[f64],
    objective: &[f64],
    entering_variable_count: usize,
    basis: &mut [usize],
) -> Option<Vec<f64>> {
    const MAX_SIMPLEX_ITERATIONS: usize = 10_000;
    const REDUCED_COST_TOLERANCE: f64 = 1.0e-10;
    const DIRECTION_TOLERANCE: f64 = 1.0e-12;
    const FEASIBILITY_TOLERANCE: f64 = 1.0e-9;

    if matrix.is_empty()
        || matrix.len() != rhs.len()
        || basis.len() != rhs.len()
        || matrix.iter().any(|row| row.len() != objective.len())
        || entering_variable_count > objective.len()
    {
        return None;
    }

    for _ in 0..MAX_SIMPLEX_ITERATIONS {
        let current_basis_matrix = basis_matrix_f64(matrix, basis)?;
        let mut basic_solution = solve_f64_system(&current_basis_matrix, rhs)?;
        if basic_solution
            .iter()
            .any(|&value| value < -FEASIBILITY_TOLERANCE)
        {
            return None;
        }
        for value in &mut basic_solution {
            if value.is_sign_negative() {
                *value = 0.0;
            }
        }

        let basic_objective: Vec<_> = basis.iter().map(|&index| objective[index]).collect();
        let dual_solution = solve_f64_system(
            &transpose_square_matrix_f64(&current_basis_matrix),
            &basic_objective,
        )?;

        let mut entering = None;
        for candidate in (0..entering_variable_count).filter(|candidate| !basis.contains(candidate))
        {
            let column = matrix_column_f64(matrix, candidate);
            let dual_contribution = dot_product_f64(&dual_solution, &column)?;
            let scale = 1.0 + objective[candidate].abs() + dual_contribution.abs();
            if objective[candidate] - dual_contribution > REDUCED_COST_TOLERANCE * scale {
                entering = Some(candidate);
                break;
            }
        }
        let Some(entering) = entering else {
            return Some(dual_solution);
        };

        let direction =
            solve_f64_system(&current_basis_matrix, &matrix_column_f64(matrix, entering))?;
        let leaving = direction
            .iter()
            .enumerate()
            .filter(|(_, coefficient)| **coefficient > DIRECTION_TOLERANCE)
            .map(|(position, coefficient)| (position, basic_solution[position] / coefficient))
            .min_by(
                |(left_position, left_ratio), (right_position, right_ratio)| {
                    left_ratio
                        .total_cmp(right_ratio)
                        .then_with(|| basis[*left_position].cmp(&basis[*right_position]))
                },
            );
        let (leaving_position, _) = leaving?;
        basis[leaving_position] = entering;
    }

    None
}

/// Finds candidate separating directions from floating-point Phase I and II duals.
///
/// Returned directions are only guides. Callers must verify their geometric
/// certificates with exact arithmetic before accepting them.
struct ProvisionalPhaseDuals {
    /// Candidate direction from the infeasibility phase.
    infeasibility_axis: Option<Vec<f64>>,
    /// Candidate direction from the shared-face-confinement phase.
    confinement_axis: Option<Vec<f64>>,
    /// Provisional Phase II basis available for exact dual certification.
    confinement_basis: Option<Vec<usize>>,
}

impl ProvisionalPhaseDuals {
    /// Returns a result with no provisional certificates.
    const fn none() -> Self {
        Self {
            infeasibility_axis: None,
            confinement_axis: None,
            confinement_basis: None,
        }
    }
}

fn provisional_phase_dual_axes(
    matrix: &[Vec<f64>],
    rhs: &[f64],
    original_objective: &[f64],
    variable_count: usize,
) -> ProvisionalPhaseDuals {
    let constraint_count = rhs.len();
    let Some(total_variable_count) = variable_count.checked_add(constraint_count) else {
        return ProvisionalPhaseDuals::none();
    };
    let mut phase_matrix = Vec::with_capacity(constraint_count);
    for (row_index, row) in matrix.iter().enumerate() {
        let mut phase_row = Vec::with_capacity(total_variable_count);
        phase_row.extend(row.iter().copied());
        phase_row.extend((0..constraint_count).map(
            |column| {
                if column == row_index { 1.0 } else { 0.0 }
            },
        ));
        phase_matrix.push(phase_row);
    }

    let mut basis: Vec<_> = (variable_count..total_variable_count).collect();
    let mut objective = vec![0.0; total_variable_count];
    for coefficient in &mut objective[variable_count..] {
        *coefficient = -1.0;
    }
    let Some(phase_one_dual) = navigate_simplex_basis_f64_values(
        &phase_matrix,
        rhs,
        &objective,
        total_variable_count,
        &mut basis,
    ) else {
        return ProvisionalPhaseDuals::none();
    };
    let phase_one_axis = (phase_one_dual.len() >= 2).then(|| phase_one_dual[2..].to_vec());

    if !pivot_artificial_variables_out_f64(&phase_matrix, rhs, variable_count, &mut basis) {
        return ProvisionalPhaseDuals {
            infeasibility_axis: phase_one_axis,
            confinement_axis: None,
            confinement_basis: None,
        };
    }
    let mut phase_two_objective = vec![0.0; total_variable_count];
    if original_objective.len() != variable_count {
        return ProvisionalPhaseDuals {
            infeasibility_axis: phase_one_axis,
            confinement_axis: None,
            confinement_basis: None,
        };
    }
    phase_two_objective[..variable_count].clone_from_slice(original_objective);
    let phase_two_dual = navigate_simplex_basis_f64_values(
        &phase_matrix,
        rhs,
        &phase_two_objective,
        variable_count,
        &mut basis,
    );
    let phase_two_axis = phase_two_dual
        .as_ref()
        .and_then(|dual| (dual.len() >= 2).then(|| dual[2..].to_vec()));
    let phase_two_basis = phase_two_dual.map(|_dual| basis);
    ProvisionalPhaseDuals {
        infeasibility_axis: phase_one_axis,
        confinement_axis: phase_two_axis,
        confinement_basis: phase_two_basis,
    }
}

/// Removes zero artificial variables from a provisional floating-point basis.
fn pivot_artificial_variables_out_f64(
    matrix: &[Vec<f64>],
    rhs: &[f64],
    original_variable_count: usize,
    basis: &mut [usize],
) -> bool {
    const ZERO_TOLERANCE: f64 = 1.0e-9;
    const PIVOT_TOLERANCE: f64 = 1.0e-12;

    for basis_position in 0..basis.len() {
        if basis[basis_position] < original_variable_count {
            continue;
        }
        let Some(current_basis_matrix) = basis_matrix_f64(matrix, basis) else {
            return false;
        };
        let Some(basic_solution) = solve_f64_system(&current_basis_matrix, rhs) else {
            return false;
        };
        if basic_solution[basis_position].abs() > ZERO_TOLERANCE {
            return false;
        }

        let replacement = (0..original_variable_count)
            .filter(|candidate| !basis.contains(candidate))
            .find(|&candidate| {
                solve_f64_system(&current_basis_matrix, &matrix_column_f64(matrix, candidate))
                    .is_some_and(|direction| direction[basis_position].abs() > PIVOT_TOLERANCE)
            });
        let Some(replacement) = replacement else {
            return false;
        };
        basis[basis_position] = replacement;
    }
    true
}

/// Certifies strict projection separation along a finite candidate direction.
fn simplices_are_strictly_separated_along<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    axis: &[f64],
) -> bool {
    if axis.len() != D || axis.iter().any(|value| !value.is_finite()) {
        return false;
    }
    let exact_axis: Vec<_> = axis.iter().copied().map(rational_from_f64).collect();
    let Some((first_min, first_max)) = exact_projection_bounds(first, &exact_axis) else {
        return false;
    };
    let Some((second_min, second_max)) = exact_projection_bounds(second, &exact_axis) else {
        return false;
    };
    first_max < second_min || second_max < first_min
}

/// Returns exact projection bounds for one realized simplex.
fn exact_projection_bounds<L, const D: usize>(
    simplex: &LabeledSimplexRealization<L, D>,
    axis: &[BigRational],
) -> Option<(BigRational, BigRational)> {
    let mut projections = simplex.coordinates().iter().map(|coordinates| {
        coordinates
            .iter()
            .zip(axis)
            .fold(rational_zero(), |sum, (coordinate, coefficient)| {
                sum + rational_from_f64(*coordinate) * coefficient
            })
    });
    let first = projections.next()?;
    Some(
        projections.fold((first.clone(), first), |(minimum, maximum), projection| {
            (minimum.min(projection.clone()), maximum.max(projection))
        }),
    )
}

/// Certifies confinement through one shared vertex with a floating-point error bound.
///
/// Scaling the provisional direction is exact because the factor is a power of
/// two. Each signed projection difference is then accepted only when its
/// conservative lower bound exceeds the unit dual margin. Ambiguous,
/// underflow-sensitive, or overflowing cases continue to exact arithmetic.
fn filtered_single_shared_vertex_confinement<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
    axis: &[f64],
) -> bool
where
    L: Eq,
{
    let [shared_label] = shared_labels else {
        return false;
    };
    let Some(first_shared_index) = first
        .labels()
        .iter()
        .position(|candidate| candidate == shared_label)
    else {
        return false;
    };
    let Some(second_shared_index) = second
        .labels()
        .iter()
        .position(|candidate| candidate == shared_label)
    else {
        return false;
    };
    let first_shared = &first.coordinates()[first_shared_index];
    let second_shared = &second.coordinates()[second_shared_index];
    if !coordinates_are_identical(first_shared, second_shared) {
        return false;
    }

    let Ok(axis): Result<[f64; D], _> = axis.try_into() else {
        return false;
    };
    let Ok(scaled_axis) = LaVector::try_new(axis.map(|value| value * 1024.0)) else {
        return false;
    };
    let Ok(shared) = LaVector::try_new(*first_shared) else {
        return false;
    };
    let Ok(Some(common)) = scaled_axis.dot_with_errbound(&shared) else {
        return false;
    };
    // Reuse the common projection and enclose its unit-margin thresholds.
    // All projections use original coordinates, never rounded differences.
    let Ok(positive_margin) = Interval::try_from_subtraction(common.upper_bound(), -1.0) else {
        return false;
    };
    let Ok(negative_margin) = Interval::try_from_subtraction(common.lower_bound(), 1.0) else {
        return false;
    };
    for (label, coordinates) in first.labels().iter().zip(first.coordinates()) {
        if label == shared_label {
            continue;
        }
        let Ok(vertex) = LaVector::try_new(*coordinates) else {
            return false;
        };
        let Ok(Some(projection)) = scaled_axis.dot_with_errbound(&vertex) else {
            return false;
        };
        // For representable x, x > RN(y) implies x > y: rounding is
        // monotone and RN(x) = x. This proves estimate - error > margin
        // without constructing another endpoint. Overflow rejects the proof.
        if projection.estimate() <= positive_margin.upper() + projection.absolute_error_bound() {
            return false;
        }
    }
    for (label, coordinates) in second.labels().iter().zip(second.coordinates()) {
        if label == shared_label {
            continue;
        }
        let Ok(vertex) = LaVector::try_new(*coordinates) else {
            return false;
        };
        let Ok(Some(projection)) = scaled_axis.dot_with_errbound(&vertex) else {
            return false;
        };
        // The symmetric strict comparison proves estimate + error < margin.
        // Overflow produces negative infinity and rejects the proof.
        if projection.estimate() >= negative_margin.lower() - projection.absolute_error_bound() {
            return false;
        }
    }
    true
}

/// Compares finite coordinates exactly while treating both signed zeros alike.
pub(in crate::geometry) fn coordinates_are_identical<const D: usize>(
    left: &[f64; D],
    right: &[f64; D],
) -> bool {
    left.iter().zip(right).all(|(left, right)| {
        let left_bits = left.to_bits();
        let right_bits = right.to_bits();
        left_bits == right_bits || (left_bits << 1 == 0 && right_bits << 1 == 0)
    })
}

/// Certifies a Phase II dual after rebuilding its two affine offsets exactly.
///
/// For direction `w`, the smallest feasible dual offsets are
/// `max_i(c_i - w·a_i)` and `max_j(c_j + w·b_j)`. Their sum is an exact upper
/// bound on the maximum non-shared barycentric weight. A non-positive bound
/// proves that every intersection point lies in the shared face.
fn exact_shared_face_confinement_certificate<L, const D: usize>(
    first: &LabeledSimplexRealization<L, D>,
    second: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
    axis: &[f64],
) -> bool
where
    L: Eq,
{
    if axis.len() != D || axis.iter().any(|value| !value.is_finite()) {
        return false;
    }
    let Some(exact_axis) = exact_axis_through_shared_face(first, shared_labels, axis) else {
        return false;
    };
    let first_offset = first
        .labels()
        .iter()
        .zip(first.coordinates())
        .map(|(label, coordinates)| {
            let objective = if shared_labels.contains(label) {
                rational_zero()
            } else {
                rational_one()
            };
            objective - exact_projection(coordinates, &exact_axis)
        })
        .max();
    let second_offset = second
        .labels()
        .iter()
        .zip(second.coordinates())
        .map(|(label, coordinates)| {
            let objective = if shared_labels.contains(label) {
                rational_zero()
            } else {
                rational_one()
            };
            objective + exact_projection(coordinates, &exact_axis)
        })
        .max();
    first_offset
        .zip(second_offset)
        .is_some_and(|(first_offset, second_offset)| {
            first_offset + second_offset <= rational_zero()
        })
}

/// Reconstructs a provisional axis in the shared face's exact affine nullspace.
///
/// Shared vertices define homogeneous constraints on `(axis, offset)`. A
/// floating-point rank search selects pivot columns, then the fraction-free
/// exact solver constructs one nullspace basis vector per free column directly
/// from the original coordinates. Combining those vectors with provisional
/// free-coordinate weights restores exact shared-face equality without a
/// rational Gram solve.
fn exact_axis_through_shared_face<L, const D: usize>(
    simplex: &LabeledSimplexRealization<L, D>,
    shared_labels: &[L],
    axis: &[f64],
) -> Option<Vec<BigRational>>
where
    L: Eq,
{
    if axis.len() != D || axis.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let shared_coordinates: Vec<_> = shared_labels
        .iter()
        .map(|label| {
            simplex
                .labels()
                .iter()
                .position(|candidate| candidate == label)
                .and_then(|index| simplex.coordinates().get(index))
        })
        .collect::<Option<_>>()?;
    if shared_coordinates.is_empty() {
        let scaled_axis: Vec<_> = axis.iter().map(|value| *value * 1024.0).collect();
        if scaled_axis.iter().any(|value| !value.is_finite()) {
            return None;
        }
        return Some(scaled_axis.into_iter().map(rational_from_f64).collect());
    }

    let constraint_matrix: Vec<Vec<f64>> = shared_coordinates
        .iter()
        .map(|coordinates| {
            let mut row = coordinates.to_vec();
            row.push(1.0);
            row
        })
        .collect();
    let pivot_columns = shared_face_pivot_columns(&constraint_matrix)?;
    let free_columns: Vec<_> = (0..=D)
        .filter(|column| !pivot_columns.contains(column))
        .collect();
    let pivot_matrix: Vec<Vec<f64>> = constraint_matrix
        .iter()
        .map(|row| pivot_columns.iter().map(|&column| row[column]).collect())
        .collect();

    let mut provisional_affine = axis.to_vec();
    provisional_affine.push(-dot_product_f64(axis, shared_coordinates[0])?);
    if provisional_affine.iter().any(|value| !value.is_finite()) {
        return None;
    }
    let mut exact_affine = vec![rational_zero(); D + 1];
    for free_column in free_columns {
        let weight = rational_from_f64(provisional_affine[free_column]);
        exact_affine[free_column] += weight.clone();
        let rhs: Vec<_> = constraint_matrix
            .iter()
            .map(|row| -row[free_column])
            .collect();
        let solution = solve_exact_runtime_system(&pivot_matrix, &rhs)?.ok()?;
        for (&pivot_column, coefficient) in pivot_columns.iter().zip(solution) {
            exact_affine[pivot_column] += weight.clone() * coefficient;
        }
    }

    let scale = rational_from_f64(1024.0);
    for value in &mut exact_affine[..D] {
        *value *= scale.clone();
    }
    exact_affine.truncate(D);
    Some(exact_affine)
}

/// Adapts rectangular affine constraints to la-stack's provisional selector.
/// Zero padding to the column count preserves the original pivots. Admit the
/// full shape first; no entry is truncated. The absolute threshold retains the
/// consumer policy `1e-12 * max(1, max |a_ij|)`, with no row/column equilibration.
/// Rank loss or non-finite arithmetic only declines this shortcut; every selected
/// basis is subsequently solved exactly against the original coordinates.
fn shared_face_pivot_columns(matrix: &[Vec<f64>]) -> Option<Vec<usize>> {
    let rows = matrix.len();
    let columns = matrix.first()?.len();
    if rows > columns || matrix.iter().any(|row| row.len() != columns) {
        return None;
    }
    match columns {
        1 => select_shared_face_columns::<1>(matrix),
        2 => select_shared_face_columns::<2>(matrix),
        3 => select_shared_face_columns::<3>(matrix),
        4 => select_shared_face_columns::<4>(matrix),
        5 => select_shared_face_columns::<5>(matrix),
        6 => select_shared_face_columns::<6>(matrix),
        7 => select_shared_face_columns::<7>(matrix),
        8 => select_shared_face_columns::<8>(matrix),
        9 => select_shared_face_columns::<9>(matrix),
        _ => None,
    }
}

/// Keeps padding proportional to the admitted affine shape, including low D.
fn select_shared_face_columns<const N: usize>(matrix: &[Vec<f64>]) -> Option<Vec<usize>> {
    let mut padded = [[0.0; N]; N];
    let mut scale = 1.0_f64;
    for (target, source) in padded.iter_mut().zip(matrix) {
        target.copy_from_slice(source);
        for &value in source {
            scale = scale.max(value.abs());
        }
    }
    let basis = column_basis(&padded, Tolerance::try_new(1.0e-12 * scale).ok()?).ok()?;
    (basis.rank() == matrix.len()).then(|| basis.columns().to_vec())
}

/// Certifies the floating-point Phase II basis with one exact dual solve.
///
/// Dual feasibility and a non-positive exact objective bound prove confinement
/// without constructing a primal solution or repeating exact simplex pivots.
fn phase_two_dual_proves_shared_face_confinement(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    objective: &[BigRational],
    basis: &[usize],
) -> bool {
    let constraint_count = rhs.len();
    let variable_count = objective.len();
    if matrix.len() != constraint_count
        || matrix.iter().any(|row| row.len() != variable_count)
        || basis.len() != constraint_count
    {
        return false;
    }

    let total_variable_count = variable_count + constraint_count;
    let phase_matrix: Vec<_> = matrix
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            let mut phase_row = Vec::with_capacity(total_variable_count);
            phase_row.extend(row.iter().cloned());
            phase_row.extend((0..constraint_count).map(|column| {
                if column == row_index {
                    rational_one()
                } else {
                    rational_zero()
                }
            }));
            phase_row
        })
        .collect();
    let mut phase_objective = vec![rational_zero(); total_variable_count];
    phase_objective[..variable_count].clone_from_slice(objective);

    let Some(current_basis_matrix) = basis_matrix(&phase_matrix, basis) else {
        return false;
    };
    let basic_objective: Vec<_> = basis
        .iter()
        .filter_map(|&index| phase_objective.get(index).cloned())
        .collect();
    if basic_objective.len() != constraint_count {
        return false;
    }
    let Ok(dual_solution) = solve_rational_system(
        &transpose_square_matrix(&current_basis_matrix),
        &basic_objective,
    ) else {
        return false;
    };
    let dual_is_feasible = (0..variable_count).all(|candidate| {
        rational_dot(&dual_solution, &matrix_column(matrix, candidate)).is_ok_and(|contribution| {
            objective[candidate].clone() - contribution <= rational_zero()
        })
    });
    dual_is_feasible
        && rational_dot(rhs, &dual_solution).is_ok_and(|value| value <= rational_zero())
}

/// Computes one exact projection along a finite candidate axis.
fn exact_projection<const D: usize>(coordinates: &[f64; D], axis: &[BigRational]) -> BigRational {
    coordinates
        .iter()
        .zip(axis)
        .fold(rational_zero(), |sum, (coordinate, coefficient)| {
            sum + rational_from_f64(*coordinate) * coefficient
        })
}

/// Certifies exact primal feasibility and dual optimality for one candidate basis.
fn certify_optimal_basis(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
    objective: &[BigRational],
    entering_variable_count: usize,
    basis: &[usize],
) -> Option<RevisedSimplexResult> {
    let current_basis_matrix = basis_matrix(matrix, basis)?;
    let basic_solution = solve_rational_system(&current_basis_matrix, rhs).ok()?;
    if basic_solution.iter().any(Signed::is_negative) {
        return None;
    }

    let basic_objective: Vec<_> = basis
        .iter()
        .map(|&index| objective[index].clone())
        .collect();
    let dual_solution = solve_rational_system(
        &transpose_square_matrix(&current_basis_matrix),
        &basic_objective,
    )
    .ok()?;
    let dual_is_feasible = (0..entering_variable_count)
        .filter(|candidate| !basis.contains(candidate))
        .all(|candidate| {
            rational_dot(&dual_solution, &matrix_column(matrix, candidate)).is_ok_and(
                |contribution| !(objective[candidate].clone() - contribution).is_positive(),
            )
        });
    if !dual_is_feasible {
        return None;
    }

    let mut solution = vec![rational_zero(); objective.len()];
    for (&variable, value) in basis.iter().zip(&basic_solution) {
        solution[variable] = value.clone();
    }
    Some(RevisedSimplexResult::Optimal {
        objective_value: rational_dot(objective, &solution).ok()?,
        solution,
    })
}

/// Explicitly rounds exact rationals to finite f64 values for provisional calculations.
fn rationals_to_f64(values: &[BigRational]) -> Option<Vec<f64>> {
    values
        .iter()
        .map(|value| value.to_rounded_f64().ok())
        .collect()
}

/// Extracts a floating-point basis matrix.
fn basis_matrix_f64(matrix: &[Vec<f64>], basis: &[usize]) -> Option<Vec<Vec<f64>>> {
    let column_count = matrix.first()?.len();
    if matrix.len() != basis.len()
        || matrix.iter().any(|row| row.len() != column_count)
        || basis.iter().any(|&column| column >= column_count)
    {
        return None;
    }
    Some(
        matrix
            .iter()
            .map(|row| basis.iter().map(|&column| row[column]).collect())
            .collect(),
    )
}

/// Copies one floating-point matrix column.
fn matrix_column_f64(matrix: &[Vec<f64>], column: usize) -> Vec<f64> {
    matrix.iter().map(|row| row[column]).collect()
}

/// Transposes a square floating-point matrix.
fn transpose_square_matrix_f64(matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let dimension = matrix.len();
    (0..dimension)
        .map(|row| (0..dimension).map(|column| matrix[column][row]).collect())
        .collect()
}

/// Adapts runtime LP vectors to the backend's checked dot product.
///
/// LP constraint vectors fit the exact basis dispatch limit. Zero padding
/// preserves the dot product without introducing a second arithmetic kernel.
/// Rejection abandons provisional navigation and leaves certification to the
/// exact path.
fn dot_product_f64(left: &[f64], right: &[f64]) -> Option<f64> {
    if left.len() != right.len() || left.len() > MAX_LP_BASIS_DIMENSION {
        return None;
    }
    if left.len() <= 8 {
        padded_provisional_dot::<8>(left, right)
    } else {
        padded_provisional_dot::<MAX_LP_BASIS_DIMENSION>(left, right)
    }
}

/// Preserves the established small-system cost while admitting D=7/8 LP rows.
fn padded_provisional_dot<const N: usize>(left: &[f64], right: &[f64]) -> Option<f64> {
    let left =
        LaVector::<N>::try_new(from_fn(|index| left.get(index).copied().unwrap_or(0.0))).ok()?;
    let right =
        LaVector::try_new(from_fn(|index| right.get(index).copied().unwrap_or(0.0))).ok()?;
    left.dot(&right).ok()
}

/// Admits a runtime LP basis for provisional LU solving in `la-stack`.
///
/// D=6..=8 programs need orders 8..=10 beyond binary64 runtime dispatch.
/// The public const-generic LU supports them; each candidate still needs exact certification.
fn solve_f64_system(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let dimension = rhs.len();
    if matrix.len() != dimension || matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }
    match dimension {
        8 => return solve_provisional_system(Matrix::<8>::zero(), matrix, rhs).ok(),
        9 => return solve_provisional_system(Matrix::<9>::zero(), matrix, rhs).ok(),
        10 => return solve_provisional_system(Matrix::<10>::zero(), matrix, rhs).ok(),
        _ => {}
    }
    la_stack::try_with_stack_matrix!(dimension, |stack_matrix| -> Result<_, LaError> {
        solve_provisional_system(stack_matrix, matrix, rhs)
    })
    .ok()
}

/// Converts an admitted basis and supplies the LP's conservative pivot policy.
fn solve_provisional_system<const N: usize>(
    mut stack_matrix: Matrix<N>,
    matrix: &[Vec<f64>],
    rhs: &[f64],
) -> Result<Vec<f64>, LaError> {
    let mut scale = 1.0_f64;
    for (row, values) in matrix.iter().enumerate() {
        for (column, &value) in values.iter().enumerate() {
            stack_matrix.set(row, column, value)?;
            scale = scale.max(value.abs());
        }
    }
    let rhs = LaVector::try_new(from_fn(|index| rhs[index]))?;
    // LU accepts an absolute threshold. Scale it by the original coefficient
    // maximum; rejecting a small pivot selects exact navigation, never a
    // geometric singularity classification.
    let tolerance = Tolerance::try_new(1.0e-14 * scale)?;
    stack_matrix
        .lu(tolerance)?
        .solve(rhs)
        .map(|solution| solution.into_array().to_vec())
}

/// Extracts the square matrix formed by the current basis columns.
fn basis_matrix(matrix: &[Vec<BigRational>], basis: &[usize]) -> Option<Vec<Vec<BigRational>>> {
    let column_count = matrix.first()?.len();
    if matrix.len() != basis.len()
        || matrix.iter().any(|row| row.len() != column_count)
        || basis.iter().any(|&column| column >= column_count)
    {
        return None;
    }
    Some(
        matrix
            .iter()
            .map(|row| basis.iter().map(|&column| row[column].clone()).collect())
            .collect(),
    )
}

/// Copies one matrix column for an exact linear solve.
fn matrix_column(matrix: &[Vec<BigRational>], column: usize) -> Vec<BigRational> {
    matrix.iter().map(|row| row[column].clone()).collect()
}

/// Transposes a square exact-rational matrix.
fn transpose_square_matrix(matrix: &[Vec<BigRational>]) -> Vec<Vec<BigRational>> {
    let dimension = matrix.len();
    (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| matrix[column][row].clone())
                .collect()
        })
        .collect()
}

/// Returns labels whose barycentric coordinates witness mass outside the shared face.
fn positive_nonshared_labels<L>(
    barycentric: &[BigRational],
    labels: &[L],
    shared_labels: &[L],
) -> SimplexRealizationBuffer<L>
where
    L: Clone + Eq,
{
    labels
        .iter()
        .zip(barycentric)
        .filter(|(label, coordinate)| !shared_labels.contains(label) && coordinate.is_positive())
        .map(|(label, _coordinate)| label.clone())
        .collect()
}

/// Solves an exact LP basis without rounding derived coefficients through binary64.
/// Admission is explicit through order ten (D+2 for D=8). Canonical rational
/// storage, pivoting, elimination and singularity remain owned by la-stack.
fn solve_rational_system(
    matrix: &[Vec<BigRational>],
    rhs: &[BigRational],
) -> Result<Vec<BigRational>, StackMatrixDispatchError> {
    let dimension = rhs.len();
    if dimension > MAX_LP_BASIS_DIMENSION {
        return Err(StackMatrixDispatchError::UnsupportedDim {
            k: dimension,
            max: MAX_LP_BASIS_DIMENSION,
        });
    }
    if matrix.len() != dimension {
        return Err(StackMatrixDispatchError::ActiveBlockDimensionMismatch {
            k: matrix.len(),
            dim: dimension,
        });
    }
    if let Some(row) = matrix.iter().find(|row| row.len() != dimension) {
        return Err(StackMatrixDispatchError::ActiveBlockDimensionMismatch {
            k: row.len(),
            dim: dimension,
        });
    }
    match dimension {
        9 => solve_exact_basis(RationalMatrix::<9>::zero(), matrix, rhs),
        10 => solve_exact_basis(RationalMatrix::<10>::zero(), matrix, rhs),
        _ => la_stack::try_with_rational_matrix!(dimension, |exact_matrix| -> Result<_, LaError> {
            solve_exact_basis(exact_matrix, matrix, rhs)
        }),
    }
    .map_err(Into::into)
}

/// Fills a shape-admitted exact matrix using only exact rational coefficients.
fn solve_exact_basis<const N: usize>(
    mut matrix: RationalMatrix<N>,
    rows: &[Vec<BigRational>],
    rhs: &[BigRational],
) -> Result<Vec<BigRational>, LaError> {
    for (row, values) in rows.iter().enumerate() {
        for (column, value) in values.iter().enumerate() {
            matrix.set(row, column, value.clone())?;
        }
    }
    let rhs = RationalVector::try_from_fn(|index| rhs[index].clone())?;
    matrix
        .solve(&rhs)
        .map(|solution| solution.into_array().into_iter().collect())
}

/// Converts a finite floating-point matrix to its exact rational representation.
fn exact_matrix_from_f64(matrix: &[Vec<f64>]) -> Vec<Vec<BigRational>> {
    matrix
        .iter()
        .map(|row| row.iter().copied().map(rational_from_f64).collect())
        .collect()
}

/// Converts a finite f64 to an exact rational value for barycentric predicates.
fn rational_from_f64(value: f64) -> BigRational {
    BigRational::from_f64(value)
        .expect("validated finite f64 coordinates must convert to BigRational")
}

/// Returns the additive identity used throughout exact barycentric arithmetic.
fn rational_zero() -> BigRational {
    BigRational::from_integer(BigInt::from(0))
}

/// Returns the multiplicative identity used throughout exact barycentric arithmetic.
fn rational_one() -> BigRational {
    BigRational::from_integer(BigInt::from(1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::realization::{SimplexIntersectionFailure, validate_simplex_intersection};
    use approx::assert_abs_diff_eq;
    use proptest::prelude::*;
    use std::assert_matches;

    /// Classification shared by the optimized predicate and legacy exact oracle.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum IntersectionClassification {
        Valid,
        Invalid,
        SingularBarycentricBasis,
    }

    /// Select non-adjacent tableau columns, force pivoting, and recover a known
    /// non-dyadic solution. Re-multiplication is independent of elimination.
    fn assert_rational_basis_round_trip(dimension: usize) {
        let mut tableau = vec![vec![rational_zero(); 2 * dimension]; dimension];
        for (row, coefficients) in tableau.iter_mut().enumerate() {
            for column in row..dimension {
                coefficients[2 * column + 1] =
                    BigRational::new(BigInt::from(row + column + 1), BigInt::from(3 + 2 * row));
            }
        }
        tableau.rotate_left(1);
        let basis: Vec<_> = (0..dimension).map(|column| 2 * column + 1).collect();
        let matrix = basis_matrix(&tableau, &basis).unwrap();
        let expected: Vec<_> = (1..=dimension)
            .map(|value| BigRational::new(BigInt::from(value), BigInt::from(11)))
            .collect();
        let rhs: Vec<_> = matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&expected)
                    .map(|(a, b)| a * b)
                    .sum::<BigRational>()
            })
            .collect();
        assert!(
            matrix
                .iter()
                .flatten()
                .any(|value| value.try_to_f64().is_err())
        );
        let solution = solve_rational_system(&matrix, &rhs).unwrap();
        assert_eq!(solution, expected);
        let recovered: Vec<_> = matrix
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&solution)
                    .map(|(a, b)| a * b)
                    .sum::<BigRational>()
            })
            .collect();
        assert_eq!(recovered, rhs);
    }

    /// Builds the standard D-simplex with vertices at the origin and coordinate axes.
    fn standard_simplex_coordinates<const D: usize>() -> Vec<[f64; D]> {
        let mut coordinates = Vec::with_capacity(D + 1);
        coordinates.push([0.0; D]);
        for axis in 0..D {
            let mut vertex = [0.0; D];
            vertex[axis] = 1.0;
            coordinates.push(vertex);
        }
        coordinates
    }

    /// Collects the complete vertex set for the independent classification oracle.
    fn intersection_polytope_vertices(coordinates: &[Vec<BigRational>]) -> Vec<Vec<BigRational>> {
        let mut vertices = Vec::new();
        visit_intersection_vertices(coordinates, MAX_ACTIVE_SETS, |beta| {
            vertices.push(beta.to_vec());
            ControlFlow::Continue(())
        })
        .unwrap();
        vertices
    }

    /// Reconstructs the pre-optimization exact active-set decision for agreement tests.
    fn legacy_active_set_classification<L, const D: usize>(
        first: &LabeledSimplexRealization<L, D>,
        second: &LabeledSimplexRealization<L, D>,
        shared_labels: &[L],
    ) -> IntersectionClassification
    where
        L: Eq,
    {
        let Ok(second_vertices_in_first) = barycentric_coordinates_of_vertices(second, first)
        else {
            return IntersectionClassification::SingularBarycentricBasis;
        };

        for beta in intersection_polytope_vertices(&second_vertices_in_first) {
            let alpha = alpha_from_beta(&beta, &second_vertices_in_first);
            let first_has_nonshared_weight = first
                .labels()
                .iter()
                .zip(&alpha)
                .any(|(label, weight)| !shared_labels.contains(label) && weight.is_positive());
            let second_has_nonshared_weight = second
                .labels()
                .iter()
                .zip(&beta)
                .any(|(label, weight)| !shared_labels.contains(label) && weight.is_positive());
            if first_has_nonshared_weight || second_has_nonshared_weight {
                return IntersectionClassification::Invalid;
            }
        }

        IntersectionClassification::Valid
    }

    /// Exercises the 7D/8D public boundary and both exact certification routes.
    fn assert_high_dimension_intersections<const D: usize>() {
        let coordinates = standard_simplex_coordinates::<D>();
        let first = LabeledSimplexRealization::try_new(0..=D, coordinates.clone()).unwrap();
        let coincident =
            LabeledSimplexRealization::try_new(D + 1..2 * D + 2, coordinates.clone()).unwrap();
        let error = validate_simplex_intersection(&first, &coincident).unwrap_err();
        assert_matches!(error, SimplexIntersectionFailure::IntersectionOutsideSharedFace { witness, .. }
            if witness.shared.is_empty() && !witness.first_only_witness.is_empty()
                && !witness.second_only_witness.is_empty());
        assert_matches!(
            intersection_via_linear_program(&first, &coincident, &[], true),
            IntersectionLinearProgramResult::Invalid(_)
        );
        assert_matches!(
            intersection_via_active_sets(&first, &coincident, &[], None),
            IntersectionLinearProgramResult::Invalid(_)
        );
        let interior = LabeledSimplexRealization::try_new(
            D + 1..2 * D + 2,
            coordinates
                .iter()
                .map(|point| point.map(|x| x.mul_add(0.5, 1.0 / 64.0))),
        )
        .unwrap();
        assert_matches!(
            validate_simplex_intersection(&first, &interior),
            Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace { .. })
        );
        let disjoint = LabeledSimplexRealization::try_new(
            D + 1..2 * D + 2,
            coordinates.iter().map(|point| point.map(|x| x + 3.0)),
        )
        .unwrap();
        assert!(validate_simplex_intersection(&first, &disjoint).is_ok());
        let collapsed = LabeledSimplexRealization::try_new(0..=D, vec![[0.0; D]; D + 1]).unwrap();
        assert_matches!(
            validate_simplex_intersection(&collapsed, &collapsed),
            Err(SimplexIntersectionFailure::SingularBarycentricBasis)
        );
        let collapsed_second =
            LabeledSimplexRealization::try_new((0..1).chain(D + 1..=2 * D), vec![[0.0; D]; D + 1])
                .unwrap();
        assert_matches!(
            validate_simplex_intersection(&first, &collapsed_second),
            Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace { .. })
        );
        for shared_count in [1, D - 1, D] {
            let (left, right, _) = boundary_degenerate_pair::<D>(shared_count, 1.0);
            assert!(validate_simplex_intersection(&left, &right).is_ok());
        }
    }

    /// Classifies one pair through the public optimized Level 4 predicate.
    fn optimized_classification<L, const D: usize>(
        first: &LabeledSimplexRealization<L, D>,
        second: &LabeledSimplexRealization<L, D>,
    ) -> IntersectionClassification
    where
        L: Clone + Eq,
    {
        match validate_simplex_intersection(first, second) {
            Ok(()) => IntersectionClassification::Valid,
            Err(SimplexIntersectionFailure::UnsupportedDimension { dimension, max }) => {
                panic!("agreement fixture dimension {dimension} exceeds exact support {max}");
            }
            Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace { .. }) => {
                IntersectionClassification::Invalid
            }
            Err(SimplexIntersectionFailure::SingularBarycentricBasis) => {
                IntersectionClassification::SingularBarycentricBasis
            }
            Err(SimplexIntersectionFailure::SolverFailed { source }) => {
                panic!("solver failed: {source:?}")
            }
            Err(SimplexIntersectionFailure::ActiveSetBudgetExceeded { limit }) => {
                panic!("budget exhausted: {limit}")
            }
        }
    }

    /// Builds a random-grid pair while preserving coordinates for shared labels.
    fn random_grid_pair<const D: usize>(
        raw_second_coordinates: &[i16],
        shared_count: usize,
    ) -> (
        LabeledSimplexRealization<usize, D>,
        LabeledSimplexRealization<usize, D>,
        Vec<usize>,
    ) {
        let first_coordinates = standard_simplex_coordinates::<D>();
        let mut second_coordinates = Vec::with_capacity(D + 1);
        for (vertex_index, first_coordinates) in first_coordinates.iter().enumerate() {
            if vertex_index < shared_count {
                second_coordinates.push(*first_coordinates);
                continue;
            }

            let mut coordinates = [0.0; D];
            let start = vertex_index * D;
            for (axis, coordinate) in coordinates.iter_mut().enumerate() {
                *coordinate = f64::from(raw_second_coordinates[start + axis]);
            }
            second_coordinates.push(coordinates);
        }

        let first = LabeledSimplexRealization::try_new(0..=D, first_coordinates)
            .expect("standard simplex is valid");
        let second_labels = (0..=D).map(|vertex_index| {
            if vertex_index < shared_count {
                vertex_index
            } else {
                D + 1 + vertex_index
            }
        });
        let second = LabeledSimplexRealization::try_new(second_labels, second_coordinates)
            .expect("generated labels and finite coordinates are valid");
        let shared_labels = (0..shared_count).collect();
        (first, second, shared_labels)
    }

    /// Builds nondegenerate simplices whose intersection is exactly a lower-dimensional shared face.
    fn boundary_degenerate_pair<const D: usize>(
        shared_count: usize,
        transverse_scale: f64,
    ) -> (
        LabeledSimplexRealization<usize, D>,
        LabeledSimplexRealization<usize, D>,
        Vec<usize>,
    ) {
        let first_coordinates = standard_simplex_coordinates::<D>();
        let mut second_coordinates = first_coordinates[..shared_count].to_vec();
        for axis in shared_count.saturating_sub(1)..D {
            let mut coordinates = [0.0; D];
            coordinates[axis] = -transverse_scale;
            second_coordinates.push(coordinates);
        }

        let first = LabeledSimplexRealization::try_new(0..=D, first_coordinates)
            .expect("standard simplex is valid");
        let second_labels = (0..shared_count).chain(D + 1..2 * D + 2 - shared_count);
        let second = LabeledSimplexRealization::try_new(second_labels, second_coordinates)
            .expect("boundary-degenerate simplex is valid");
        let shared_labels = (0..shared_count).collect();
        (first, second, shared_labels)
    }

    /// Verifies optimized and legacy exact classifications for one pair.
    fn assert_optimized_agrees_with_legacy<const D: usize>(
        first: &LabeledSimplexRealization<usize, D>,
        second: &LabeledSimplexRealization<usize, D>,
        shared_labels: &[usize],
    ) {
        assert_eq!(
            optimized_classification(first, second),
            legacy_active_set_classification(first, second, shared_labels),
            "optimized classifier disagreed with the legacy exact active-set oracle"
        );
        assert_eq!(
            optimized_classification(second, first),
            legacy_active_set_classification(second, first, shared_labels),
            "optimized classifier disagreed with the legacy exact active-set oracle after swapping simplex order"
        );
    }

    /// Checks the exact-degeneracy regression where non-shared vertices collapse
    /// onto the only shared vertex.
    fn assert_collapsed_simplex_agreement<const D: usize>() {
        let raw_second_coordinates = vec![0; (D + 1) * D];
        let (first, second, shared_labels) = random_grid_pair::<D>(&raw_second_coordinates, 1);
        assert_optimized_agrees_with_legacy(&first, &second, &shared_labels);
    }

    /// Builds two D-simplices that meet exactly in the facet opposite the last axis.
    fn shared_facet_pair<const D: usize>() -> (
        LabeledSimplexRealization<usize, D>,
        LabeledSimplexRealization<usize, D>,
        Vec<usize>,
    ) {
        let first_coordinates = standard_simplex_coordinates::<D>();
        let mut second_coordinates = first_coordinates[..D].to_vec();
        let mut opposite_apex = [0.0; D];
        opposite_apex[D - 1] = -1.0;
        second_coordinates.push(opposite_apex);

        let first = LabeledSimplexRealization::try_new(0..=D, first_coordinates)
            .expect("standard simplex is valid");
        let second = LabeledSimplexRealization::try_new(
            (0..D).chain(std::iter::once(D + 1)),
            second_coordinates,
        )
        .expect("opposite simplex is valid");
        (first, second, (0..D).collect())
    }

    /// Builds one D-simplex strictly inside another with disjoint labels.
    fn crossing_pair<const D: usize>() -> (
        LabeledSimplexRealization<usize, D>,
        LabeledSimplexRealization<usize, D>,
    ) {
        let first = LabeledSimplexRealization::try_new(0..=D, standard_simplex_coordinates::<D>())
            .expect("standard simplex is valid");
        let dimension = f64::from(u32::try_from(D + 1).expect("tested dimensions fit in u32"));
        let step = 1.0 / (4.0 * dimension);
        let base = [step; D];
        let mut second_coordinates = Vec::with_capacity(D + 1);
        second_coordinates.push(base);
        for axis in 0..D {
            let mut vertex = base;
            vertex[axis] += step;
            second_coordinates.push(vertex);
        }
        let second = LabeledSimplexRealization::try_new(D + 1..=2 * D + 1, second_coordinates)
            .expect("interior simplex is valid");
        (first, second)
    }

    /// Checks the shared-facet classification and exact affine-nullspace certificate.
    fn assert_shared_facet_is_valid<const D: usize>() {
        let (first, second, shared) = shared_facet_pair::<D>();
        assert_matches!(
            intersection_via_linear_program(&first, &second, &shared, false),
            IntersectionLinearProgramResult::Valid
        );
        assert_matches!(
            intersection_via_active_sets(&first, &second, &shared, None),
            IntersectionLinearProgramResult::Valid
        );

        let mut provisional_axis = vec![0.0; D];
        provisional_axis[D - 1] = 1.0;
        let exact_axis = exact_axis_through_shared_face(&first, &shared, &provisional_axis)
            .expect("shared facet has an exact normal");
        let shared_projection = exact_projection(&first.coordinates()[0], &exact_axis);
        for coordinates in &first.coordinates()[..D] {
            assert_eq!(
                exact_projection(coordinates, &exact_axis),
                shared_projection
            );
        }
        assert!(exact_shared_face_confinement_certificate(
            &first,
            &second,
            &shared,
            &provisional_axis,
        ));
    }

    /// Checks that a full-dimensional interior overlap produces witnesses on both sides.
    fn assert_crossing_is_invalid<const D: usize>() {
        let (first, second) = crossing_pair::<D>();
        assert_matches!(
            intersection_via_linear_program(&first, &second, &[], false),
            IntersectionLinearProgramResult::Invalid(witness)
                if !witness.first_only_witness.is_empty()
                    && !witness.second_only_witness.is_empty()
        );
    }

    /// Retains a finite unit direction even when its unnormalized squared length overflows.
    fn assert_large_separator_ray_is_normalized<const D: usize>() {
        let mut ray = [0.0; D];
        ray[0] = 3.0e200;
        ray[1] = -4.0e200;
        let mut rays = Vec::new();
        push_normalized_ray(&mut rays, ray);

        assert_eq!(rays.len(), 1);
        let normalized = &rays[0];
        assert_abs_diff_eq!(normalized[0], 0.6, epsilon = f64::EPSILON);
        assert_abs_diff_eq!(normalized[1], -0.8, epsilon = f64::EPSILON);
        for &coordinate in &normalized[2..] {
            assert_abs_diff_eq!(coordinate, 0.0, epsilon = f64::EPSILON);
        }
        let squared_length: f64 = normalized
            .iter()
            .map(|coordinate| coordinate * coordinate)
            .sum();
        assert_abs_diff_eq!(squared_length, 1.0, epsilon = 4.0 * f64::EPSILON);
    }

    /// Rejected provisional rays must not contaminate already collected directions.
    fn assert_unusable_separator_rays_are_skipped<const D: usize>() {
        let mut unit_ray = [0.0; D];
        unit_ray[0] = 1.0;
        let mut rays = vec![unit_ray];
        for ray in [[0.0; D], [f64::MAX; D], [f64::INFINITY; D], [f64::NAN; D]] {
            push_normalized_ray(&mut rays, ray);
            assert_eq!(rays.len(), 1);
            assert_eq!(rays[0].map(f64::to_bits), unit_ray.map(f64::to_bits));
        }
    }

    macro_rules! separator_ray_tests {
        ($($dim:literal),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<large_separator_ray_is_normalized_ $dim d>]() {
                        assert_large_separator_ray_is_normalized::<$dim>();
                    }

                    #[test]
                    fn [<unusable_separator_rays_are_skipped_ $dim d>]() {
                        assert_unusable_separator_rays_are_skipped::<$dim>();
                    }
                )+
            }
        };
    }

    macro_rules! generate_intersection_agreement_tests {
        ($dim:literal, $random_test:ident, $boundary_test:ident) => {
            generate_intersection_agreement_tests!($dim, $random_test, $boundary_test, 16);
        };
        ($dim:literal, $random_test:ident, $boundary_test:ident, $cases:literal) => {
            proptest! {
                #![proptest_config(ProptestConfig {
                    cases: $cases,
                    ..ProptestConfig::default()
                })]

                #[test]
                fn $random_test(
                    raw_second_coordinates in proptest::collection::vec(
                        -4_i16..=4,
                        ($dim + 1) * $dim,
                    ),
                    shared_count in 0_usize..=$dim + 1,
                ) {
                    let (first, second, shared_labels) = random_grid_pair::<$dim>(
                        &raw_second_coordinates,
                        shared_count,
                    );
                    assert_optimized_agrees_with_legacy(&first, &second, &shared_labels);
                }

                #[test]
                fn $boundary_test(
                    shared_count in 1_usize..$dim,
                    scale_exponent in 0_u8..=40,
                ) {
                    let transverse_scale = 2.0_f64.powi(-i32::from(scale_exponent));
                    let (first, second, shared_labels) = boundary_degenerate_pair::<$dim>(
                        shared_count,
                        transverse_scale,
                    );
                    assert_optimized_agrees_with_legacy(&first, &second, &shared_labels);
                }
            }
        };
    }

    separator_ray_tests!(2, 3, 4, 5);

    #[test]
    fn provisional_lp_bases_preserve_pivoting_and_scale_through_dimension_eight() {
        // An upper bidiagonal matrix with diagonal 2 and superdiagonal 1
        // maps the all-ones vector to [3, ..., 3, 2]. Rotating its rows
        // forces pivoting without changing that analytically known solution.
        for dimension in 2..=10 {
            for scale in [1.0, 1.0e150] {
                let mut matrix = vec![vec![0.0; dimension]; dimension];
                let mut rhs = vec![3.0 * scale; dimension];
                for (index, row) in matrix.iter_mut().enumerate() {
                    row[index] = 2.0 * scale;
                    if index + 1 < dimension {
                        row[index + 1] = scale;
                    }
                }
                rhs[dimension - 1] = 2.0 * scale;
                matrix.rotate_left(1);
                rhs.rotate_left(1);
                let solution = solve_f64_system(&matrix, &rhs).unwrap();
                assert_eq!(solution.len(), dimension);
                for value in solution {
                    assert_abs_diff_eq!(value, 1.0, epsilon = 1.0e-14);
                }
            }
        }
    }

    #[test]
    fn provisional_lp_basis_rejects_invalid_and_unusable_systems() {
        assert_matches!(solve_f64_system(&[], &[]).as_deref(), Some([]));
        assert!(solve_f64_system(&[vec![1.0]], &[]).is_none());
        assert!(solve_f64_system(&[vec![]], &[1.0]).is_none());
        assert!(solve_f64_system(&[vec![0.0]], &[1.0]).is_none());
        assert!(solve_f64_system(&vec![vec![0.0; 9]; 9], &[0.0; 9]).is_none());
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(solve_f64_system(&[vec![invalid]], &[1.0]).is_none());
            assert!(solve_f64_system(&[vec![1.0]], &[invalid]).is_none());
        }
        assert!(solve_f64_system(&[vec![0.5]], &[f64::MAX]).is_none());
    }

    #[test]
    fn exact_simplex_recovers_when_provisional_lu_rejects_a_small_pivot() {
        let delta = 2.0_f64.powi(-48);
        let matrix = vec![vec![1.0, 1.0], vec![1.0, 1.0 + delta]];
        let rhs = [2.0, 2.0 + delta];
        assert!(solve_f64_system(&matrix, &rhs).is_none());
        let matrix = exact_matrix_from_f64(&matrix);
        let rhs: Vec<_> = rhs.into_iter().map(rational_from_f64).collect();
        let objective = [rational_one(), rational_zero()];
        let mut basis = [0, 1];
        assert_matches!(
            revised_simplex_maximize(&matrix, &rhs, &objective, 2, &mut basis),
            RevisedSimplexResult::Optimal { solution, objective_value }
                if solution == vec![rational_one(), rational_one()]
                    && objective_value == rational_one()
        );
    }

    #[test]
    fn provisional_dot_product_admits_only_matching_finite_lp_vectors() {
        assert_abs_diff_eq!(dot_product_f64(&[], &[]).unwrap(), 0.0);
        for dimension in 1..=10 {
            let mut right = vec![0.0; dimension];
            right[dimension - 1] = -3.0;
            assert_abs_diff_eq!(
                dot_product_f64(&vec![2.0; dimension], &right).unwrap(),
                -6.0
            );
        }
        assert!(dot_product_f64(&[1.0], &[]).is_none());
        assert!(dot_product_f64(&[1.0; 11], &[1.0; 11]).is_none());
        assert!(dot_product_f64(&[f64::NAN], &[1.0]).is_none());
        assert!(dot_product_f64(&[1.0], &[f64::INFINITY]).is_none());
        assert!(dot_product_f64(&[f64::MAX], &[2.0]).is_none());
    }

    #[test]
    fn rational_lp_bases_round_trip_through_dimension_ten() {
        // Geometric D <= 8 uses LP systems as large as D + 2.
        for dimension in 1..=10 {
            assert_rational_basis_round_trip(dimension);
        }
    }

    #[test]
    fn rational_lp_basis_rejects_singular_malformed_and_unsupported_systems() {
        let one = rational_one();
        let zero = rational_zero();
        assert_eq!(solve_rational_system(&[], &[]), Ok(vec![]));
        assert_matches!(
            solve_rational_system(&[vec![one.clone()]], &[]),
            Err(StackMatrixDispatchError::ActiveBlockDimensionMismatch { .. })
        );
        assert_matches!(
            solve_rational_system(&[vec![]], std::slice::from_ref(&one)),
            Err(StackMatrixDispatchError::ActiveBlockDimensionMismatch { .. })
        );
        for dimension in [1, 9, 10] {
            assert_matches!(
                solve_rational_system(
                    &vec![vec![zero.clone(); dimension]; dimension],
                    &vec![one.clone(); dimension]
                ),
                Err(StackMatrixDispatchError::La {
                    source: LaError::Singular { .. }
                })
            );
        }
        assert_matches!(
            solve_rational_system(&vec![vec![zero.clone(); 11]; 11], &vec![zero; 11]),
            Err(StackMatrixDispatchError::UnsupportedDim { k: 11, max: 10 })
        );
    }

    #[test]
    fn rational_high_order_solves_preserve_large_exponents() {
        for dimension in [9, 10] {
            let mut matrix = vec![vec![rational_zero(); dimension]; dimension];
            let expected: Vec<_> = (0..dimension)
                .map(|i| BigRational::new(BigInt::from(i + 1), BigInt::from(3)))
                .collect();
            for (i, row) in matrix.iter_mut().enumerate() {
                row[i] = if i.is_multiple_of(2) {
                    BigRational::from_integer(BigInt::from(1) << 2048)
                } else {
                    BigRational::new(BigInt::from(1), BigInt::from(1) << 2048)
                };
            }
            matrix.rotate_left(1);
            let rhs: Vec<BigRational> = matrix
                .iter()
                .map(|row| row.iter().zip(&expected).map(|(a, b)| a * b).sum())
                .collect();
            let solution = solve_rational_system(&matrix, &rhs).unwrap();
            assert_eq!(solution, expected);
            for (row, rhs) in matrix.iter().zip(rhs) {
                assert_eq!(
                    row.iter()
                        .zip(&solution)
                        .map(|(a, b)| a * b)
                        .sum::<BigRational>(),
                    rhs
                );
            }
        }
    }

    #[test]
    fn intersections_through_seven_dimensions() {
        assert_high_dimension_intersections::<7>();
        assert_shared_facet_is_valid::<7>();
    }

    #[test]
    fn intersections_through_eight_dimensions() {
        assert_high_dimension_intersections::<8>();
        assert_shared_facet_is_valid::<8>();
    }

    #[test]
    fn shared_face_basis_selection_is_provisional_and_shape_checked() {
        for scale in [1.0, 1.0e100] {
            let rows = vec![vec![0.0, scale, 2.0 * scale], vec![scale, 0.0, scale]];
            let columns = shared_face_pivot_columns(&rows).unwrap();
            assert_eq!(columns, vec![0, 1]);
            let matrix = basis_matrix(&exact_matrix_from_f64(&rows), &columns).unwrap();
            assert!(solve_rational_system(&matrix, &[rational_one(), rational_one()]).is_ok());
            let reversed = vec![rows[1].clone(), rows[0].clone()];
            assert_eq!(shared_face_pivot_columns(&reversed), Some(columns));
        }
        // An invertible exact matrix can be numerically rejected by the policy.
        let near = vec![vec![1.0, 1.0], vec![1.0, 1.0 + 2.0_f64.powi(-48)]];
        assert!(shared_face_pivot_columns(&near).is_none());
        assert!(
            solve_rational_system(
                &exact_matrix_from_f64(&near),
                &[rational_one(), rational_one()]
            )
            .is_ok()
        );
        assert!(shared_face_pivot_columns(&[vec![1.0e-100]]).is_none());
        for invalid in [f64::NAN, f64::INFINITY] {
            assert!(shared_face_pivot_columns(&[vec![1.0, invalid]]).is_none());
        }
        assert!(shared_face_pivot_columns(&[vec![1.0], vec![]]).is_none());
        assert!(shared_face_pivot_columns(&[vec![1.0; 10]]).is_none());
    }

    #[test]
    fn exact_objective_dot_preserves_cancellation_and_full_lp_lengths() {
        let third = BigRational::new(1.into(), 3.into());
        let huge = BigRational::from_integer(BigInt::from(1) << 2048);
        // D=8 Phase I/II objectives have 2(D+1)+(D+2)=28 entries.
        let mut left = vec![third.clone(); 28];
        left[0] = huge.clone();
        left[27] = -huge;
        let right = vec![rational_one(); 28];
        assert_eq!(
            rational_dot(&left, &right).unwrap(),
            third * BigRational::from_integer(26.into())
        );
        assert_eq!(rational_dot(&[], &[]).unwrap(), rational_zero());
        assert_matches!(
            rational_dot(&left, &right[..27]),
            Err(LaError::DimensionMismatch {
                left: 28,
                right: 27,
                ..
            })
        );
    }

    #[test]
    fn active_set_failure_never_becomes_empty_intersection() {
        let coordinates = vec![
            vec![rational_one(), rational_zero()],
            vec![rational_zero(), rational_one()],
        ];
        assert_matches!(
            visit_intersection_vertices(&coordinates, 0, |_| ControlFlow::Continue(())),
            Err(ActiveSetFailure::Budget { limit: 0 })
        );
        let unsupported = vec![vec![rational_zero(); 11]; 11];
        assert_matches!(
            visit_intersection_vertices(&unsupported, MAX_ACTIVE_SETS, |_| ControlFlow::Continue(
                ()
            )),
            Err(ActiveSetFailure::Solver {
                source: StackMatrixDispatchError::UnsupportedDim { k: 11, max: 10 }
            })
        );
    }

    generate_intersection_agreement_tests!(
        2,
        optimized_intersection_agrees_with_legacy_random_2d,
        optimized_intersection_agrees_with_legacy_boundary_degenerate_2d
    );
    generate_intersection_agreement_tests!(
        3,
        optimized_intersection_agrees_with_legacy_random_3d,
        optimized_intersection_agrees_with_legacy_boundary_degenerate_3d
    );
    generate_intersection_agreement_tests!(
        4,
        optimized_intersection_agrees_with_legacy_random_4d,
        optimized_intersection_agrees_with_legacy_boundary_degenerate_4d
    );
    generate_intersection_agreement_tests!(
        5,
        optimized_intersection_agrees_with_legacy_random_5d,
        optimized_intersection_agrees_with_legacy_boundary_degenerate_5d
    );
    // Bound the larger exact active-set oracle while extending agreement to D=6.
    generate_intersection_agreement_tests!(
        6,
        optimized_intersection_agrees_with_legacy_random_6d,
        optimized_intersection_agrees_with_legacy_boundary_degenerate_6d,
        4
    );

    #[test]
    fn optimized_intersection_agrees_with_legacy_for_collapsed_simplices_2d_to_5d() {
        assert_collapsed_simplex_agreement::<2>();
        assert_collapsed_simplex_agreement::<3>();
        assert_collapsed_simplex_agreement::<4>();
        assert_collapsed_simplex_agreement::<5>();
    }

    #[test]
    fn exact_linear_program_classifies_disjoint_triangles_as_valid() {
        let first =
            LabeledSimplexRealization::try_new([0, 1, 2], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
                .unwrap();
        let second =
            LabeledSimplexRealization::try_new([3, 4, 5], [[2.0, 2.0], [3.0, 2.0], [2.0, 3.0]])
                .unwrap();

        assert_matches!(
            intersection_via_linear_program(&first, &second, &[], false),
            IntersectionLinearProgramResult::Valid
        );
    }

    #[test]
    fn exact_linear_program_accepts_a_shared_edge_intersection_2d() {
        assert_shared_facet_is_valid::<2>();
    }

    #[test]
    fn exact_linear_program_reports_crossing_triangle_witnesses_2d() {
        assert_crossing_is_invalid::<2>();
    }

    #[test]
    fn exact_linear_program_accepts_a_shared_facet_intersection_3d() {
        assert_shared_facet_is_valid::<3>();
    }

    #[test]
    fn exact_linear_program_reports_crossing_tetrahedra_3d() {
        assert_crossing_is_invalid::<3>();
    }

    #[test]
    fn exact_linear_program_accepts_a_shared_facet_intersection_4d() {
        assert_shared_facet_is_valid::<4>();
    }

    #[test]
    fn exact_linear_program_reports_crossing_simplices_4d() {
        assert_crossing_is_invalid::<4>();
    }

    #[test]
    fn exact_linear_program_accepts_a_shared_facet_intersection_5d() {
        assert_shared_facet_is_valid::<5>();
    }

    #[test]
    fn exact_linear_program_accepts_a_shared_facet_intersection_6d() {
        assert_shared_facet_is_valid::<6>();
    }

    #[test]
    fn exact_linear_program_reports_crossing_simplices_5d() {
        assert_crossing_is_invalid::<5>();
    }

    #[test]
    fn exact_linear_program_reports_a_singular_verified_basis() {
        let first =
            LabeledSimplexRealization::try_new([0, 1, 2], [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
                .unwrap();
        let second =
            LabeledSimplexRealization::try_new([3, 4, 5], [[0.0, 1.0], [1.0, 1.0], [0.0, 2.0]])
                .unwrap();

        assert_matches!(
            intersection_via_linear_program(&first, &second, &[], true),
            IntersectionLinearProgramResult::SingularBarycentricBasis
        );
    }

    #[test]
    fn exact_linear_program_runs_phase_one_and_phase_two() {
        let matrix = vec![vec![rational_one(), rational_one()]];
        let rhs = vec![rational_one()];
        let objective = vec![rational_one(), rational_zero()];

        assert_matches!(
            maximize_nonnegative_equality_linear_program(&matrix, &rhs, &objective),
            ExactLinearProgramResult::Optimal {
                objective_value,
                ..
            } if objective_value == rational_one()
        );
    }

    #[test]
    fn exact_linear_program_reports_phase_one_infeasibility() {
        let matrix = vec![vec![rational_one()], vec![rational_one()]];
        let rhs = vec![rational_one(), rational_one() + rational_one()];
        let objective = vec![rational_one()];

        assert_matches!(
            maximize_nonnegative_equality_linear_program(&matrix, &rhs, &objective),
            ExactLinearProgramResult::Infeasible
        );
    }

    #[test]
    fn exact_linear_program_pivots_zero_artificial_variable_out_before_phase_two() {
        let matrix = vec![
            vec![rational_one(), rational_zero()],
            vec![rational_zero(), -rational_one()],
        ];
        let rhs = vec![rational_one(), rational_zero()];
        let objective = vec![rational_one(), rational_zero()];

        assert_matches!(
            maximize_nonnegative_equality_linear_program(&matrix, &rhs, &objective),
            ExactLinearProgramResult::Optimal {
                solution,
                objective_value,
            } if solution == vec![rational_one(), rational_zero()]
                && objective_value == rational_one()
        );
    }

    #[test]
    fn exact_linear_program_uses_exact_simplex_when_coefficients_exceed_f64_range() {
        let huge = BigRational::from_integer(BigInt::from(1_u8) << usize::from(u16::MAX));
        let matrix = vec![vec![huge.clone()]];
        let rhs = vec![huge];
        let objective = vec![rational_one()];

        assert_matches!(
            maximize_nonnegative_equality_linear_program(&matrix, &rhs, &objective),
            ExactLinearProgramResult::Optimal {
                solution,
                objective_value,
            } if solution == vec![rational_one()] && objective_value == rational_one()
        );
    }

    /// Checks certified separation and range-sensitive fallback against simplices
    /// in opposite orthants, whose only common point is their shared origin.
    fn assert_shared_vertex_filter_and_fallback<const D: usize>() {
        for (scale, certifies) in [
            (1.0, true),
            (2.0_f64.powi(-40), false),
            (f64::from_bits(1), false),
            (f64::MAX, false),
        ] {
            let (first, second, shared) = boundary_degenerate_pair::<D>(1, scale);
            assert_eq!(
                filtered_single_shared_vertex_confinement(&first, &second, &shared, &[1.0; D]),
                certifies,
                "unexpected certificate for scale {scale}"
            );
            assert_eq!(
                filtered_single_shared_vertex_confinement(&second, &first, &shared, &[-1.0; D]),
                certifies
            );
            assert_eq!(
                optimized_classification(&first, &second),
                IntersectionClassification::Valid
            );
            assert_eq!(
                optimized_classification(&second, &first),
                IntersectionClassification::Valid
            );
        }

        let (first, second, shared) = boundary_degenerate_pair::<D>(1, 1.0);
        let mut tangent_axis = [0.0; D];
        tangent_axis[0] = 1.0;
        // Zero projections and an exact unit margin must remain inconclusive.
        for axis in [
            tangent_axis,
            [1.0 / 1024.0; D],
            [f64::MAX; D],
            [f64::NAN; D],
        ] {
            assert!(!filtered_single_shared_vertex_confinement(
                &first, &second, &shared, &axis
            ));
        }
        assert!(!filtered_single_shared_vertex_confinement(
            &first,
            &second,
            &shared,
            &[1.0; D][..D - 1]
        ));
        // Adjacent representable axes distinguish a strict unit margin from
        // equality on both sides of the common vertex.
        for (component, certifies) in [
            ((1.0_f64 / 1024.0).next_down(), false),
            ((1.0_f64 / 1024.0).next_up(), true),
        ] {
            assert_eq!(
                filtered_single_shared_vertex_confinement(
                    &first,
                    &second,
                    &shared,
                    &[component; D]
                ),
                certifies
            );
            assert_eq!(
                filtered_single_shared_vertex_confinement(
                    &second,
                    &first,
                    &shared,
                    &[-component; D]
                ),
                certifies
            );
        }
        assert_matches!(
            intersection_via_linear_program(&first, &second, &shared, false),
            IntersectionLinearProgramResult::Valid
        );

        // Identical geometry with different non-shared labels overlaps outside
        // the shared vertex; a missing certificate must not turn into success.
        let (first, second, shared) = boundary_degenerate_pair::<D>(1, -1.0);
        assert!(!filtered_single_shared_vertex_confinement(
            &first, &second, &shared, &[1.0; D]
        ));
        assert_matches!(
            validate_simplex_intersection(&first, &second),
            Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace { .. })
        );
    }

    /// Large common translations obscure a small, exactly representable gap in
    /// floating-point projections, but do not change the intersection geometry.
    fn assert_shared_vertex_filter_cancellation<const D: usize>() {
        let (first, second, shared) = boundary_degenerate_pair::<D>(1, 1.0);
        let translation = 2.0_f64.powi(50);
        let translate = |simplex: &LabeledSimplexRealization<usize, D>| {
            LabeledSimplexRealization::try_new(
                simplex.labels().iter().copied(),
                simplex
                    .coordinates()
                    .iter()
                    .map(|point| point.map(|value| value + translation)),
            )
            .unwrap()
        };
        let first = translate(&first);
        let second = translate(&second);
        assert!(!filtered_single_shared_vertex_confinement(
            &first, &second, &shared, &[1.0; D]
        ));
        assert!(exact_shared_face_confinement_certificate(
            &first, &second, &shared, &[1.0; D]
        ));
        assert_eq!(
            optimized_classification(&first, &second),
            IntersectionClassification::Valid
        );
        assert_eq!(
            optimized_classification(&second, &first),
            IntersectionClassification::Valid
        );
    }

    /// Multiple nonzero products need an error allowance even when the exact
    /// signed gap lies just above one; translation also exercises common error.
    fn assert_shared_vertex_filter_rounding_margin<const D: usize>() {
        for translation in [0.0, 0.25] {
            let mut coordinates = standard_simplex_coordinates::<D>();
            coordinates[1][0] = 0.5;
            coordinates[1][1] = 0.5;
            let first = LabeledSimplexRealization::try_new(
                0..=D,
                coordinates
                    .iter()
                    .map(|point| point.map(|value| translation + value)),
            )
            .unwrap();
            let second = LabeledSimplexRealization::try_new(
                core::iter::once(0).chain(D + 1..=2 * D),
                coordinates
                    .iter()
                    .map(|point| point.map(|value| 2.0_f64.mul_add(-value, translation))),
            )
            .unwrap();
            for (left, right) in [(&first, &second), (&second, &first)] {
                assert_eq!(validate_simplex_intersection(left, right), Ok(()));
            }
            for (component, certifies) in [
                ((1.0_f64 / 1024.0).next_down(), false),
                (1.0 / 1024.0, false),
                ((1.0_f64 / 1024.0).next_up(), false),
                (2.0 / 1024.0, true),
            ] {
                // Only the smaller simplex is near the unit margin, so reversing
                // the pair checks each strict comparison independently.
                for (left, right, direction, left_scale, right_scale) in [
                    (&first, &second, 1.0, 1.0, 2.0),
                    (&second, &first, -1.0, 2.0, 1.0),
                ] {
                    let axis = [direction * component; D];
                    let scaled_axis = LaVector::try_new(axis.map(|value| value * 1024.0)).unwrap();
                    let common = scaled_axis
                        .dot_with_errbound(&LaVector::try_new(left.coordinates()[0]).unwrap())
                        .unwrap()
                        .unwrap();
                    if translation != 0.0 {
                        assert!(common.absolute_error_bound() > 0.0);
                    }
                    let mut exact_margin_holds = true;
                    for (simplex, side, scale) in
                        [(left, 1.0, left_scale), (right, -1.0, right_scale)]
                    {
                        let vertex = simplex.coordinates()[1];
                        let projection = scaled_axis
                            .dot_with_errbound(&LaVector::try_new(vertex).unwrap())
                            .unwrap()
                            .unwrap();
                        assert!(projection.absolute_error_bound() > 0.0);
                        // Form exact differences from stored coordinates, without
                        // reusing the filter's separately rounded projections.
                        let gap = vertex.iter().zip(&axis).zip(&left.coordinates()[0]).fold(
                            rational_zero(),
                            |sum, ((&value, &coefficient), &shared)| {
                                sum + rational_from_f64(side * coefficient * 1024.0)
                                    * (rational_from_f64(value) - rational_from_f64(shared))
                            },
                        );
                        assert_eq!(gap, rational_from_f64(component * 1024.0 * scale));
                        exact_margin_holds &= gap >= rational_one();
                    }
                    assert_eq!(
                        filtered_single_shared_vertex_confinement(left, right, &[0], &axis),
                        certifies
                    );
                    assert_eq!(
                        exact_shared_face_confinement_certificate(left, right, &[0], &axis),
                        exact_margin_holds
                    );
                }
            }
        }
    }

    /// A nonzero subnormal common projection has no floating certificate, but
    /// exact arithmetic still proves intersection only at the shared vertex.
    fn assert_shared_vertex_filter_subnormal_common<const D: usize>() {
        let (first, second, shared) = boundary_degenerate_pair::<D>(1, 1.0);
        let translate = |simplex: &LabeledSimplexRealization<usize, D>| {
            LabeledSimplexRealization::try_new(
                simplex.labels().iter().copied(),
                simplex
                    .coordinates()
                    .iter()
                    .map(|point| point.map(|value| value + f64::from_bits(1))),
            )
            .unwrap()
        };
        let first = translate(&first);
        let second = translate(&second);
        for (left, right, direction) in [(&first, &second, 1.0), (&second, &first, -1.0)] {
            let axis = [direction; D];
            let scaled_axis = LaVector::try_new([direction * 1024.0; D]).unwrap();
            let common = LaVector::try_new(left.coordinates()[0]).unwrap();
            assert!(scaled_axis.dot_with_errbound(&common).unwrap().is_none());
            assert!(!filtered_single_shared_vertex_confinement(
                left, right, &shared, &axis
            ));
            assert!(exact_shared_face_confinement_certificate(
                left, right, &shared, &axis
            ));
            assert_eq!(validate_simplex_intersection(left, right), Ok(()));
        }
    }

    /// A finite projection at either range endpoint cannot enclose a unit
    /// margin. The opposite orthants still meet only at their shared vertex.
    fn assert_shared_vertex_filter_margin_overflow<const D: usize>() {
        for (extreme, inward) in [
            (f64::MAX, f64::MAX.next_down()),
            (-f64::MAX, -f64::MAX.next_down()),
        ] {
            let mut coordinates = standard_simplex_coordinates::<D>();
            for point in &mut coordinates {
                point[0] = extreme;
            }
            coordinates[1][0] = inward;
            coordinates[1][1] = 1.0;
            let first =
                LabeledSimplexRealization::try_new(0..=D, coordinates.iter().copied()).unwrap();
            let second = LabeledSimplexRealization::try_new(
                core::iter::once(0).chain(D + 1..=2 * D),
                coordinates.iter().map(|point| {
                    from_fn(|axis| if axis == 0 { point[axis] } else { -point[axis] })
                }),
            )
            .unwrap();
            let mut axis = [0.0; D];
            axis[0] = 1.0 / 1024.0;
            let common = LaVector::try_new(axis.map(|value| value * 1024.0))
                .unwrap()
                .dot_with_errbound(&LaVector::try_new(first.coordinates()[0]).unwrap())
                .unwrap()
                .unwrap();
            // The projection itself is exact and finite; only its outward unit
            // margin fails. This distinguishes threshold overflow from dot overflow.
            assert_eq!(common.lower_bound().to_bits(), extreme.to_bits());
            assert_eq!(common.upper_bound().to_bits(), extreme.to_bits());
            for (left, right, direction) in [(&first, &second, 1.0), (&second, &first, -1.0)] {
                assert!(!filtered_single_shared_vertex_confinement(
                    left,
                    right,
                    &[0],
                    &axis
                ));
                // All non-shared vertices have a strictly positive/negative sum
                // in the remaining coordinates, independently of the huge offset.
                let separating_axis: [f64; D] =
                    from_fn(|axis| if axis == 0 { 0.0 } else { direction });
                assert!(exact_shared_face_confinement_certificate(
                    left,
                    right,
                    &[0],
                    &separating_axis
                ));
                assert_eq!(validate_simplex_intersection(left, right), Ok(()));
            }
        }
    }

    macro_rules! shared_vertex_bound_tests {
        ($($dim:literal),+ $(,)?) => {
            pastey::paste! {
                $(
                    #[test]
                    fn [<shared_vertex_filter_and_fallback_ $dim d>]() {
                        assert_shared_vertex_filter_and_fallback::<$dim>();
                    }

                    #[test]
                    fn [<shared_vertex_filter_cancellation_ $dim d>]() {
                        assert_shared_vertex_filter_cancellation::<$dim>();
                    }

                    #[test]
                    fn [<shared_vertex_filter_rounding_margin_ $dim d>]() {
                        assert_shared_vertex_filter_rounding_margin::<$dim>();
                    }

                    #[test]
                    fn [<shared_vertex_filter_subnormal_common_ $dim d>]() {
                        assert_shared_vertex_filter_subnormal_common::<$dim>();
                    }

                    #[test]
                    fn [<shared_vertex_filter_margin_overflow_ $dim d>]() {
                        assert_shared_vertex_filter_margin_overflow::<$dim>();
                    }

                    proptest! {
                        #![proptest_config(ProptestConfig {
                            cases: 32,
                            ..ProptestConfig::default()
                        })]

                        #[test]
                        fn [<shared_vertex_filter_certificate_is_exact_ $dim d>](
                            raw_coordinates in proptest::collection::vec(-8_i16..=4, ($dim + 1) * $dim),
                            raw_axis in proptest::collection::vec(1_i16..=8, $dim),
                        ) {
                            let (first, second, shared) = random_grid_pair::<$dim>(&raw_coordinates, 1);
                            let axis: Vec<_> = raw_axis.into_iter().map(f64::from).collect();
                            if filtered_single_shared_vertex_confinement(&first, &second, &shared, &axis) {
                                prop_assert!(exact_shared_face_confinement_certificate(&first, &second, &shared, &axis));
                                prop_assert_eq!(optimized_classification(&first, &second), IntersectionClassification::Valid);
                            }
                        }
                    }
                )+
            }
        };
    }

    shared_vertex_bound_tests!(2, 3, 4, 5, 6);

    #[test]
    fn exact_axis_rejects_post_scaling_overflow() {
        let simplex = LabeledSimplexRealization::try_new([0, 1], [[0.0], [1.0]]).unwrap();

        assert_eq!(
            exact_axis_through_shared_face(&simplex, &[], &[f64::MAX]),
            None
        );
    }

    #[test]
    fn exact_axis_rejects_non_finite_provisional_affine_offset() {
        let simplex = LabeledSimplexRealization::try_new(
            [0, 1, 2],
            [[f64::MAX, 0.0], [0.0, 1.0], [0.0, 0.0]],
        )
        .unwrap();

        assert_eq!(
            exact_axis_through_shared_face(&simplex, &[0], &[2.0, 0.0]),
            None
        );
    }

    #[test]
    fn phase_two_dual_certificate_proves_zero_objective_bound() {
        let matrix = vec![vec![rational_one(), rational_one()]];
        let rhs = vec![rational_one()];
        let objective = vec![rational_zero(), rational_zero()];

        assert!(phase_two_dual_proves_shared_face_confinement(
            &matrix,
            &rhs,
            &objective,
            &[0],
        ));
    }
}
