//! Exact revised-simplex support for realized-simplex intersection.
//!
//! This private geometry module owns the provisional floating-point basis search,
//! exact Phase I/II certification, artificial-variable handling, dual certificates,
//! and matrix helpers used by Level 4 realization validation.

#![forbid(unsafe_code)]

use crate::geometry::matrix::{StackMatrixDispatchError, solve_exact_runtime_system};
use crate::geometry::realization::{
    LabeledSimplexRealization, SimplexIntersectionWitness, SimplexRealizationBuffer,
};
use la_stack::{BigInt, BigRational, FromPrimitive, Signed, ToPrimitive};

/// Tries cheap candidate separators in the normal space of one shared face.
#[expect(
    clippy::too_many_lines,
    reason = "the separator search is one bounded projection-and-certification workflow"
)]
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
    let shared_deltas: Vec<Vec<f64>> = shared_coordinates[1..]
        .iter()
        .map(|coordinates| {
            coordinates
                .iter()
                .zip(base)
                .map(|(coordinate, base_coordinate)| coordinate - base_coordinate)
                .collect()
        })
        .collect();
    let gram: Vec<Vec<f64>> = shared_deltas
        .iter()
        .map(|left| {
            shared_deltas
                .iter()
                .map(|right| dot_product_f64(left, right))
                .collect()
        })
        .collect();

    let mut axis = vec![0.0; D];
    let mut normalized_rays = Vec::with_capacity(2 * D);
    for (label, coordinates) in first.labels().iter().zip(first.coordinates()) {
        if shared_labels.contains(label) {
            continue;
        }
        let raw_ray: Vec<_> = coordinates
            .iter()
            .zip(base)
            .map(|(coordinate, shared)| coordinate - shared)
            .collect();
        let Some(ray) = project_ray_orthogonal_to_shared_face(raw_ray, &shared_deltas, &gram)
        else {
            return false;
        };
        for coordinate in 0..D {
            axis[coordinate] += ray[coordinate];
        }
        push_normalized_ray(&mut normalized_rays, ray);
    }
    for (label, coordinates) in second.labels().iter().zip(second.coordinates()) {
        if shared_labels.contains(label) {
            continue;
        }
        let raw_ray: Vec<_> = base
            .iter()
            .zip(coordinates)
            .map(|(shared, coordinate)| shared - coordinate)
            .collect();
        let Some(ray) = project_ray_orthogonal_to_shared_face(raw_ray, &shared_deltas, &gram)
        else {
            return false;
        };
        for coordinate in 0..D {
            axis[coordinate] += ray[coordinate];
        }
        push_normalized_ray(&mut normalized_rays, ray);
    }
    if axis.iter().all(|value| value.is_finite())
        && provisional_axis_certifies_confinement(first, second, shared_labels, &axis)
    {
        return true;
    }

    let mut refined_axis = vec![0.0; D];
    for ray in &normalized_rays {
        for (value, component) in refined_axis.iter_mut().zip(ray) {
            *value += component;
        }
    }
    for _ in 0..16 {
        let mut adjusted = false;
        for ray in &normalized_rays {
            let projection = dot_product_f64(&refined_axis, ray);
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

/// Removes provisional components tangent to the shared affine face.
fn project_ray_orthogonal_to_shared_face(
    mut ray: Vec<f64>,
    shared_deltas: &[Vec<f64>],
    gram: &[Vec<f64>],
) -> Option<Vec<f64>> {
    if shared_deltas.is_empty() {
        return Some(ray);
    }
    let residuals: Vec<_> = shared_deltas
        .iter()
        .map(|delta| dot_product_f64(delta, &ray))
        .collect();
    let correction = solve_f64_system(gram.to_vec(), residuals)?;
    for (coordinate, value) in ray.iter_mut().enumerate() {
        for (coefficient, delta) in correction.iter().zip(shared_deltas) {
            *value = (-coefficient).mul_add(delta[coordinate], *value);
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
fn push_normalized_ray(rays: &mut Vec<Vec<f64>>, ray: Vec<f64>) {
    let norm = ray
        .iter()
        .fold(0.0_f64, |value, component| value.hypot(*component));
    if norm.is_finite() && norm > 0.0 {
        rays.push(ray.into_iter().map(|component| component / norm).collect());
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
        let Some(coordinates) = barycentric_coordinates_of_vertices(second, first) else {
            return IntersectionLinearProgramResult::SingularBarycentricBasis;
        };
        Some(coordinates)
    } else {
        None
    };
    let Some(program) = provisional_intersection_program(first, second, shared_labels) else {
        return intersection_via_active_sets(
            first,
            second,
            shared_labels,
            precomputed_barycentric.as_deref(),
        );
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
                    precomputed_barycentric.as_deref(),
                );
            }
            IntersectionLinearProgramResult::Invalid(SimplexIntersectionWitness {
                shared: shared_labels.iter().cloned().collect(),
                first_only_witness,
                second_only_witness,
            })
        }
        ExactLinearProgramResult::Failed => intersection_via_active_sets(
            first,
            second,
            shared_labels,
            precomputed_barycentric.as_deref(),
        ),
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
        let Some(coordinates) = barycentric_coordinates_of_vertices(second, first) else {
            return IntersectionLinearProgramResult::SingularBarycentricBasis;
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

    for beta in intersection_polytope_vertices(second_vertices_in_first) {
        let alpha = alpha_from_beta(&beta, second_vertices_in_first);
        let first_only_witness = positive_nonshared_labels(&alpha, first.labels(), shared_labels);
        let second_only_witness = positive_nonshared_labels(&beta, second.labels(), shared_labels);

        if !first_only_witness.is_empty() || !second_only_witness.is_empty() {
            return IntersectionLinearProgramResult::Invalid(SimplexIntersectionWitness {
                shared: shared_labels.iter().cloned().collect(),
                first_only_witness,
                second_only_witness,
            });
        }
    }

    IntersectionLinearProgramResult::Valid
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
) -> Option<Vec<Vec<BigRational>>> {
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
) -> Option<Vec<BigRational>> {
    if D == 0 {
        return Some(vec![rational_one()]);
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

    let lambdas = solve_rational_system(matrix, rhs)?;
    let lambda_sum = lambdas
        .iter()
        .fold(rational_zero(), |sum, value| sum + value.clone());
    let mut barycentric = Vec::with_capacity(D + 1);
    barycentric.push(rational_one() - lambda_sum);
    barycentric.extend(lambdas);
    Some(barycentric)
}

/// Enumerates candidate vertices of the intersection polytope in second-simplex weights.
fn intersection_polytope_vertices(
    second_vertices_in_first: &[Vec<BigRational>],
) -> Vec<Vec<BigRational>> {
    let variable_count = second_vertices_in_first.len();
    let active_count = variable_count.saturating_sub(1);
    let constraint_count = variable_count * 2;
    let mut active_set = Vec::with_capacity(active_count);
    let mut vertices = Vec::new();

    enumerate_active_sets(
        constraint_count,
        active_count,
        0,
        &mut active_set,
        &mut |active_constraints| {
            if let Some(beta) =
                intersection_vertex_for_active_set(second_vertices_in_first, active_constraints)
                && beta_is_feasible(&beta, second_vertices_in_first)
            {
                vertices.push(beta);
            }
        },
    );

    vertices
}

/// Recursively enumerates active constraint sets for the simplex-intersection LP.
fn enumerate_active_sets<F>(
    constraint_count: usize,
    active_count: usize,
    start: usize,
    active_set: &mut Vec<usize>,
    on_active_set: &mut F,
) where
    F: FnMut(&[usize]),
{
    if active_set.len() == active_count {
        on_active_set(active_set);
        return;
    }

    let remaining = active_count - active_set.len();
    let last_start = constraint_count.saturating_sub(remaining);
    for constraint in start..=last_start {
        active_set.push(constraint);
        enumerate_active_sets(
            constraint_count,
            active_count,
            constraint + 1,
            active_set,
            on_active_set,
        );
        active_set.pop();
    }
}

/// Solves one active-constraint system and returns the candidate beta weights.
fn intersection_vertex_for_active_set(
    second_vertices_in_first: &[Vec<BigRational>],
    active_constraints: &[usize],
) -> Option<Vec<BigRational>> {
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

    solve_rational_system(matrix, rhs)
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
/// Phase I introduces one artificial variable per equality. Bland's pivot
/// ordering makes degenerate exact-arithmetic bases deterministic and avoids
/// cycling. Artificial variables are pivoted out at zero before Phase II.
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
        let Some(basic_solution) =
            solve_rational_system(current_basis_matrix.clone(), rhs.to_vec())
        else {
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
            let Some(direction) = solve_rational_system(
                current_basis_matrix.clone(),
                matrix_column(matrix, candidate),
            ) else {
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
        let Some(basic_solution) =
            solve_rational_system(current_basis_matrix.clone(), rhs.to_vec())
        else {
            return RevisedSimplexResult::Failed;
        };
        if basic_solution.iter().any(Signed::is_negative) {
            return RevisedSimplexResult::Failed;
        }

        let basic_objective: Vec<_> = basis
            .iter()
            .map(|&index| objective[index].clone())
            .collect();
        let Some(dual_solution) = solve_rational_system(
            transpose_square_matrix(&current_basis_matrix),
            basic_objective,
        ) else {
            return RevisedSimplexResult::Failed;
        };

        let entering = (0..entering_variable_count)
            .filter(|candidate| !basis.contains(candidate))
            .find(|&candidate| {
                let reduced_cost = objective[candidate].clone()
                    - dot_product(&dual_solution, &matrix_column(matrix, candidate));
                reduced_cost.is_positive()
            });
        let Some(entering) = entering else {
            let mut solution = vec![rational_zero(); objective.len()];
            for (&variable, value) in basis.iter().zip(&basic_solution) {
                solution[variable] = value.clone();
            }
            return RevisedSimplexResult::Optimal {
                objective_value: dot_product(objective, &solution),
                solution,
            };
        };

        let Some(direction) =
            solve_rational_system(current_basis_matrix, matrix_column(matrix, entering))
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
    let float_matrix = rational_matrix_to_f64(matrix)?;
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
        let mut basic_solution = solve_f64_system(current_basis_matrix.clone(), rhs.to_vec())?;
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
            transpose_square_matrix_f64(&current_basis_matrix),
            basic_objective,
        )?;

        let entering = (0..entering_variable_count)
            .filter(|candidate| !basis.contains(candidate))
            .find(|&candidate| {
                let column = matrix_column_f64(matrix, candidate);
                let dual_contribution = dot_product_f64(&dual_solution, &column);
                let scale = 1.0 + objective[candidate].abs() + dual_contribution.abs();
                objective[candidate] - dual_contribution > REDUCED_COST_TOLERANCE * scale
            });
        let Some(entering) = entering else {
            return Some(dual_solution);
        };

        let direction =
            solve_f64_system(current_basis_matrix, matrix_column_f64(matrix, entering))?;
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
        let Some(basic_solution) = solve_f64_system(current_basis_matrix.clone(), rhs.to_vec())
        else {
            return false;
        };
        if basic_solution[basis_position].abs() > ZERO_TOLERANCE {
            return false;
        }

        let replacement = (0..original_variable_count)
            .filter(|candidate| !basis.contains(candidate))
            .find(|&candidate| {
                solve_f64_system(
                    current_basis_matrix.clone(),
                    matrix_column_f64(matrix, candidate),
                )
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

    let scaled_axis: Vec<_> = axis.iter().map(|value| *value * 1024.0).collect();
    if scaled_axis.len() != D || scaled_axis.iter().any(|value| !value.is_finite()) {
        return false;
    }
    first
        .labels()
        .iter()
        .zip(first.coordinates())
        .filter(|(label, _)| *label != shared_label)
        .all(|(_, coordinates)| {
            certified_dot_difference_lower_bound(&scaled_axis, coordinates, first_shared)
                .is_some_and(|lower_bound| lower_bound > 1.0)
        })
        && second
            .labels()
            .iter()
            .zip(second.coordinates())
            .filter(|(label, _)| *label != shared_label)
            .all(|(_, coordinates)| {
                certified_dot_difference_lower_bound(&scaled_axis, second_shared, coordinates)
                    .is_some_and(|lower_bound| lower_bound > 1.0)
            })
}

/// Returns a conservative lower bound for `axis · (left - right)`.
fn certified_dot_difference_lower_bound<const D: usize>(
    axis: &[f64],
    left: &[f64; D],
    right: &[f64; D],
) -> Option<f64> {
    let mut estimate = 0.0;
    let mut magnitude = 0.0;
    for ((coefficient, left_value), right_value) in axis.iter().zip(left).zip(right) {
        let left_product = coefficient * left_value;
        let right_product = coefficient * right_value;
        if !left_product.is_finite()
            || !right_product.is_finite()
            || product_underflowed(*coefficient, *left_value, left_product)
            || product_underflowed(*coefficient, *right_value, right_product)
        {
            return None;
        }
        estimate = coefficient.mul_add(*left_value, estimate);
        estimate = (-coefficient).mul_add(*right_value, estimate);
        magnitude += left_product.abs() + right_product.abs();
        if !estimate.is_finite() || !magnitude.is_finite() {
            return None;
        }
    }

    let operation_count = f64::from(u32::try_from(2 * D).ok()?);
    let denominator = 1.0 - operation_count * f64::EPSILON;
    if denominator <= 0.0 {
        return None;
    }
    let gamma = operation_count * f64::EPSILON / denominator;
    let magnitude_upper_bound = magnitude / denominator.powi(2);
    let error_bound = 4.0 * gamma * magnitude_upper_bound;
    error_bound.is_finite().then_some(estimate - error_bound)
}

/// Detects a product whose exact non-zero value rounded into the subnormal range.
fn product_underflowed(left: f64, right: f64, product: f64) -> bool {
    left != 0.0 && right != 0.0 && (product == 0.0 || product.abs() < f64::MIN_POSITIVE)
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
    let pivot_columns = independent_columns_f64(&constraint_matrix)?;
    let free_columns: Vec<_> = (0..=D)
        .filter(|column| !pivot_columns.contains(column))
        .collect();
    let pivot_matrix: Vec<Vec<f64>> = constraint_matrix
        .iter()
        .map(|row| pivot_columns.iter().map(|&column| row[column]).collect())
        .collect();

    let mut provisional_affine = axis.to_vec();
    provisional_affine.push(-dot_product_f64(axis, shared_coordinates[0]));
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

/// Selects a full-rank set of matrix columns with provisional elimination.
fn independent_columns_f64(matrix: &[Vec<f64>]) -> Option<Vec<usize>> {
    const PIVOT_TOLERANCE: f64 = 1.0e-12;

    let row_count = matrix.len();
    let column_count = matrix.first()?.len();
    if row_count == 0 || matrix.iter().any(|row| row.len() != column_count) {
        return None;
    }
    let mut work = matrix.to_vec();
    let scale = work
        .iter()
        .flatten()
        .fold(0.0_f64, |maximum, value| maximum.max(value.abs()));
    let mut pivot_columns = Vec::with_capacity(row_count);
    for column in 0..column_count {
        let pivot_position = (pivot_columns.len()..row_count).max_by(|&left, &right| {
            work[left][column]
                .abs()
                .total_cmp(&work[right][column].abs())
        })?;
        if work[pivot_position][column].abs() <= PIVOT_TOLERANCE * scale.max(1.0) {
            continue;
        }
        let pivot_row = pivot_columns.len();
        work.swap(pivot_row, pivot_position);
        let pivot = work[pivot_row][column];
        let pivot_values = work[pivot_row].clone();
        for row_values in work.iter_mut().skip(pivot_row + 1) {
            let factor = row_values[column] / pivot;
            for (value, pivot_value) in row_values[column..].iter_mut().zip(&pivot_values[column..])
            {
                *value -= factor * pivot_value;
            }
        }
        pivot_columns.push(column);
        if pivot_columns.len() == row_count {
            return Some(pivot_columns);
        }
    }
    None
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
    let Some(dual_solution) = solve_rational_system(
        transpose_square_matrix(&current_basis_matrix),
        basic_objective,
    ) else {
        return false;
    };
    let dual_is_feasible = (0..variable_count).all(|candidate| {
        objective[candidate].clone()
            - dot_product(&dual_solution, &matrix_column(matrix, candidate))
            <= rational_zero()
    });
    dual_is_feasible && dot_product(rhs, &dual_solution) <= rational_zero()
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
    let basic_solution = solve_rational_system(current_basis_matrix.clone(), rhs.to_vec())?;
    if basic_solution.iter().any(Signed::is_negative) {
        return None;
    }

    let basic_objective: Vec<_> = basis
        .iter()
        .map(|&index| objective[index].clone())
        .collect();
    let dual_solution = solve_rational_system(
        transpose_square_matrix(&current_basis_matrix),
        basic_objective,
    )?;
    let dual_is_feasible = (0..entering_variable_count)
        .filter(|candidate| !basis.contains(candidate))
        .all(|candidate| {
            let reduced_cost = objective[candidate].clone()
                - dot_product(&dual_solution, &matrix_column(matrix, candidate));
            !reduced_cost.is_positive()
        });
    if !dual_is_feasible {
        return None;
    }

    let mut solution = vec![rational_zero(); objective.len()];
    for (&variable, value) in basis.iter().zip(&basic_solution) {
        solution[variable] = value.clone();
    }
    Some(RevisedSimplexResult::Optimal {
        objective_value: dot_product(objective, &solution),
        solution,
    })
}

/// Converts an exact rational matrix to finite f64 values for basis navigation.
fn rational_matrix_to_f64(matrix: &[Vec<BigRational>]) -> Option<Vec<Vec<f64>>> {
    matrix.iter().map(|row| rationals_to_f64(row)).collect()
}

/// Converts exact rationals to finite f64 values for provisional calculations.
fn rationals_to_f64(values: &[BigRational]) -> Option<Vec<f64>> {
    values
        .iter()
        .map(|value| value.to_f64().filter(|converted| converted.is_finite()))
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

/// Computes a floating-point dot product used only for provisional pivot selection.
fn dot_product_f64(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

/// Solves a small floating-point system with scaled partial pivoting.
#[expect(
    clippy::needless_range_loop,
    reason = "index-based elimination keeps pivot row/column operations explicit"
)]
fn solve_f64_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    const PIVOT_TOLERANCE: f64 = 1.0e-14;

    let dimension = rhs.len();
    if matrix.len() != dimension || matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }

    for pivot_column in 0..dimension {
        let pivot_row = (pivot_column..dimension).max_by(|&left, &right| {
            matrix[left][pivot_column]
                .abs()
                .total_cmp(&matrix[right][pivot_column].abs())
        })?;
        let row_scale = matrix[pivot_row]
            .iter()
            .map(|&value| value.abs())
            .fold(0.0_f64, f64::max);
        if !row_scale.is_finite()
            || matrix[pivot_row][pivot_column].abs() <= PIVOT_TOLERANCE * row_scale.max(1.0)
        {
            return None;
        }
        if pivot_row != pivot_column {
            matrix.swap(pivot_column, pivot_row);
            rhs.swap(pivot_column, pivot_row);
        }

        for row in pivot_column + 1..dimension {
            let factor = matrix[row][pivot_column] / matrix[pivot_column][pivot_column];
            matrix[row][pivot_column] = 0.0;
            for column in pivot_column + 1..dimension {
                matrix[row][column] =
                    factor.mul_add(-matrix[pivot_column][column], matrix[row][column]);
            }
            rhs[row] = factor.mul_add(-rhs[pivot_column], rhs[row]);
        }
    }

    let mut solution = vec![0.0; dimension];
    for row in (0..dimension).rev() {
        let mut sum = rhs[row];
        for column in row + 1..dimension {
            sum = matrix[row][column].mul_add(-solution[column], sum);
        }
        solution[row] = sum / matrix[row][row];
        if !solution[row].is_finite() {
            return None;
        }
    }
    Some(solution)
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

/// Computes an exact dot product without introducing floating-point decisions.
fn dot_product(left: &[BigRational], right: &[BigRational]) -> BigRational {
    left.iter()
        .zip(right)
        .fold(rational_zero(), |sum, (left, right)| {
            sum + left.clone() * right.clone()
        })
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

#[expect(
    clippy::needless_range_loop,
    reason = "index-based elimination keeps pivot row/column operations explicit"
)]
/// Solves a square rational linear system by Gaussian elimination.
fn solve_rational_system(
    mut matrix: Vec<Vec<BigRational>>,
    mut rhs: Vec<BigRational>,
) -> Option<Vec<BigRational>> {
    let dimension = rhs.len();
    if matrix.len() != dimension || matrix.iter().any(|row| row.len() != dimension) {
        return None;
    }
    match try_solve_exact_f64_system(&matrix, &rhs) {
        ExactF64Solve::Solution(solution) => return Some(solution),
        ExactF64Solve::Singular => return None,
        ExactF64Solve::Unsupported => {}
    }

    for pivot_col in 0..dimension {
        let pivot_row =
            (pivot_col..dimension).find(|&row| matrix[row][pivot_col] != rational_zero())?;
        if pivot_row != pivot_col {
            matrix.swap(pivot_col, pivot_row);
            rhs.swap(pivot_col, pivot_row);
        }

        let pivot_value = matrix[pivot_col][pivot_col].clone();
        for row in pivot_col + 1..dimension {
            if matrix[row][pivot_col] == rational_zero() {
                continue;
            }
            let factor = matrix[row][pivot_col].clone() / pivot_value.clone();
            matrix[row][pivot_col] = rational_zero();
            for col in pivot_col + 1..dimension {
                matrix[row][col] =
                    matrix[row][col].clone() - factor.clone() * matrix[pivot_col][col].clone();
            }
            rhs[row] = rhs[row].clone() - factor * rhs[pivot_col].clone();
        }
    }

    let mut solution = vec![rational_zero(); dimension];
    for row in (0..dimension).rev() {
        let mut sum = rhs[row].clone();
        for col in row + 1..dimension {
            sum -= matrix[row][col].clone() * solution[col].clone();
        }
        solution[row] = sum / matrix[row][row].clone();
    }

    Some(solution)
}

/// Outcome of attempting the fraction-free exact `f64` solver.
enum ExactF64Solve {
    /// The runtime stack dispatcher cannot represent this rational system.
    Unsupported,
    /// The system is exactly singular.
    Singular,
    /// The exact system solution.
    Solution(Vec<BigRational>),
}

/// Uses fraction-free Bareiss elimination when every coefficient came directly
/// from a finite `f64`; otherwise leaves the rational solver as the fallback.
fn try_solve_exact_f64_system(matrix: &[Vec<BigRational>], rhs: &[BigRational]) -> ExactF64Solve {
    let Some(float_matrix) = exact_rational_matrix_as_f64(matrix) else {
        return ExactF64Solve::Unsupported;
    };
    let Some(float_rhs) = exact_rationals_as_f64(rhs) else {
        return ExactF64Solve::Unsupported;
    };
    let Some(result) = solve_exact_runtime_system(&float_matrix, &float_rhs) else {
        return ExactF64Solve::Unsupported;
    };
    match result {
        Ok(solution) => ExactF64Solve::Solution(solution),
        Err(StackMatrixDispatchError::La {
            source: la_stack::LaError::Singular { .. },
        }) => ExactF64Solve::Singular,
        Err(_) => ExactF64Solve::Unsupported,
    }
}

/// Converts rationals to `f64` only when doing so preserves each value exactly.
fn exact_rational_matrix_as_f64(matrix: &[Vec<BigRational>]) -> Option<Vec<Vec<f64>>> {
    matrix
        .iter()
        .map(|row| exact_rationals_as_f64(row))
        .collect()
}

/// Converts a finite floating-point matrix to its exact rational representation.
fn exact_matrix_from_f64(matrix: &[Vec<f64>]) -> Vec<Vec<BigRational>> {
    matrix
        .iter()
        .map(|row| row.iter().copied().map(rational_from_f64).collect())
        .collect()
}

/// Converts rationals to finite `f64` values and verifies an exact round trip.
fn exact_rationals_as_f64(values: &[BigRational]) -> Option<Vec<f64>> {
    values
        .iter()
        .map(|value| {
            let converted = value.to_f64().filter(|candidate| candidate.is_finite())?;
            (rational_from_f64(converted) == *value).then_some(converted)
        })
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
    use proptest::prelude::*;
    use std::assert_matches;

    /// Classification shared by the optimized predicate and legacy exact oracle.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum IntersectionClassification {
        Valid,
        Invalid,
        SingularBarycentricBasis,
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

    /// Reconstructs the pre-optimization exact active-set decision for agreement tests.
    fn legacy_active_set_classification<L, const D: usize>(
        first: &LabeledSimplexRealization<L, D>,
        second: &LabeledSimplexRealization<L, D>,
        shared_labels: &[L],
    ) -> IntersectionClassification
    where
        L: Eq,
    {
        let Some(second_vertices_in_first) = barycentric_coordinates_of_vertices(second, first)
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
            Err(SimplexIntersectionFailure::IntersectionOutsideSharedFace { .. }) => {
                IntersectionClassification::Invalid
            }
            Err(SimplexIntersectionFailure::SingularBarycentricBasis) => {
                IntersectionClassification::SingularBarycentricBasis
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

    macro_rules! generate_intersection_agreement_tests {
        ($dim:literal, $random_test:ident, $boundary_test:ident) => {
            proptest! {
                #![proptest_config(ProptestConfig {
                    cases: 16,
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

    #[test]
    fn filtered_dot_difference_rejects_overflow_and_underflow() {
        assert_eq!(
            certified_dot_difference_lower_bound(&[f64::MAX], &[2.0], &[0.0]),
            None
        );
        assert_eq!(
            certified_dot_difference_lower_bound(&[f64::MIN_POSITIVE], &[0.5], &[0.0],),
            None
        );
    }

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
