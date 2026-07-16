//! Fail-closed recognition and factorization of a narrow batch-filter shape.
//!
//! The first version intentionally accepts only filters equivalent to
//! `must: [FieldCondition(key, Integer Value | Integer Any)]`. Any residual
//! condition makes the plan unsupported, so callers can safely fall back to
//! the existing per-query filter path.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::{Display, Formatter};

use crate::types::{
    AnyVariants, Condition, Filter, IntPayloadType, Match, MatchAny, MatchValue, PayloadKeyType,
    ValueVariants,
};

/// A dynamically sized query bit mask.
///
/// Unlike a single `u64`, this remains correct for batches larger than 64
/// queries. The explicit query indices stored on each atom are convenient for
/// sparse iteration, while this mask is convenient for set operations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueryMask {
    query_count: usize,
    words: Vec<u64>,
}

impl QueryMask {
    fn from_query_indices(query_count: usize, query_indices: &[usize]) -> Self {
        let mut words = vec![0; query_count.div_ceil(u64::BITS as usize)];
        for &query_index in query_indices {
            debug_assert!(query_index < query_count);
            words[query_index / u64::BITS as usize] |= 1_u64 << (query_index % u64::BITS as usize);
        }
        Self { query_count, words }
    }

    pub fn query_count(&self) -> usize {
        self.query_count
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn contains(&self, query_index: usize) -> bool {
        if query_index >= self.query_count {
            return false;
        }
        self.words[query_index / u64::BITS as usize] & (1_u64 << (query_index % u64::BITS as usize))
            != 0
    }

    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }
}

/// One unique integer atom and all queries which reference it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorizedAtom {
    value: IntPayloadType,
    query_indices: Vec<usize>,
    query_mask: QueryMask,
}

impl FactorizedAtom {
    pub fn value(&self) -> IntPayloadType {
        self.value
    }

    pub fn query_indices(&self) -> &[usize] {
        &self.query_indices
    }

    pub fn query_mask(&self) -> &QueryMask {
        &self.query_mask
    }

    /// Number of distinct queries which reference this atom.
    pub fn reference_count(&self) -> usize {
        self.query_indices.len()
    }
}

/// A batch filter represented as unique atoms and their query memberships.
///
/// Atom IDs are stable within a plan and correspond to positions in
/// [`Self::atoms`]. Atoms are sorted by integer value for deterministic plans.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FactorizedFilterPlan {
    key: PayloadKeyType,
    query_count: usize,
    atoms: Vec<FactorizedAtom>,
    query_atoms: Vec<Vec<usize>>,
}

impl FactorizedFilterPlan {
    /// Build a factorized plan, rejecting any shape which the implementation
    /// cannot prove equivalent to a union of integer equality atoms.
    pub fn try_from_filters(filters: &[Filter]) -> Result<Self, FactorizedFilterPlanError> {
        Self::try_from_filter_iter(filters)
    }

    /// Reference-based variant used by the query executor to avoid cloning
    /// wire filters before recognizing the supported shape.
    pub fn try_from_filter_refs(filters: &[&Filter]) -> Result<Self, FactorizedFilterPlanError> {
        Self::try_from_filter_iter(filters.iter().copied())
    }

    fn try_from_filter_iter<'a>(
        filters: impl IntoIterator<Item = &'a Filter>,
    ) -> Result<Self, FactorizedFilterPlanError> {
        let filters = filters.into_iter().collect::<Vec<_>>();
        if filters.is_empty() {
            return Err(FactorizedFilterPlanError::EmptyBatch);
        }

        let mut key: Option<PayloadKeyType> = None;
        let mut atom_to_queries = BTreeMap::<IntPayloadType, Vec<usize>>::new();
        let mut query_atom_values = Vec::with_capacity(filters.len());

        for (query_index, &filter) in filters.iter().enumerate() {
            let (query_key, mut values) = extract_integer_atoms(filter)
                .ok_or(FactorizedFilterPlanError::UnsupportedFilterShape { query_index })?;

            if let Some(expected) = &key {
                if expected != query_key {
                    return Err(FactorizedFilterPlanError::DifferentKey {
                        query_index,
                        expected: expected.clone(),
                        actual: query_key.clone(),
                    });
                }
            } else {
                key = Some(query_key.clone());
            }

            // `AnyVariants::Integers` is currently an IndexSet, but keep an
            // explicit order-preserving deduplication so reference counts stay
            // query-based if the wire representation changes later. Preserving
            // atom order is also required to match stock tie behavior.
            let mut seen_values = BTreeSet::new();
            values.retain(|value| seen_values.insert(*value));
            for &value in &values {
                atom_to_queries.entry(value).or_default().push(query_index);
            }
            query_atom_values.push(values);
        }

        // A shared execution order must preserve every query's original atom
        // order. Build a precedence DAG from adjacent atoms in each query and
        // use a deterministic topological order. Conflicting orders such as
        // [A, B] and [B, A] cannot be executed once globally while preserving
        // stock tie semantics, so fail closed and use the stock path.
        let mut successors = atom_to_queries
            .keys()
            .copied()
            .map(|value| (value, BTreeSet::new()))
            .collect::<BTreeMap<_, _>>();
        let mut in_degree = atom_to_queries
            .keys()
            .copied()
            .map(|value| (value, 0_usize))
            .collect::<BTreeMap<_, _>>();
        for values in &query_atom_values {
            for pair in values.windows(2) {
                let [from, to] = pair else {
                    unreachable!("windows of two always contain two values")
                };
                if from != to && successors.get_mut(from).unwrap().insert(*to) {
                    *in_degree.get_mut(to).unwrap() += 1;
                }
            }
        }

        let mut ready = in_degree
            .iter()
            .filter_map(|(&value, &degree)| (degree == 0).then_some(value))
            .collect::<BTreeSet<_>>();
        let mut unique_atom_values = Vec::with_capacity(atom_to_queries.len());
        while let Some(value) = ready.pop_first() {
            unique_atom_values.push(value);
            for &successor in &successors[&value] {
                let degree = in_degree.get_mut(&successor).unwrap();
                *degree -= 1;
                if *degree == 0 {
                    ready.insert(successor);
                }
            }
        }
        if unique_atom_values.len() != atom_to_queries.len() {
            return Err(FactorizedFilterPlanError::IncompatibleAtomOrder);
        }

        let atom_positions = unique_atom_values
            .iter()
            .enumerate()
            .map(|(position, &value)| (value, position))
            .collect::<BTreeMap<_, _>>();
        let query_atoms = query_atom_values
            .into_iter()
            .map(|values| {
                values
                    .into_iter()
                    .map(|value| atom_positions[&value])
                    .collect()
            })
            .collect();
        let query_count = filters.len();
        let atoms = unique_atom_values
            .into_iter()
            .map(|value| {
                let query_indices = atom_to_queries.remove(&value).unwrap();
                FactorizedAtom {
                    value,
                    query_mask: QueryMask::from_query_indices(query_count, &query_indices),
                    query_indices,
                }
            })
            .collect();

        Ok(Self {
            key: key.expect("a non-empty batch always has a key"),
            query_count,
            atoms,
            query_atoms,
        })
    }

    pub fn key(&self) -> &PayloadKeyType {
        &self.key
    }

    pub fn query_count(&self) -> usize {
        self.query_count
    }

    pub fn atoms(&self) -> &[FactorizedAtom] {
        &self.atoms
    }

    pub fn atom_reference_count(&self) -> usize {
        self.query_atoms.iter().map(Vec::len).sum()
    }

    /// Atom IDs referenced by a query.
    pub fn query_atoms(&self, query_index: usize) -> Option<&[usize]> {
        self.query_atoms.get(query_index).map(Vec::as_slice)
    }
}

impl TryFrom<&[Filter]> for FactorizedFilterPlan {
    type Error = FactorizedFilterPlanError;

    fn try_from(filters: &[Filter]) -> Result<Self, Self::Error> {
        Self::try_from_filters(filters)
    }
}

/// Why a batch could not use the factorized fast path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FactorizedFilterPlanError {
    EmptyBatch,
    UnsupportedFilterShape {
        query_index: usize,
    },
    DifferentKey {
        query_index: usize,
        expected: PayloadKeyType,
        actual: PayloadKeyType,
    },
    IncompatibleAtomOrder,
}

impl Display for FactorizedFilterPlanError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyBatch => formatter.write_str("cannot factorize an empty filter batch"),
            Self::UnsupportedFilterShape { query_index } => {
                write!(
                    formatter,
                    "query {query_index} has an unsupported filter shape"
                )
            }
            Self::DifferentKey {
                query_index,
                expected,
                actual,
            } => write!(
                formatter,
                "query {query_index} uses key {actual}, expected shared key {expected}"
            ),
            Self::IncompatibleAtomOrder => formatter
                .write_str("query atom orders conflict and cannot preserve stock tie semantics"),
        }
    }
}

impl std::error::Error for FactorizedFilterPlanError {}

fn extract_integer_atoms(filter: &Filter) -> Option<(&PayloadKeyType, Vec<IntPayloadType>)> {
    if filter.should.is_some() || filter.min_should.is_some() || filter.must_not.is_some() {
        return None;
    }

    let [Condition::Field(field)] = filter.must.as_deref()? else {
        return None;
    };

    if field.range.is_some()
        || field.geo_bounding_box.is_some()
        || field.geo_radius.is_some()
        || field.geo_polygon.is_some()
        || field.values_count.is_some()
        || field.is_empty.is_some()
        || field.is_null.is_some()
    {
        return None;
    }

    let values = match field.r#match.as_ref()? {
        Match::Value(MatchValue {
            value: ValueVariants::Integer(value),
        }) => vec![*value],
        Match::Any(MatchAny {
            any: AnyVariants::Integers(values),
        }) => values.iter().copied().collect(),
        _ => return None,
    };

    Some((&field.key, values))
}

#[cfg(test)]
mod tests {
    use crate::types::{Condition, FieldCondition, Filter, Match, PayloadKeyType};

    use super::{FactorizedFilterPlan, FactorizedFilterPlanError};

    fn key(value: &str) -> PayloadKeyType {
        value.parse().unwrap()
    }

    fn value_filter(field: &str, value: i64) -> Filter {
        Filter::new_must(Condition::Field(FieldCondition::new_match(
            key(field),
            Match::from(value),
        )))
    }

    fn any_filter(field: &str, values: Vec<i64>) -> Filter {
        Filter::new_must(Condition::Field(FieldCondition::new_match(
            key(field),
            Match::from(values),
        )))
    }

    #[test]
    fn factorizes_value_and_any_by_unique_atom() {
        let filters = [
            value_filter("group_id", 10),
            any_filter("group_id", vec![10, 20, 30]),
            any_filter("group_id", vec![20, 30]),
        ];

        let plan = FactorizedFilterPlan::try_from_filters(&filters).unwrap();

        assert_eq!(plan.key(), &key("group_id"));
        assert_eq!(plan.query_count(), 3);
        assert_eq!(
            plan.atoms()
                .iter()
                .map(|atom| atom.value())
                .collect::<Vec<_>>(),
            vec![10, 20, 30]
        );
        assert_eq!(plan.atoms()[0].query_indices(), &[0, 1]);
        assert_eq!(plan.atoms()[0].reference_count(), 2);
        assert!(plan.atoms()[0].query_mask().contains(0));
        assert!(plan.atoms()[0].query_mask().contains(1));
        assert!(!plan.atoms()[0].query_mask().contains(2));
        assert_eq!(plan.atoms()[0].query_mask().count_ones(), 2);
        assert_eq!(plan.atoms()[1].query_indices(), &[1, 2]);
        assert_eq!(plan.atoms()[2].query_indices(), &[1, 2]);
        assert_eq!(plan.query_atoms(0), Some([0].as_slice()));
        assert_eq!(plan.query_atoms(1), Some([0, 1, 2].as_slice()));
        assert_eq!(plan.query_atoms(2), Some([1, 2].as_slice()));
    }

    #[test]
    fn accepts_empty_any_as_an_empty_query_atom_set() {
        let plan =
            FactorizedFilterPlan::try_from_filters(&[any_filter("group_id", vec![])]).unwrap();

        assert_eq!(plan.query_count(), 1);
        assert!(plan.atoms().is_empty());
        assert_eq!(plan.query_atoms(0), Some([].as_slice()));
    }

    #[test]
    fn duplicate_atoms_count_each_query_once() {
        let filters = [
            any_filter("group_id", vec![7, 7, 9, 7, 9]),
            value_filter("group_id", 7),
        ];

        let plan = FactorizedFilterPlan::try_from_filters(&filters).unwrap();

        assert_eq!(plan.query_atoms(0), Some([0, 1].as_slice()));
        assert_eq!(plan.atoms()[0].value(), 7);
        assert_eq!(plan.atoms()[0].query_indices(), &[0, 1]);
        assert_eq!(plan.atoms()[0].reference_count(), 2);
        assert_eq!(plan.atoms()[1].value(), 9);
        assert_eq!(plan.atoms()[1].query_indices(), &[0]);
        assert_eq!(plan.atoms()[1].reference_count(), 1);
    }

    #[test]
    fn preserves_non_sorted_atom_order_for_stock_ties() {
        let filters = [
            any_filter("group_id", vec![20, 10]),
            value_filter("group_id", 20),
        ];

        let plan = FactorizedFilterPlan::try_from_filters(&filters).unwrap();

        assert_eq!(
            plan.atoms()
                .iter()
                .map(|atom| atom.value())
                .collect::<Vec<_>>(),
            vec![20, 10]
        );
        assert_eq!(plan.query_atoms(0), Some([0, 1].as_slice()));
    }

    #[test]
    fn rejects_conflicting_atom_orders() {
        let error = FactorizedFilterPlan::try_from_filters(&[
            any_filter("group_id", vec![20, 10]),
            any_filter("group_id", vec![10, 20]),
        ])
        .unwrap_err();

        assert_eq!(error, FactorizedFilterPlanError::IncompatibleAtomOrder);
    }

    #[test]
    fn rejects_different_keys() {
        let error = FactorizedFilterPlan::try_from_filters(&[
            value_filter("group_id", 1),
            value_filter("tenant_id", 1),
        ])
        .unwrap_err();

        assert!(matches!(
            error,
            FactorizedFilterPlanError::DifferentKey { query_index: 1, .. }
        ));
    }

    #[test]
    fn rejects_filter_and_field_residuals() {
        let mut filter_with_clause_residual = value_filter("group_id", 1);
        filter_with_clause_residual.should = Some(vec![Condition::Field(
            FieldCondition::new_match(key("group_id"), Match::from(2_i64)),
        )]);
        assert!(matches!(
            FactorizedFilterPlan::try_from_filters(&[filter_with_clause_residual]),
            Err(FactorizedFilterPlanError::UnsupportedFilterShape { query_index: 0 })
        ));

        let mut field_with_residual =
            FieldCondition::new_match(key("group_id"), Match::from(1_i64));
        field_with_residual.is_empty = Some(false);
        let filter_with_field_residual = Filter::new_must(Condition::Field(field_with_residual));
        assert!(matches!(
            FactorizedFilterPlan::try_from_filters(&[filter_with_field_residual]),
            Err(FactorizedFilterPlanError::UnsupportedFilterShape { query_index: 0 })
        ));
    }

    #[test]
    fn query_mask_supports_more_than_sixty_four_queries() {
        let filters: Vec<_> = (0..130)
            .map(|query_index| value_filter("group_id", (query_index % 2) as i64))
            .collect();

        let plan = FactorizedFilterPlan::try_from_filters(&filters).unwrap();
        let odd_queries = &plan.atoms()[1];

        assert_eq!(odd_queries.reference_count(), 65);
        assert_eq!(odd_queries.query_mask().words().len(), 3);
        assert!(odd_queries.query_mask().contains(1));
        assert!(odd_queries.query_mask().contains(65));
        assert!(odd_queries.query_mask().contains(129));
        assert!(!odd_queries.query_mask().contains(128));
        assert!(!odd_queries.query_mask().contains(130));
    }
}
