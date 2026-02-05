//! UUID generation and validation utilities.

use thiserror::Error;
use uuid::Uuid;

/// Errors that can occur during UUID validation.
///
/// # Examples
///
/// ```rust
/// use delaunay::core::util::UuidValidationError;
///
/// let err = UuidValidationError::NilUuid;
/// assert!(matches!(err, UuidValidationError::NilUuid));
/// ```
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum UuidValidationError {
    /// The UUID is nil (all zeros), which is not allowed.
    #[error("UUID is nil (all zeros) which is not allowed")]
    NilUuid,
    /// The UUID is not version 4.
    #[error("UUID is not version 4: expected version 4, found version {found}")]
    InvalidVersion {
        /// The version number that was found.
        found: usize,
    },
}

/// Validates that a UUID is not nil and is version 4.
///
/// This function performs comprehensive UUID validation to ensure that UUIDs
/// used throughout the system meet our requirements:
/// - Must not be nil (all zeros)
/// - Must be version 4 (randomly generated)
///
/// # Arguments
///
/// * `uuid` - The UUID to validate
///
/// # Returns
///
/// Returns `Ok(())` if the UUID is valid, or a `UuidValidationError` if invalid.
///
/// # Errors
///
/// Returns `UuidValidationError::NilUuid` if the UUID is nil,
/// or `UuidValidationError::InvalidVersion` if the UUID is not version 4.
///
/// # Examples
///
/// ```
/// use delaunay::core::collections::Uuid;
/// use delaunay::core::util::{make_uuid, validate_uuid};
///
/// // Valid UUID (version 4)
/// let valid_uuid = make_uuid();
/// assert!(validate_uuid(&valid_uuid).is_ok());
///
/// // Invalid UUID (nil)
/// let nil_uuid = Uuid::nil();
/// assert!(validate_uuid(&nil_uuid).is_err());
/// ```
pub const fn validate_uuid(uuid: &Uuid) -> Result<(), UuidValidationError> {
    // Check if UUID is nil
    if uuid.is_nil() {
        return Err(UuidValidationError::NilUuid);
    }

    // Check if UUID is version 4
    let version = uuid.get_version_num();
    if version != 4 {
        return Err(UuidValidationError::InvalidVersion { found: version });
    }

    Ok(())
}

/// The function `make_uuid` generates a version 4 [Uuid].
///
/// # Returns
///
/// a randomly generated [Uuid] (Universally Unique Identifier) using the
/// `new_v4` method from the [Uuid] struct.
///
/// # Example
///
/// ```
/// use delaunay::core::util::make_uuid;
/// let uuid = make_uuid();
/// assert_eq!(uuid.get_version_num(), 4);
/// ```
#[must_use]
pub fn make_uuid() -> Uuid {
    Uuid::new_v4()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_uuid_comprehensive() {
        // Sub-test: UUID creation uniqueness
        let uuid1 = make_uuid();
        let uuid2 = make_uuid();
        let uuid3 = make_uuid();
        assert_ne!(uuid1, uuid2, "UUIDs should be unique");
        assert_ne!(uuid1, uuid3, "UUIDs should be unique");
        assert_ne!(uuid2, uuid3, "UUIDs should be unique");
        assert_eq!(uuid1.get_version_num(), 4, "Should be version 4");
        assert_eq!(uuid2.get_version_num(), 4, "Should be version 4");
        assert_eq!(uuid3.get_version_num(), 4, "Should be version 4");

        // Sub-test: UUID format validation
        let uuid = make_uuid();
        let uuid_string = uuid.to_string();
        assert_eq!(
            uuid_string.len(),
            36,
            "UUID should be 36 chars (with hyphens)"
        );
        assert_eq!(
            uuid_string.chars().filter(|&c| c == '-').count(),
            4,
            "UUID should have 4 hyphens"
        );
        let parts: Vec<&str> = uuid_string.split('-').collect();
        assert_eq!(parts.len(), 5, "UUID should have 5 hyphen-separated parts");
        assert_eq!(parts[0].len(), 8, "First part should be 8 chars");
        assert_eq!(parts[1].len(), 4, "Second part should be 4 chars");
        assert_eq!(parts[2].len(), 4, "Third part should be 4 chars");
        assert_eq!(parts[3].len(), 4, "Fourth part should be 4 chars");
        assert_eq!(parts[4].len(), 12, "Fifth part should be 12 chars");

        // Sub-test: Valid UUID (version 4)
        let valid_uuid = make_uuid();
        assert!(
            validate_uuid(&valid_uuid).is_ok(),
            "Valid v4 UUID should pass validation"
        );

        // Test nil UUID
        let nil_uuid = Uuid::nil();
        let nil_result = validate_uuid(&nil_uuid);
        assert!(nil_result.is_err(), "Nil UUID should fail validation");
        match nil_result {
            Err(UuidValidationError::NilUuid) => (), // Expected
            Err(other) => panic!("Expected NilUuid error, got: {other:?}"),
            Ok(()) => panic!("Expected error for nil UUID, but validation passed"),
        }

        // Test wrong version UUID (version 1)
        let v1_uuid = Uuid::parse_str("550e8400-e29b-11d4-a716-446655440000").unwrap();
        assert_eq!(v1_uuid.get_version_num(), 1);
        let version_result = validate_uuid(&v1_uuid);
        assert!(
            version_result.is_err(),
            "Non-v4 UUID should fail validation"
        );
        match version_result {
            Err(UuidValidationError::InvalidVersion { found }) => {
                assert_eq!(found, 1, "Should report correct version number");
            }
            Err(other) => panic!("Expected InvalidVersion error, got: {other:?}"),
            Ok(()) => panic!("Expected error for version 1 UUID, but validation passed"),
        }

        // Test error display formatting
        let nil_error = UuidValidationError::NilUuid;
        let nil_error_string = format!("{nil_error}");
        assert!(
            nil_error_string.contains("nil"),
            "Nil error message should contain 'nil'"
        );
        assert!(
            nil_error_string.contains("not allowed"),
            "Nil error message should mention 'not allowed'"
        );

        let version_error = UuidValidationError::InvalidVersion { found: 3 };
        let version_error_string = format!("{version_error}");
        assert!(
            version_error_string.contains("version 4"),
            "Version error should mention 'version 4'"
        );
        assert!(
            version_error_string.contains("found version 3"),
            "Version error should show found version"
        );

        // Test PartialEq for UuidValidationError
        let error1 = UuidValidationError::NilUuid;
        let error2 = UuidValidationError::NilUuid;
        assert_eq!(error1, error2, "Same nil errors should be equal");

        let error3 = UuidValidationError::InvalidVersion { found: 2 };
        let error4 = UuidValidationError::InvalidVersion { found: 2 };
        assert_eq!(error3, error4, "Same version errors should be equal");

        let error5 = UuidValidationError::InvalidVersion { found: 3 };
        assert_ne!(
            error3, error5,
            "Different version errors should not be equal"
        );
        assert_ne!(error1, error3, "Different error types should not be equal");
    }
}
