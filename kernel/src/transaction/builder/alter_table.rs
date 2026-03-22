//! Builder for alter-table (schema evolution) transactions.
//!
//! This module contains [`AlterTableTransactionBuilder`], which validates and constructs an
//! [`AlterTableTransaction`] from user-provided schema operations.
//!
//! Use [`alter_table()`](super::super::alter_table::alter_table) as the entry point rather
//! than constructing the builder directly.

use std::sync::Arc;

use uuid::Uuid;

use crate::committer::Committer;
use crate::schema::{ColumnMetadataKey, MetadataValue, SchemaRef, StructField, StructType};
use crate::snapshot::SnapshotRef;
use crate::table_features::{ColumnMappingMode, Operation};
use crate::table_properties::COLUMN_MAPPING_MAX_COLUMN_ID;
use crate::transaction::alter_table::AlterTableTransaction;
use crate::{DeltaResult, Engine, Error};

/// Operations that can be applied to evolve a table schema.
///
/// Operations are validated and applied in order during
/// [`AlterTableTransactionBuilder::build`]. Each operation sees the schema
/// state after all prior operations have been applied.
#[derive(Debug, Clone)]
pub(crate) enum SchemaOperation {
    /// Add a new top-level column.
    AddColumn { field: StructField },
}

/// Result of applying schema operations to a table schema.
#[derive(Debug)]
pub(crate) struct SchemaEvolutionResult {
    /// The evolved schema after all operations are applied.
    pub schema: SchemaRef,
    /// The new max column ID (if column mapping is active and columns were added).
    /// Used to update `delta.columnMapping.maxColumnId` in table properties.
    pub new_max_column_id: Option<i64>,
}

/// Builder for constructing an [`AlterTableTransaction`].
///
/// Created via [`alter_table()`](super::super::alter_table::alter_table). Schema operations
/// are buffered and validated when [`build`](Self::build) is called.
///
/// # Example
///
/// ```rust,no_run
/// use delta_kernel::schema::{StructField, DataType};
/// use delta_kernel::committer::FileSystemCommitter;
/// use delta_kernel::snapshot::SnapshotRef;
/// # use delta_kernel::Engine;
/// # fn example(engine: &dyn Engine, snapshot: SnapshotRef) -> delta_kernel::DeltaResult<()> {
///
/// let result = snapshot
///     .alter_table()
///     .add_column(StructField::nullable("email", DataType::STRING))
///     .build(engine, Box::new(FileSystemCommitter::new()))?
///     .commit(engine)?;
/// # Ok(())
/// # }
/// ```
pub struct AlterTableTransactionBuilder {
    /// The snapshot of the existing table to evolve.
    snapshot: SnapshotRef,
    /// Ordered list of schema operations to apply.
    operations: Vec<SchemaOperation>,
}

impl AlterTableTransactionBuilder {
    /// Create a new builder for the given snapshot.
    pub(crate) fn new(snapshot: impl Into<SnapshotRef>) -> Self {
        Self {
            snapshot: snapshot.into(),
            operations: vec![],
        }
    }

    /// Add a new top-level column to the table schema.
    ///
    /// The field must not already exist in the schema. If the table has existing data files,
    /// the field should be nullable -- existing files will read NULL for this column.
    ///
    /// If column mapping is enabled, the builder automatically assigns a new column ID and
    /// physical name. Otherwise the logical name is used as-is.
    ///
    /// # Arguments
    ///
    /// * `field` - The [`StructField`] to add (name, type, nullability, metadata)
    ///
    /// # Errors (at build time)
    ///
    /// - Column with the same name already exists at the top level
    /// - Field is non-nullable (no default value support)
    pub fn add_column(mut self, field: StructField) -> Self {
        self.operations.push(SchemaOperation::AddColumn { field });
        self
    }

    /// Builds an [`AlterTableTransaction`] that can be committed to evolve the schema.
    ///
    /// This method:
    /// 1. Validates the table supports writes
    /// 2. Applies each operation sequentially against the evolving schema
    /// 3. Assigns new column IDs and physical names for added columns
    ///    (when column mapping is enabled)
    /// 4. Updates `delta.columnMapping.maxColumnId` in table properties
    /// 5. Constructs new Metadata action with evolved schema
    /// 6. Validates the new configuration via `TableConfiguration::try_new_from()`
    ///
    /// # Arguments
    ///
    /// * `engine` - The engine instance for validation
    /// * `committer` - The committer to use for the transaction
    ///
    /// # Errors
    ///
    /// - No operations were specified
    /// - Any individual operation fails validation (see per-method errors above)
    /// - Table does not support writes (unsupported features)
    /// - `TableConfiguration` validation fails on the evolved state
    pub fn build(
        self,
        _engine: &dyn Engine,
        committer: Box<dyn Committer>,
    ) -> DeltaResult<AlterTableTransaction> {
        if self.operations.is_empty() {
            return Err(Error::generic(
                "At least one schema operation must be specified for ALTER TABLE",
            ));
        }

        let table_config = self.snapshot.table_configuration();

        // Validate the table supports writes
        table_config.ensure_operation_supported(Operation::Write)?;

        let schema = table_config.logical_schema();
        let column_mapping_mode = table_config.column_mapping_mode();

        // Parse current max column ID from table properties
        let current_max_column_id: i64 = table_config
            .table_properties()
            .unknown_properties
            .get(COLUMN_MAPPING_MAX_COLUMN_ID)
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        // Apply all schema operations
        let result = apply_schema_operations(
            &schema,
            &self.operations,
            column_mapping_mode,
            current_max_column_id,
        )?;

        // Build evolved metadata
        let mut evolved_metadata = table_config.metadata().clone().with_schema(result.schema)?;

        // Update max column ID in configuration if it changed
        if let Some(new_max_id) = result.new_max_column_id {
            let mut config = table_config.metadata().configuration().clone();
            config.insert(
                COLUMN_MAPPING_MAX_COLUMN_ID.to_string(),
                new_max_id.to_string(),
            );
            evolved_metadata = evolved_metadata.with_configuration(config);
        }

        // Validate the evolved configuration is valid
        let _new_config = crate::table_configuration::TableConfiguration::try_new_from(
            table_config,
            Some(evolved_metadata.clone()),
            None, // no protocol change
            self.snapshot.version(),
        )?;

        AlterTableTransaction::try_new_alter_table(self.snapshot, evolved_metadata, committer)
    }
}

/// Applies a sequence of schema operations to the given schema.
///
/// Each operation is validated against the current schema state (after prior operations have
/// been applied). Returns the final schema and any metadata changes needed.
///
/// # Arguments
///
/// * `schema` - The current table schema (logical schema from snapshot)
/// * `operations` - Ordered list of schema operations to apply
/// * `column_mapping_mode` - The table's column mapping mode
/// * `current_max_column_id` - Current `delta.columnMapping.maxColumnId` value
/// # Errors
///
/// Returns an error if any operation fails validation. The error message identifies which
/// operation failed and why.
pub(crate) fn apply_schema_operations(
    schema: &SchemaRef,
    operations: &[SchemaOperation],
    column_mapping_mode: ColumnMappingMode,
    current_max_column_id: i64,
) -> DeltaResult<SchemaEvolutionResult> {
    let mut evolving_schema: StructType = schema.as_ref().clone();
    let mut max_column_id = current_max_column_id;

    for op in operations {
        match op {
            SchemaOperation::AddColumn { field } => {
                // Validate field doesn't already exist
                if evolving_schema.field(&field.name).is_some() {
                    return Err(Error::generic(format!(
                        "Cannot add column '{}': column already exists in the schema",
                        field.name
                    )));
                }

                // New columns must be nullable (existing data files will have NULL for this column)
                if !field.is_nullable() {
                    return Err(Error::generic(format!(
                        "Cannot add non-nullable column '{}': existing data files would have no \
                         value for this column. Use a nullable column instead.",
                        field.name
                    )));
                }

                let new_field = if column_mapping_mode != ColumnMappingMode::None {
                    assign_alter_column_mapping(field, &mut max_column_id)?
                } else {
                    field.clone()
                };

                // Append the new field at the end of the schema
                evolving_schema = evolving_schema.with_field_inserted_after(None, new_field)?;
            }
        }
    }

    let new_max_column_id = (max_column_id != current_max_column_id).then_some(max_column_id);

    Ok(SchemaEvolutionResult {
        schema: Arc::new(evolving_schema),
        new_max_column_id,
    })
}

/// Assigns column mapping metadata (column ID and physical name) to a field for ALTER TABLE
/// ADD COLUMN. Unlike the CREATE TABLE variant, this does not reject pre-existing metadata
/// (it overwrites it).
fn assign_alter_column_mapping(field: &StructField, max_id: &mut i64) -> DeltaResult<StructField> {
    let mut new_field = field.clone();

    // Assign new column ID
    *max_id += 1;
    new_field.metadata.insert(
        ColumnMetadataKey::ColumnMappingId.as_ref().to_string(),
        MetadataValue::Number(*max_id),
    );

    // Assign physical name
    let physical_name = format!("col-{}", Uuid::new_v4());
    new_field.metadata.insert(
        ColumnMetadataKey::ColumnMappingPhysicalName
            .as_ref()
            .to_string(),
        MetadataValue::String(physical_name),
    );

    // Recursively process nested types
    new_field.data_type = process_nested_data_type_for_alter(&field.data_type, max_id)?;

    Ok(new_field)
}

/// Process nested data types to assign column mapping metadata to any nested struct fields
/// during ALTER TABLE operations.
fn process_nested_data_type_for_alter(
    data_type: &crate::schema::DataType,
    max_id: &mut i64,
) -> DeltaResult<crate::schema::DataType> {
    use crate::schema::{ArrayType, DataType, MapType};
    match data_type {
        DataType::Struct(inner) => {
            let new_fields: Vec<StructField> = inner
                .fields()
                .map(|field| assign_alter_column_mapping(field, max_id))
                .collect::<DeltaResult<_>>()?;
            let new_inner = StructType::try_new(new_fields)?;
            Ok(DataType::Struct(Box::new(new_inner)))
        }
        DataType::Array(array_type) => {
            let new_element_type =
                process_nested_data_type_for_alter(array_type.element_type(), max_id)?;
            Ok(DataType::Array(Box::new(ArrayType::new(
                new_element_type,
                array_type.contains_null(),
            ))))
        }
        DataType::Map(map_type) => {
            let new_key_type = process_nested_data_type_for_alter(map_type.key_type(), max_id)?;
            let new_value_type = process_nested_data_type_for_alter(map_type.value_type(), max_id)?;
            Ok(DataType::Map(Box::new(MapType::new(
                new_key_type,
                new_value_type,
                map_type.value_contains_null(),
            ))))
        }
        // Primitive and Variant types don't contain nested struct fields
        DataType::Primitive(_) | DataType::Variant(_) => Ok(data_type.clone()),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::schema::{DataType, StructField, StructType};
    use crate::table_features::ColumnMappingMode;

    use super::*;

    // Helper to create a SchemaRef for tests
    fn test_schema() -> SchemaRef {
        Arc::new(StructType::new_unchecked(vec![
            StructField::new("id", DataType::INTEGER, false),
            StructField::nullable("name", DataType::STRING),
            StructField::nullable("age", DataType::INTEGER),
        ]))
    }

    #[test]
    fn add_column_no_column_mapping() {
        let schema = test_schema();
        let ops = vec![SchemaOperation::AddColumn {
            field: StructField::nullable("email", DataType::STRING),
        }];

        let result = apply_schema_operations(&schema, &ops, ColumnMappingMode::None, 0)
            .expect("add_column should succeed");

        assert_eq!(result.schema.num_fields(), 4);
        let email_field = result
            .schema
            .field("email")
            .expect("email field should exist");
        assert_eq!(email_field.data_type(), &DataType::STRING);
        assert!(email_field.is_nullable());
        assert!(result.new_max_column_id.is_none());
    }

    #[test]
    fn add_column_with_column_mapping() {
        let schema = test_schema();
        let ops = vec![SchemaOperation::AddColumn {
            field: StructField::nullable("email", DataType::STRING),
        }];

        let result = apply_schema_operations(&schema, &ops, ColumnMappingMode::Name, 3)
            .expect("add_column should succeed");

        assert_eq!(result.schema.num_fields(), 4);
        let email_field = result
            .schema
            .field("email")
            .expect("email field should exist");

        // Check column mapping metadata was assigned
        let col_id = email_field
            .get_config_value(&ColumnMetadataKey::ColumnMappingId)
            .expect("should have column ID");
        assert_eq!(col_id, &MetadataValue::Number(4));

        let phys_name = email_field
            .get_config_value(&ColumnMetadataKey::ColumnMappingPhysicalName)
            .expect("should have physical name");
        assert!(matches!(phys_name, MetadataValue::String(s) if s.starts_with("col-")));

        assert_eq!(result.new_max_column_id, Some(4));
    }

    #[test]
    fn add_duplicate_column_fails() {
        let schema = test_schema();
        let ops = vec![SchemaOperation::AddColumn {
            field: StructField::nullable("name", DataType::STRING),
        }];

        let err = apply_schema_operations(&schema, &ops, ColumnMappingMode::None, 0)
            .expect_err("duplicate column should fail");
        assert!(err.to_string().contains("already exists"));
    }

    #[test]
    fn add_non_nullable_column_fails() {
        let schema = test_schema();
        let ops = vec![SchemaOperation::AddColumn {
            field: StructField::new("email", DataType::STRING, false),
        }];

        let err = apply_schema_operations(&schema, &ops, ColumnMappingMode::None, 0)
            .expect_err("non-nullable should fail");
        assert!(err.to_string().contains("non-nullable"));
    }

    #[test]
    fn empty_operations_succeeds_at_schema_level() {
        let schema = test_schema();
        let result = apply_schema_operations(&schema, &[], ColumnMappingMode::None, 0)
            .expect("empty ops should succeed at schema level");
        assert_eq!(result.schema.num_fields(), 3);
    }
}
