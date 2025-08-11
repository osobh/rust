use std::ffi::c_void;

/// SQL data types
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SQLDataType {
    Int64 = 0,
    Double = 1,
    Varchar = 2,
    Boolean = 3,
    Timestamp = 4,
    Decimal = 5,
}

/// Column metadata
#[repr(C)]
#[derive(Debug, Clone)]
pub struct ColumnInfo {
    pub data_type: SQLDataType,
    pub column_id: u32,
    pub name: [i8; 64],
    pub nullable: bool,
    pub max_length: u32, // For VARCHAR
}

/// Table schema
#[repr(C)]
pub struct TableSchema {
    pub columns: *mut ColumnInfo,
    pub num_columns: u32,
    pub num_rows: u64,
    pub table_name: [i8; 64],
}

/// Columnar table storage on GPU
#[repr(C)]
pub struct ColumnTable {
    pub column_data: *mut *mut c_void,  // Array of column data pointers
    pub null_masks: *mut *mut bool,     // Null masks for each column
    pub schema: TableSchema,
    pub capacity: u64,                  // Allocated row capacity
    pub row_ids: *mut u32,             // Row identifier mapping
}

/// SQL Value types for query results
#[derive(Debug, Clone)]
pub enum SQLValue {
    Int(i64),
    Double(f64),
    String(String),
    Boolean(bool),
    Timestamp(i64),
    Decimal(String),
    Null,
}

/// Query result structure
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub data_types: Vec<SQLDataType>,
    pub rows: Vec<Vec<SQLValue>>,
    pub execution_time_ms: f64,
    pub rows_scanned: u64,
    pub throughput_gbps: f64,
}

/// GPU Result structure
#[repr(C)]
pub struct RbResult {
    pub code: i32,
    pub msg: [u8; 256],
    pub millis: f64,
    pub value: usize,
}

#[allow(non_camel_case_types)]
#[repr(i32)]
pub enum RbStatus {
    Ok = 0,
    NotInitialized = 1,
    Cuda = 2,
    Thrust = 3,
    InvalidArg = 4,
    Oom = 5,
    KernelLaunch = 6,
    DeviceNotFound = 7,
    Unknown = 255,
}

#[repr(C)]
pub struct TestResult {
    pub success: bool,
    pub query_throughput_gbps: f32,
    pub rows_per_second: f32,
    pub rows_processed: usize,
    pub elapsed_ms: f64,
    pub error_msg: [i8; 256],
}