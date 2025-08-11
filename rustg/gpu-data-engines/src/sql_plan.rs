use std::ffi::c_void;
use crate::sql_types::ColumnTable;

/// SQL query execution plan nodes
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum PlanNodeType {
    Scan = 0,
    Filter = 1,
    Project = 2,
    HashJoin = 3,
    SortMergeJoin = 4,
    Aggregate = 5,
    Sort = 6,
    Limit = 7,
}

/// Filter predicate for WHERE clauses
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum ComparisonOp {
    EQ = 0, NE = 1, LT = 2, LE = 3, GT = 4, GE = 5, LIKE = 6, IN = 7
}

#[repr(C)]
pub struct FilterNode {
    pub column_id: u32,
    pub op: ComparisonOp,
    pub value: FilterValue,
}

#[repr(C)]
pub union FilterValue {
    pub int_val: i64,
    pub double_val: f64,
    pub string_val: *const i8,
    pub bool_val: bool,
    pub list_val: std::mem::ManuallyDrop<StringList>,
}

#[repr(C)]
pub struct StringList {
    pub values: *const c_void,
    pub count: usize,
}

/// Join operations
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum JoinType {
    Inner = 0,
    LeftOuter = 1,
    RightOuter = 2,
    FullOuter = 3,
}

#[repr(C)]
pub struct JoinNode {
    pub left_column: u32,
    pub right_column: u32,
    pub join_type: JoinType,
    pub left_table: *mut ColumnTable,
    pub right_table: *mut ColumnTable,
}

/// Aggregation operations
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum AggFunc {
    Sum = 0, Count = 1, Avg = 2, Min = 3, Max = 4, StdDev = 5
}

#[repr(C)]
pub struct AggregateNode {
    pub group_columns: *mut u32,
    pub num_group_cols: u32,
    pub func: AggFunc,
    pub agg_column: u32,
}

/// Query execution context
#[repr(C)]
pub struct QueryContext {
    pub intermediate_results: *mut ColumnTable,
    pub num_intermediate: u32,
    pub row_counts: *mut usize,
    pub execution_stream: *mut c_void,
}

/// Query execution plan
#[derive(Debug, Clone)]
pub struct QueryPlan {
    pub nodes: Vec<PlanNode>,
    pub estimated_cost: f64,
    pub estimated_rows: u64,
}

#[derive(Debug, Clone)]
pub struct PlanNode {
    pub node_type: PlanNodeType,
    pub children: Vec<usize>,
    pub params: NodeParams,
}

#[derive(Debug, Clone)]
pub enum NodeParams {
    Scan { table: String, columns: Vec<u32> },
    Filter { predicate: String },
    Project { expressions: Vec<String> },
    Join { join_type: JoinType, on: String },
    Aggregate { group_by: Vec<u32>, func: AggFunc },
    Sort { columns: Vec<u32>, ascending: Vec<bool> },
    Limit { count: usize },
}