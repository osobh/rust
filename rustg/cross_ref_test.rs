/// Main struct that uses Helper
pub struct Main {
    /// Field referencing Helper::helper_method
    pub helper: Helper,
}

/// Helper struct referenced by Main
pub struct Helper;

impl Helper {
    /// Method referenced in Main::helper
    pub fn helper_method(&self) {}
}