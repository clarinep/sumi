//! Typed error declarations for the Koma encoder.

use std::fmt;

/// Result type alias for Koma operations.
pub type Result<T> = std::result::Result<T, KomaError>;

/// Errors that can occur during WebP encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KomaError {
    /// Provided dimensions are invalid (0 or exceeding WebP max limit of 16383x16383).
    InvalidDimensions { width: u32, height: u32 },
    /// Input buffer size does not match expected size for the given dimensions.
    InvalidBufferSize { expected: usize, actual: usize },
    /// Encoding failed in underlying VP8/WebP engine.
    EncodingFailed(String),
    /// Allocation error or buffer capacity exceeded.
    CapacityExceeded,
}

impl fmt::Display for KomaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { width, height } => {
                write!(f, "Invalid image dimensions: {}x{} (must be 1..=16383)", width, height)
            }
            Self::InvalidBufferSize { expected, actual } => {
                write!(
                    f,
                    "Invalid pixel buffer size: expected {} bytes, got {} bytes",
                    expected, actual
                )
            }
            Self::EncodingFailed(msg) => write!(f, "WebP encoding failed: {}", msg),
            Self::CapacityExceeded => write!(f, "Encoder buffer capacity exceeded"),
        }
    }
}

impl std::error::Error for KomaError {}
