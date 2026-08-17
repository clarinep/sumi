//! Error types and result definitions for the koma encoder.
//!
//! This module provides the [`KomaError`] type and [`Result`] alias.

use std::fmt;

/// A specialized [`Result`](std::result::Result) type for koma operations.
pub type Result<T> = std::result::Result<T, KomaError>;

/// Represents errors that can occur during WebP encoding or validation.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KomaError {
    /// Canvas dimensions are 0 or exceed the maximum allowed WebP size of 16,383 by 16,383 pixels.
    InvalidDimensions {
        /// The width provided by the caller.
        width: u32,
        /// The height provided by the caller.
        height: u32,
    },
    /// The input RGBA pixel slice is smaller than the required byte count.
    BufferTooSmall {
        /// The expected minimum byte length (`width * height * 4`).
        expected: usize,
        /// The actual byte length of the provided buffer.
        actual: usize,
    },
    /// An internal bitstream or container assembly failure occurred.
    EncodeFailure(&'static str),
}

impl fmt::Display for KomaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { width, height } => {
                write!(f, "invalid dimensions: {width}x{height} pixels (must be 1 to 16383)")
            }
            Self::BufferTooSmall { expected, actual } => {
                write!(
                    f,
                    "buffer size mismatch: expected at least {expected} bytes, received {actual} bytes"
                )
            }
            Self::EncodeFailure(reason) => write!(f, "encoding failed: {reason}"),
        }
    }
}

impl std::error::Error for KomaError {}
