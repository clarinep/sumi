use std::{
    error::Error,
    fmt::{Display, Formatter, Result as FmtResult},
};

#[derive(Debug, Clone)]
pub enum RenderError {
    CardNotFound(String),
    Timeout,
    Internal(String),
    EncodeError(String),
}

impl Display for RenderError {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::CardNotFound(s) => write!(f, "card img not found: {s}"),
            Self::Timeout => write!(f, "render timed out"),
            Self::Internal(s) => write!(f, "internal error: {s}"),
            Self::EncodeError(s) => write!(f, "encoding error: {s}"),
        }
    }
}

impl Error for RenderError {}
