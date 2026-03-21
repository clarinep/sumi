use thiserror::Error;

#[derive(Error, Debug, Clone)]
pub enum RenderError {
    #[error("card img not found: {0}")]
    CardNotFound(String),

    #[error("render timed out")]
    Timeout,

    #[error("internal error: {0}")]
    Internal(String),

    #[error("encoding error: {0}")]
    EncodeError(String),
}
