//! VP8 lossy video bitstream encoder implementation according to RFC 6386.
//!
//! Provides modular kernels for boolean arithmetic encoding, discrete cosine transforms,
//! Walsh-Hadamard transforms, coefficient tables, and macroblock frame encoding.

pub mod bool_coder;
pub mod config;
pub mod frame;
pub mod tables;
pub mod transform;

#[doc(inline)]
pub use bool_coder::BoolEncoder;
#[doc(inline)]
pub use config::{EncoderConfig, EncoderConfigBuilder};
#[doc(inline)]
pub use frame::{encode_coeffs_block, encode_lossy_frame, quality_to_q_index};
#[doc(inline)]
pub use tables::{
    AC_QLOOKUP, DC_QLOOKUP, PCAT1, PCAT2, PCAT3, PCAT4, PCAT5, PCAT6, PROB_EOB, PROB_ONE, PROB_TWO,
    PROB_ZERO, VP8_START_CODE, ZIGZAG,
};
#[doc(inline)]
pub use transform::{fdct_4x4, forward_wht_4x4, idct_4x4, inverse_wht_4x4};
