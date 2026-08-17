//! High-throughput memory pooling and zero-allocation scratch buffers.
//!
//! Reuses large intermediate pixel planes, transform scratchpads, and bitstream buffers
//! across multiple rendering jobs using lock-free queue structures (`ArrayQueue`).

use crossbeam_queue::ArrayQueue;
use std::sync::LazyLock;

use crate::alpha::extract_and_compress_alpha;
use crate::color::rgba_to_yuv420p;
use crate::error::{KomaError, Result};
use crate::riff::assemble_webp;
use crate::vp8::{encode_lossy_frame, EncoderConfig};

/// Global lock-free pool of pre-allocated [`EncoderScratch`] buffers.
pub static ENCODER_SCRATCH_POOL: LazyLock<ScratchPool> =
    LazyLock::new(|| ScratchPool::new(16));

/// A lock-free pool of [`EncoderScratch`] instances.
#[derive(Debug)]
pub struct ScratchPool {
    queue: ArrayQueue<EncoderScratch>,
}

impl ScratchPool {
    /// Creates a new pool with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            queue: ArrayQueue::new(capacity),
        }
    }

    /// Retrieves a scratch buffer from the pool or allocates a new one if exhausted.
    pub fn get(&self) -> ScratchGuard {
        let scratch = self.queue.pop().unwrap_or_else(EncoderScratch::new);
        ScratchGuard {
            scratch: Some(scratch),
        }
    }

    /// Returns a scratch buffer back to the pool.
    pub fn recycle(&self, scratch: EncoderScratch) {
        let _ = self.queue.push(scratch);
    }
}

/// Pre-allocated working buffers matching canvas dimensions.
///
/// Avoids all dynamic heap allocations during high-frequency card drop rendering loops.
#[derive(Debug, Default)]
pub struct EncoderScratch {
    pub y_plane: Vec<u8>,
    pub u_plane: Vec<u8>,
    pub v_plane: Vec<u8>,
    pub recon_y: Vec<u8>,
    pub recon_u: Vec<u8>,
    pub recon_v: Vec<u8>,
    pub alpha_plane: Vec<u8>,
    pub alph_chunk: Vec<u8>,
    pub vp8_entropy: Vec<u8>,
    pub vp8_frame: Vec<u8>,
    pub final_webp: Vec<u8>,
}

impl EncoderScratch {
    /// Creates a new empty [`EncoderScratch`] instance.
    pub fn new() -> Self {
        Self::default()
    }

    /// Prepares internal vectors to hold the requested padded dimensions.
    #[inline]
    pub fn prepare(&mut self, pad_width: usize, pad_height: usize, total_pixels: usize) {
        let y_len = pad_width * pad_height;
        let uv_len = (pad_width / 2) * (pad_height / 2);

        if self.y_plane.len() < y_len {
            self.y_plane.resize(y_len, 0);
            self.recon_y.resize(y_len, 0);
        }
        if self.u_plane.len() < uv_len {
            self.u_plane.resize(uv_len, 0);
            self.recon_u.resize(uv_len, 0);
            self.v_plane.resize(uv_len, 0);
            self.recon_v.resize(uv_len, 0);
        }
        if self.alpha_plane.len() < total_pixels {
            self.alpha_plane.resize(total_pixels, 0);
        }
        if self.alph_chunk.capacity() < total_pixels {
            self.alph_chunk.reserve(total_pixels);
        }
        if self.vp8_entropy.capacity() < total_pixels {
            self.vp8_entropy.reserve(total_pixels);
        }
        if self.vp8_frame.capacity() < total_pixels {
            self.vp8_frame.reserve(total_pixels);
        }
        if self.final_webp.capacity() < total_pixels {
            self.final_webp.reserve(total_pixels);
        }
    }

    /// Encodes raw RGBA pixel data using the pooled scratch memory.
    ///
    /// # Errors
    ///
    /// Returns [`KomaError::InvalidDimensions`] if `width == 0` or `height == 0`.
    /// Returns [`KomaError::BufferTooSmall`] if `rgba.len() < width * height * 4`.
    pub fn encode_rgba(
        &mut self,
        width: u32,
        height: u32,
        rgba: &[u8],
        config: &EncoderConfig,
    ) -> Result<&[u8]> {
        if width == 0 || height == 0 {
            tracing::error!(
                "encode_rgba failed: invalid dimensions {}x{} (must be > 0)",
                width, height
            );
            return Err(KomaError::InvalidDimensions { width, height });
        }

        let total_pixels = (width as usize) * (height as usize);
        let expected_bytes = total_pixels * 4;
        if rgba.len() < expected_bytes {
            tracing::error!(
                "encode_rgba failed: buffer too small. expected at least {} bytes, got {}",
                expected_bytes,
                rgba.len()
            );
            return Err(KomaError::BufferTooSmall {
                expected: expected_bytes,
                actual: rgba.len(),
            });
        }

        let pad_width = ((width as usize + 15) / 16) * 16;
        let pad_height = ((height as usize + 15) / 16) * 16;

        self.prepare(pad_width, pad_height, total_pixels);

        // 1. Extract and compress alpha plane (with zero-alloc scratch reuse)
        let has_alpha = extract_and_compress_alpha(
            rgba,
            width as usize,
            height as usize,
            &mut self.alpha_plane,
            &mut self.alph_chunk,
        );
        let alpha_payload = if has_alpha {
            Some(self.alph_chunk.as_slice())
        } else {
            None
        };

        // 2. Convert RGBA to planar YUV420p with auto-vectorized contiguous row loops
        rgba_to_yuv420p(
            rgba,
            width as usize,
            height as usize,
            pad_width,
            pad_height,
            &mut self.y_plane,
            &mut self.u_plane,
            &mut self.v_plane,
        );

        // 3. Encode lossy VP8 frame using calibrated RFC 6386 probabilities and lookup tables
        encode_lossy_frame(
            width,
            height,
            pad_width,
            pad_height,
            &self.y_plane,
            &self.u_plane,
            &self.v_plane,
            &mut self.recon_y,
            &mut self.recon_u,
            &mut self.recon_v,
            &mut self.vp8_entropy,
            &mut self.vp8_frame,
            config,
        );

        // 4. Assemble WebP / VP8X container
        assemble_webp(
            width,
            height,
            &self.vp8_frame,
            alpha_payload,
            &mut self.final_webp,
        )?;

        Ok(&self.final_webp)
    }
}

/// RAII guard providing scoped access to a pooled [`EncoderScratch`] buffer.
///
/// Automatically returns the underlying memory buffer back to the global pool on drop.
#[derive(Debug)]
pub struct ScratchGuard {
    scratch: Option<EncoderScratch>,
}

impl std::ops::Deref for ScratchGuard {
    type Target = EncoderScratch;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.scratch.as_ref().unwrap()
    }
}

impl std::ops::DerefMut for ScratchGuard {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.scratch.as_mut().unwrap()
    }
}

impl Drop for ScratchGuard {
    fn drop(&mut self) {
        if let Some(scratch) = self.scratch.take() {
            ENCODER_SCRATCH_POOL.recycle(scratch);
        }
    }
}
