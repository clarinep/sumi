use std::fmt::{Debug, Formatter, Result as FmtResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Point<T> {
    pub(super) x: T,
    pub(super) y: T,
}

impl<T> Point<T> {
    #[inline]
    pub(super) const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Size<T> {
    pub(super) width: T,
    pub(super) height: T,
}

impl<T> Size<T> {
    #[inline]
    pub(super) const fn new(width: T, height: T) -> Self {
        Self { width, height }
    }
}

// pixels uncompressed rgba image
// this will act as a container that will replace our previous image::RgbaImage
#[derive(Clone, PartialEq, Eq)]
pub(super) struct RawCardImage {
    pub(super) size: Size<u32>,
    pub(super) pixels: Box<[u8]>,
}

impl Debug for RawCardImage {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("RawCardImage")
            .field("width", &self.size.width)
            .field("height", &self.size.height)
            .field("pixels_len", &self.pixels.len())
            .finish()
    }
}
