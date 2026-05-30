use std::fmt::{Debug, Formatter, Result as FmtResult};

/// raw uncompressed rgba image
/// this will act as a container that will replace our previous `image::RgbaImage`.
pub struct RawCardImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Box<[u8]>,
}

impl Debug for RawCardImage {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("RawCardImage")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("pixels_len", &self.pixels.len())
            .finish()
    }
}
