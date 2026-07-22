use std::{
    fs::{self, File},
    io::Write,
    path::Path,
};

/// Helper to create a dummy webp file with a valid webp header.
/// This prevents `CardRenderer::new` and `CardCache` from failing to parse cards.
pub fn create_dummy_webp(dir: &Path, name: &str) {
    let path = dir.join(name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap();
    }
    
    // A tiny valid 1x1 WebP image base64 decoded
    let webp_data: &[u8] = &[
        0x52, 0x49, 0x46, 0x46, 0x1A, 0x00, 0x00, 0x00, 0x57, 0x45, 0x42, 0x50, 0x56, 0x50, 0x38,
        0x4C, 0x0D, 0x00, 0x00, 0x00, 0x2F, 0x00, 0x00, 0x00, 0x10, 0x07, 0x10, 0x11, 0x11, 0x88,
        0x88, 0xFE, 0x07, 0x00,
    ];

    let mut file = File::create(path).unwrap();
    file.write_all(webp_data).unwrap();
}
