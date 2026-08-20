//! RIFF WebP container packaging for Simple and Extended WebP bitstreams.

use bytes::Bytes;

/// Writes a 32-bit little-endian FourCC tag and chunk length.
#[inline(always)]
fn write_chunk_header(out: &mut Vec<u8>, tag: &[u8; 4], len: u32) {
    out.extend_from_slice(tag);
    out.extend_from_slice(&len.to_le_bytes());
}

/// Packages a VP8 lossy bitstream and optional alpha stream into a complete RIFF WebP file.
pub fn package_webp_riff(
    width: u32,
    height: u32,
    vp8_payload: &[u8],
    alph_payload: Option<&[u8]>,
) -> Bytes {
    let has_alpha = alph_payload.is_some();

    if !has_alpha {
        // Simple WebP File Format:
        // 'RIFF' (4) + FileSize (4) + 'WEBP' (4) + 'VP8 ' (4) + VP8Size (4) + Payload + [pad]
        let vp8_pad = vp8_payload.len() % 2;
        let riff_payload_len = 4 + 8 + vp8_payload.len() + vp8_pad;
        let total_file_size = 8 + riff_payload_len;

        let mut out = Vec::with_capacity(total_file_size);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(riff_payload_len as u32).to_le_bytes());
        out.extend_from_slice(b"WEBP");

        write_chunk_header(&mut out, b"VP8 ", vp8_payload.len() as u32);
        out.extend_from_slice(vp8_payload);
        if vp8_pad > 0 {
            out.push(0);
        }

        Bytes::from(out)
    } else {
        // Extended WebP File Format (VP8X + ALPH + VP8):
        let alph = alph_payload.unwrap();
        let vp8x_data = crate::alpha::build_vp8x_payload(width, height, true);

        let vp8x_chunk_len = 8 + 10; // 'VP8X' + 10 bytes payload (even, no pad)
        let alph_pad = alph.len() % 2;
        let alph_chunk_len = 8 + alph.len() + alph_pad;
        let vp8_pad = vp8_payload.len() % 2;
        let vp8_chunk_len = 8 + vp8_payload.len() + vp8_pad;

        let riff_payload_len = 4 + vp8x_chunk_len + alph_chunk_len + vp8_chunk_len;
        let total_file_size = 8 + riff_payload_len;

        let mut out = Vec::with_capacity(total_file_size);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(riff_payload_len as u32).to_le_bytes());
        out.extend_from_slice(b"WEBP");

        // VP8X Chunk
        write_chunk_header(&mut out, b"VP8X", 10);
        out.extend_from_slice(&vp8x_data);

        // ALPH Chunk
        write_chunk_header(&mut out, b"ALPH", alph.len() as u32);
        out.extend_from_slice(alph);
        if alph_pad > 0 {
            out.push(0);
        }

        // VP8 Chunk
        write_chunk_header(&mut out, b"VP8 ", vp8_payload.len() as u32);
        out.extend_from_slice(vp8_payload);
        if vp8_pad > 0 {
            out.push(0);
        }

        Bytes::from(out)
    }
}
