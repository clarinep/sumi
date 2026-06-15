use bytes::Bytes;
use webpx::{AlphaFilter, EncoderConfig, Preset, Unstoppable};

use crate::renderer::error::RenderError;

const WEBP_QUALITY: f32 = 85.0;
const WEBP_SPEED: u8 = 0;
const WEBP_ALPHA_QUALITY: u8 = 80;
const WEBP_THREAD_LEVEL: u8 = 0;
const WEBP_SEGMENTS: u8 = 1;

/// we take raw pixels of two diff cards from atlas (memory)
/// paste it into canvas and draw print nums
/// but because of that huge amount of alpha pixels and bigger image dimension
/// now if we send that pixels data over to discord, lets just say the file size would be 2 MB
/// so we would suffer two things, our side and the users side of having to load that shit
/// this is why we do reencoding back to a "new" compressed .webp giving us ~500 KB final file
/// the point of using jpeg would be pointless as we need alpha support
/// the point of skipping alpha compression in webpx is also pointless - similar reason
/// using lossless is also not worth, although encoding takes 100ms instead of our current 350ms
/// it is uncompressed and our drop image dimension is huge so again file size would be 2 MB - bad.
pub fn encode_webp(width: u32, height: u32, pixel_data: &[u8]) -> Result<Bytes, RenderError> {

    // we keep encoding on one thread so we avoid slowing down the server.
    // this is by far the only overhead we have, it'll take ~100ms per img.
    // see all other options at https://crates.io/crates/webpx
    let settings = EncoderConfig::new()
        // preset needs more testing to check which one is best for our cards
        .preset(Preset::Picture)
        .hint(webpx::ImageHint::Picture)
        .quality(WEBP_QUALITY)
        .method(WEBP_SPEED)
        .thread_level(WEBP_THREAD_LEVEL)
        .alpha_compression(true)
        .alpha_quality(WEBP_ALPHA_QUALITY)
        .alpha_filter(AlphaFilter::None) // none filter maximizes alpha encoding throughput
        .low_memory(false)
        .pass(1)
        .exact(false)
        .partitions(3) // 3 = 8 partitions, unlocks parallel decoding for clients and parallelizes entropy stage
        .segments(WEBP_SEGMENTS);

    let webp_data: Vec<u8> = settings
        .encode_rgba(pixel_data, width, height, Unstoppable)
        .map_err(|error| RenderError::EncodeError(error.to_string()))?;

    Ok(Bytes::from(webp_data))
}
