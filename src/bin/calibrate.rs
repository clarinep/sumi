use std::{
    env,
    path::Path,
    time::Instant,
};

use butteraugli::{ButteraugliParams, butteraugli};
use imgref::ImgRef;
use rgb::RGB8;
use sumi::renderer::CardCache;
use webpx::{AlphaFilter, EncoderConfig, ImageHint, Preset, Unstoppable};

const QUALITY_STEPS: &[f32] = &[65.0, 70.0, 75.0, 78.0, 80.0, 82.0, 85.0, 90.0];

fn encode_at_quality(width: u32, height: u32, rgba: &[u8], quality: f32) -> Vec<u8> {
    let settings = EncoderConfig::new()
        .preset(Preset::Picture)
        .hint(ImageHint::Picture)
        .quality(quality)
        .method(0)
        .alpha_compression(true)
        .alpha_quality(85)
        .alpha_filter(AlphaFilter::None)
        .low_memory(false)
        .pass(1)
        .exact(false)
        .segments(1);

    settings
        .encode_rgba(rgba, width, height, Unstoppable)
        .expect("WebP encode failed")
}

// blend alpha pix ke discord bg color biar butteraugli gak binggung
#[inline]
fn composite_over_discord_bg(c: &[u8]) -> RGB8 {
    let a = c[3] as u32;
    if a == 0 {
        return RGB8::new(49, 51, 56);
    }
    if a == 255 {
        return RGB8::new(c[0], c[1], c[2]);
    }
    let inv_a = 255 - a;
    let r = ((c[0] as u32 * a + 49 * inv_a + 127) / 255) as u8;
    let g = ((c[1] as u32 * a + 51 * inv_a + 127) / 255) as u8;
    let b = ((c[2] as u32 * a + 56 * inv_a + 127) / 255) as u8;
    RGB8::new(r, g, b)
}

fn compute_butteraugli_score(
    width: usize,
    height: usize,
    original_rgba: &[u8],
    distorted_rgba: &[u8],
) -> f64 {
    let orig_rgb: Vec<RGB8> = original_rgba
        .chunks_exact(4)
        .map(composite_over_discord_bg)
        .collect();

    let dist_rgb: Vec<RGB8> = distorted_rgba
        .chunks_exact(4)
        .map(composite_over_discord_bg)
        .collect();

    let img1 = ImgRef::new(&orig_rgb, width, height);
    let img2 = ImgRef::new(&dist_rgb, width, height);
    let params = ButteraugliParams::default();

    match butteraugli(img1, img2, &params) {
        Ok(res) => res.score,
        Err(e) => {
            eprintln!("butteraugli comparison error: {e:?}");
            99.0
        }
    }
}

struct QualityResult {
    quality: f32,
    avg_score: f64,
    max_score: f64,
    avg_size_kb: f64,
    avg_time_ms: f64,
}

#[tokio::main]
async fn main() {
    let cards_dir = env::var("CARDS_DIR").unwrap_or_else(|_| "assets".to_string());
    println!("loading cards from '{}'..", cards_dir);

    let cache = match CardCache::new(Path::new(&cards_dir)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to start card cache: {e}");
            eprintln!("pls set CARDS_DIR");
            std::process::exit(1);
        }
    };

    let card_names = cache.card_names();
    if card_names.is_empty() {
        eprintln!("? no cards found in '{cards_dir}'");
        std::process::exit(1);
    }

    let test_count = card_names.len().min(15);
    println!("selected {} samples for test..", test_count);
    println!("!! testing {} quality levels per card..\n", QUALITY_STEPS.len());

    let mut results = Vec::with_capacity(QUALITY_STEPS.len());

    for &quality in QUALITY_STEPS {
        let mut total_score = 0.0;
        let mut max_score = 0.0f64;
        let mut total_size_bytes = 0;
        let mut total_time_ms = 0.0;

        for name in &card_names[..test_count] {
            let card = cache.get(name).await.expect("Failed to load card image");
            let width = card.size.width;
            let height = card.size.height;
            let original_pixels = &card.pixels;

            let start = Instant::now();
            let webp_data = encode_at_quality(width, height, original_pixels, quality);
            total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            total_size_bytes += webp_data.len();

            let (decoded_pixels, _, _) = webpx::decode_rgba(&webp_data).expect("Decode failed");

            let score = compute_butteraugli_score(
                width as usize,
                height as usize,
                original_pixels,
                &decoded_pixels,
            );

            total_score += score;
            if score > max_score {
                max_score = score;
            }
        }

        let n = test_count as f64;
        results.push(QualityResult {
            quality,
            avg_score: total_score / n,
            max_score,
            avg_size_kb: (total_size_bytes as f64 / n) / 1024.0,
            avg_time_ms: total_time_ms / n,
        });
    }

    let baseline_size = results
        .iter()
        .find(|r| (r.quality - 85.0).abs() < f32::EPSILON)
        .map(|r| r.avg_size_kb)
        .unwrap_or(147.5);

    println!("┌─────────┬──────────────┬──────────────┬────────────┬─────────────┬─────────────┬──────────────────────────┐");
    println!("│ Quality │ Avg Score    │ Max Score    │ Avg Size   │ Savings %   │ Encode Time │ Rating                   │");
    println!("├─────────┼──────────────┼──────────────┼────────────┼─────────────┼─────────────┼──────────────────────────┤");

    for r in &results {
        let savings = (1.0 - (r.avg_size_kb / baseline_size)) * 100.0;

        let verdict = if r.avg_score < 1.0 {
            "Fully lossless"
        } else if r.avg_score <= 1.5 {
            "Sweet spot"
        } else if r.avg_score <= 2.5 {
            "Moderate trashy"
        } else {
            "Extreme trashy artifacts"
        };

        println!(
            "│ Q {:<5.1} │ {:<12.3} │ {:<12.3} │ {:>6.1} KB  │ {:>+6.1}%     │ {:>7.2} ms │ {:<24} │",
            r.quality, r.avg_score, r.max_score, r.avg_size_kb, savings, r.avg_time_ms, verdict
        );
    }
}
