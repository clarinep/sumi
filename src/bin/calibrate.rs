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

// Quality steps to sweep across during calibration
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

// Discord Dark Theme background (#313338 -> 49, 51, 56)
// Blending transparent pixels over Discord background ensures Butteraugli
// evaluates what human eyes actually see, rather than comparing invisible alpha=0 pixels.
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
    println!("═════════════════════════════════════════════════════════════════════════════════");
    println!("  Sumi Offline WebP Quality Calibrator (Butteraugli Psychovisual Metric)        ");
    println!("═════════════════════════════════════════════════════════════════════════════════\n");

    let cards_dir = env::var("CARDS_DIR").unwrap_or_else(|_| "assets".to_string());
    println!("📂 Loading card library from '{}'...", cards_dir);

    let cache = match CardCache::new(Path::new(&cards_dir)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("❌ Failed to initialize card cache: {e}");
            eprintln!("   Ensure CARDS_DIR is set or run from the sumi project root with assets/ present.");
            std::process::exit(1);
        }
    };

    let card_names = cache.card_names();
    if card_names.is_empty() {
        eprintln!("❌ No cards found in '{cards_dir}'. Cannot run calibration.");
        std::process::exit(1);
    }

    let test_count = card_names.len().min(15);
    println!("🎴 Selected {} sample card(s) for psychovisual distortion testing.", test_count);
    println!("🔬 Testing {} quality levels per card...\n", QUALITY_STEPS.len());

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

            // 1. Time WebP compression at this quality
            let start = Instant::now();
            let webp_data = encode_at_quality(width, height, original_pixels, quality);
            total_time_ms += start.elapsed().as_secs_f64() * 1000.0;
            total_size_bytes += webp_data.len();

            // 2. Decode back to uncompressed RGBA for comparison
            let (decoded_pixels, _, _) = webpx::decode_rgba(&webp_data).expect("Decode failed");

            // 3. Compute Butteraugli perceptual difference
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
            "✓ Perceptual Lossless"
        } else if r.avg_score <= 1.5 {
            "★ Discord Sweet Spot"
        } else if r.avg_score <= 2.5 {
            "⚡ Moderate Degradation"
        } else {
            "⚠ Noticeable Artifacts"
        };

        println!(
            "│ Q {:<5.1} │ {:<12.3} │ {:<12.3} │ {:>6.1} KB  │ {:>+6.1}%     │ {:>7.2} ms │ {:<24} │",
            r.quality, r.avg_score, r.max_score, r.avg_size_kb, savings, r.avg_time_ms, verdict
        );
    }

    println!("└─────────┴──────────────┴──────────────┴────────────┴─────────────┴─────────────┴──────────────────────────┘\n");
    println!("💡 Evaluation Guide:");
    println!("   • Score < 1.0 : Indistinguishable from lossless to human vision.");
    println!("   • Score 1.0 - 1.5 : Pristine on chat/mobile screens; saves 20-35% network bandwidth.");
    println!("   • Score > 2.0 : Visual degradation / ringing appears on sharp lines.\n");
    println!("👉 Pick the quality level with the best savings while keeping Avg Score ≤ 1.2,");
    println!("   then update `const WEBP_QUALITY: f32` in `src/renderer/encoder.rs`.\n");
}
