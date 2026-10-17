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

fn compute_butteraugli_score(
    width: usize,
    height: usize,
    original_rgba: &[u8],
    distorted_rgba: &[u8],
) -> f64 {
    let orig_rgb: Vec<RGB8> = original_rgba
        .chunks_exact(4)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
        .collect();

    let dist_rgb: Vec<RGB8> = distorted_rgba
        .chunks_exact(4)
        .map(|c| RGB8::new(c[0], c[1], c[2]))
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

    println!("┌─────────┬──────────────┬──────────────┬────────────┬─────────────┬─────────────┬──────────────────────────┐");
    println!("│ Quality │ Avg Score    │ Max Score    │ Avg Size   │ Savings %   │ Encode Time │ Rating                   │");
    println!("├─────────┼──────────────┼──────────────┼────────────┼─────────────┼─────────────┼──────────────────────────┤");

    let mut baseline_size = 0.0;

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
        let avg_score = total_score / n;
        let avg_size_kb = (total_size_bytes as f64 / n) / 1024.0;
        let avg_time_ms = total_time_ms / n;

        if (quality - 85.0).abs() < f32::EPSILON {
            baseline_size = avg_size_kb;
        }

        let savings = if baseline_size > 0.0 {
            (1.0 - (avg_size_kb / baseline_size)) * 100.0
        } else {
            0.0
        };

        let verdict = if avg_score < 1.0 {
            "✓ Perceptual Lossless"
        } else if avg_score <= 1.5 {
            "★ Discord Sweet Spot"
        } else if avg_score <= 2.0 {
            "⚡ Moderate Degradation"
        } else {
            "⚠ Noticeable Artifacts"
        };

        println!(
            "│ Q {:<5.1} │ {:<12.3} │ {:<12.3} │ {:>6.1} KB  │ {:>+6.1}%     │ {:>7.2} ms │ {:<24} │",
            quality, avg_score, max_score, avg_size_kb, -savings, avg_time_ms, verdict
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
