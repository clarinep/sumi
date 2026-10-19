use std::{env, path::Path, sync::Arc, time::Instant};

use fast_ssim2::{PixelDescriptor, PixelSlice, compute_ssimulacra2};
use rgb::RGB8;
use sumi::renderer::{CardCache, PrintNumber, RawCardImage, create_drop_raw, init_font};
use webpx::{AlphaFilter, EncoderConfig, ImageHint, Preset, Unstoppable};

const QUALITIES: [f32; 8] = [65.0, 70.0, 75.0, 78.0, 80.0, 82.0, 85.0, 90.0];
const DISCORD_BG: RGB8 = RGB8 { r: 49, g: 51, b: 56 }; // #313338 dark theme

// Target cards fixed specifically to IMG_8376 and IMG_8377 for deterministic, reproducible results
const DEFAULT_CARD_LEFT: &str = "IMG_8376";
const DEFAULT_CARD_RIGHT: &str = "IMG_8377";
const TEST_PRINT_NUMBER: u32 = 1;

// Encode webp with identical zero-overhead production settings as sumi runtime
fn encode_webp(w: u32, h: u32, rgba: &[u8], quality: f32) -> Vec<u8> {
    EncoderConfig::new()
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
        .segments(1)
        .encode_rgba(rgba, w, h, Unstoppable)
        .expect("webp encode failed")
}

// Blend alpha onto Discord background (#313338) so invisible transparent pixels
// do not skew or penalize psychovisual perceptual metrics
#[inline]
fn blend_bg(c: &[u8]) -> RGB8 {
    match c[3] {
        0 => DISCORD_BG,
        255 => RGB8::new(c[0], c[1], c[2]),
        a => {
            let alpha = u32::from(a);
            let inv_a = 255 - alpha;
            let blend = |fg: u8, bg: u8| {
                ((u32::from(fg) * alpha + u32::from(bg) * inv_a + 127) / 255) as u8
            };
            RGB8::new(
                blend(c[0], DISCORD_BG.r),
                blend(c[1], DISCORD_BG.g),
                blend(c[2], DISCORD_BG.b),
            )
        }
    }
}

// Convert RGBA buffer to contiguous RGB8 with transparent alpha blended onto Discord bg
fn blend_to_rgb(rgba: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity((rgba.len() / 4) * 3);
    for chunk in rgba.chunks_exact(4) {
        let c = blend_bg(chunk);
        rgb.push(c.r);
        rgb.push(c.g);
        rgb.push(c.b);
    }
    rgb
}

// Compute SIMD-accelerated SSIMULACRA 2 score (higher = better, 100 = lossless)
fn score_ssimulacra2(w: u32, h: u32, orig_rgba: &[u8], dist_rgba: &[u8]) -> f64 {
    let orig_rgb = blend_to_rgb(orig_rgba);
    let dist_rgb = blend_to_rgb(dist_rgba);
    let stride = (w * 3) as usize;

    let orig_slice =
        PixelSlice::new(&orig_rgb, w, h, stride, PixelDescriptor::RGB8_SRGB).expect("valid source slice");
    let dist_slice =
        PixelSlice::new(&dist_rgb, w, h, stride, PixelDescriptor::RGB8_SRGB).expect("valid distorted slice");

    compute_ssimulacra2(&orig_slice, &dist_slice).unwrap_or(0.0)
}

async fn fetch_target_card(cache: &CardCache, raw_target: &str) -> Arc<RawCardImage> {
    let clean = raw_target
        .strip_suffix(".webp")
        .or_else(|| raw_target.strip_suffix(".WEBP"))
        .unwrap_or(raw_target);

    if let Ok(card) = cache.get(clean).await {
        return card;
    }

    for name in cache.card_names() {
        if name.as_ref() == clean
            || name.ends_with(&format!("/{}", clean))
            || name.eq_ignore_ascii_case(clean)
        {
            if let Ok(card) = cache.get(name).await {
                return card;
            }
        }
    }

    panic!("could not find card '{raw_target}' in card cache");
}

struct Stat {
    q: f32,
    score: f64,
    kb: f64,
    ms: f64,
}

#[tokio::main]
async fn main() {
    let cards_dir = env::var("CARDS_DIR").unwrap_or_else(|_| "assets".to_string());
    let card_left_name =
        env::var("CARD_LEFT").unwrap_or_else(|_| DEFAULT_CARD_LEFT.to_string());
    let card_right_name =
        env::var("CARD_RIGHT").unwrap_or_else(|_| DEFAULT_CARD_RIGHT.to_string());

    println!("═════════════════════════════════════════════════════════════════════════════════");
    println!("  sumi SSIMULACRA 2 Quality Calibration & Drop Fidelity Test");
    println!("═════════════════════════════════════════════════════════════════════════════════");
    println!("  assets dir : {cards_dir}");
    println!("  target pair: {card_left_name}.webp + {card_right_name}.webp");
    println!("  print num  : #{TEST_PRINT_NUMBER}");
    println!("  qualities  : {:?}\n", QUALITIES);

    init_font();

    let cache = match CardCache::new(Path::new(&cards_dir)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("failed to init card cache: {e}\npls set CARDS_DIR");
            std::process::exit(1);
        }
    };

    // Load both specific cards requested
    let left_card = fetch_target_card(&cache, &card_left_name).await;
    let right_card = fetch_target_card(&cache, &card_right_name).await;

    // ─────────────────────────────────────────────────────────────────────────────
    // 1. FULL DROP COMPOSITION TEST (#1 / #1)
    // ─────────────────────────────────────────────────────────────────────────────
    println!("─────────────────────────────────────────────────────────────────────────────────");
    println!("  1. FULL DROP COMPOSITE: {card_left_name} (#1) + {card_right_name} (#1)");
    println!("─────────────────────────────────────────────────────────────────────────────────");

    let print_left = PrintNumber::new(TEST_PRINT_NUMBER);
    let print_right = PrintNumber::new(TEST_PRINT_NUMBER);

    // Create the uncompressed reference canvas buffer exactly as Sumi generates
    let (drop_raw_buffer, drop_w, drop_h) =
        create_drop_raw(&left_card, &right_card, print_left, print_right);

    let mut drop_stats = Vec::with_capacity(QUALITIES.len());

    for &q in &QUALITIES {
        let start = Instant::now();
        let webp_bytes = encode_webp(drop_w, drop_h, &drop_raw_buffer, q);
        let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
        let kb = webp_bytes.len() as f64 / 1024.0;

        let (dist_pixels, _, _) =
            webpx::decode_rgba(&webp_bytes).expect("failed to decode compressed drop webp");
        let ssim2 = score_ssimulacra2(drop_w, drop_h, &drop_raw_buffer, &dist_pixels);

        drop_stats.push(Stat {
            q,
            score: ssim2,
            kb,
            ms: encode_ms,
        });
    }

    let baseline_kb = drop_stats
        .iter()
        .find(|s| (s.q - 85.0).abs() < f32::EPSILON)
        .map_or(147.5, |s| s.kb);

    println!("  quality     ssim2 score      drop size     savings       encode     rating");
    println!("  {}", "─".repeat(77));

    for s in &drop_stats {
        let savings = (1.0 - (s.kb / baseline_kb)) * 100.0;
        let rating = match s.score {
            score if score >= 90.0 => "visually lossless",
            score if score >= 85.0 => "elysium",
            score if score >= 78.0 => "great",
            score if score >= 70.0 => "sweet spot",
            score if score >= 55.0 => "moderate loss",
            _ => "heavy loss",
        };

        println!(
            "  Q {:<5.1}   {:>10.3}     {:>8.1} KB    {:>+6.1}%    {:>7.2} ms    {rating}",
            s.q, s.score, s.kb, savings, s.ms
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // 2. INDIVIDUAL CARD FIDELITY TESTS
    // ─────────────────────────────────────────────────────────────────────────────
    println!("\n─────────────────────────────────────────────────────────────────────────────────");
    println!("  2. INDIVIDUAL CARD FIDELITY: {card_left_name}.webp & {card_right_name}.webp");
    println!("─────────────────────────────────────────────────────────────────────────────────");

    for (name, card) in [(&card_left_name, &left_card), (&card_right_name, &right_card)] {
        let (w, h) = (card.size.width, card.size.height);
        let orig = &card.pixels;

        println!("  Card: {name} ({}x{})", w, h);
        println!("  quality     ssim2 score      card size     savings       encode     rating");
        println!("  {}", "─".repeat(77));

        let card_baseline_webp = encode_webp(w, h, orig, 85.0);
        let card_baseline_kb = card_baseline_webp.len() as f64 / 1024.0;

        for &q in &QUALITIES {
            let start = Instant::now();
            let webp_bytes = encode_webp(w, h, orig, q);
            let encode_ms = start.elapsed().as_secs_f64() * 1000.0;
            let kb = webp_bytes.len() as f64 / 1024.0;

            let (dist_pixels, _, _) =
                webpx::decode_rgba(&webp_bytes).expect("failed to decode card webp");
            let ssim2 = score_ssimulacra2(w, h, orig, &dist_pixels);

            let savings = (1.0 - (kb / card_baseline_kb)) * 100.0;
            let rating = match ssim2 {
                score if score >= 90.0 => "visually lossless",
                score if score >= 85.0 => "elysium",
                score if score >= 78.0 => "great",
                score if score >= 70.0 => "sweet spot",
                score if score >= 55.0 => "moderate loss",
                _ => "heavy loss",
            };

            println!(
                "  Q {:<5.1}   {:>10.3}     {:>8.1} KB    {:>+6.1}%    {:>7.2} ms    {rating}",
                q, ssim2, kb, savings, encode_ms
            );
        }
        println!();
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // 3. RECOMMENDATION
    // ─────────────────────────────────────────────────────────────────────────────
    if let Some(best) = drop_stats.iter().find(|s| s.score >= 80.0) {
        let savings = (1.0 - (best.kb / baseline_kb)) * 100.0;
        println!("═════════════════════════════════════════════════════════════════════════════════");
        println!(
            "  RECOMMENDED SWEET SPOT: Q {:.1} (Score: {:.2}, Drop Size: {:.1} KB, Savings: {:+.1}%)",
            best.q, best.score, best.kb, savings
        );
        println!("═════════════════════════════════════════════════════════════════════════════════");
    }
}
