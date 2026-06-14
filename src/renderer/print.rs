use std::sync::LazyLock;

use fontdue::{Font, FontSettings};

use super::pixels::{Point, RawCardImage};

const TEXT_SIZE: f32 = 60.0;

struct Letter {
    coverage: Vec<u8>,
    width: u32,
    height: u32,
    advance_width: i32,
    offset_x: i32,
    offset_y: i32,
}

struct LetterSet {
    hash: Letter,
    digits: [Letter; 10],
}

static LETTERS: LazyLock<LetterSet> = LazyLock::new(|| {
    let font_data = include_bytes!("../../assets/LexendDeca-Bold.ttf") as &[u8];
    let font =
        Font::from_bytes(font_data, FontSettings::default()).expect("could not load font file");
    let metrics = font.horizontal_line_metrics(TEXT_SIZE).unwrap();
    let ascent = metrics.ascent;

    let render_char = |c: char| -> Letter {
        let (metrics, coverage) = font.rasterize(c, TEXT_SIZE);
        let metric_width = metrics.width as u32;
        let metric_height = metrics.height as u32;
        Letter {
            coverage,
            width: metric_width,
            height: metric_height,
            advance_width: metrics.advance_width.round() as i32,
            offset_x: metrics.xmin,
            offset_y: (ascent - metrics.ymin as f32 - metrics.height as f32).round() as i32,
        }
    };

    LetterSet {
        hash: render_char('#'),
        digits: std::array::from_fn(|i| render_char((b'0' + i as u8) as char)),
    }
});

pub fn init_font() {
    LazyLock::force(&LETTERS);
}

#[inline]
#[allow(clippy::many_single_char_names)]
pub fn draw_print_number(canvas: &mut RawCardImage, print_number: &[u8], mut pos: Point<i32>) {
    let canvas_width = canvas.size.width.cast_signed();
    let canvas_height = canvas.size.height.cast_signed();
    let canvas_buf = &mut canvas.pixels;

    for &b in print_number {
        let letter = match b {
            b'#' => &LETTERS.hash,
            b'0'..=b'9' => &LETTERS.digits[(b - b'0') as usize],
            _ => continue,
        };

        let letter_width = letter.width.cast_signed();
        let letter_height = letter.height.cast_signed();

        let draw_y = pos.y + letter.offset_y;

        for draw_y_offset in 0..letter_height {
            let canvas_y = draw_y + draw_y_offset;

            if canvas_y < 0 || canvas_y >= canvas_height {
                continue;
            }

            let draw_x_start = 0.max(-(pos.x + letter.offset_x));
            let draw_x_end = letter_width.min(canvas_width - (pos.x + letter.offset_x));

            if draw_x_start >= draw_x_end {
                continue;
            }

            let canvas_y_u = usize::try_from(canvas_y).unwrap();
            let canvas_width_u = usize::try_from(canvas_width).unwrap();
            let canvas_x_u = usize::try_from(pos.x + letter.offset_x + draw_x_start).unwrap();
            let canvas_pixel_start = (canvas_y_u * canvas_width_u + canvas_x_u) * 4;

            let draw_y_offset_u = usize::try_from(draw_y_offset).unwrap();
            let letter_width_u = usize::try_from(letter_width).unwrap();
            let draw_x_start_u = usize::try_from(draw_x_start).unwrap();
            let letter_pixel_start = draw_y_offset_u * letter_width_u + draw_x_start_u;

            let count = usize::try_from(draw_x_end - draw_x_start).unwrap();

            let target_pixels = &mut canvas_buf[canvas_pixel_start..canvas_pixel_start + count * 4];
            let glyph_row = &letter.coverage[letter_pixel_start..letter_pixel_start + count];

            for (pixel, &coverage) in target_pixels.chunks_exact_mut(4).zip(glyph_row) {
                if coverage == 255 {
                    pixel.copy_from_slice(&[255, 255, 255, 255]);
                } else if coverage > 0 {
                    let fg_a = u32::from(coverage);
                    let inv_fg_a = 255 - fg_a;
                    let bg_a = u32::from(pixel[3]);

                    let out_a_times_255 = fg_a * 255 + bg_a * inv_fg_a;
                    let out_a = out_a_times_255 / 255;

                    if out_a == 0 {
                        continue;
                    }

                    let r = u32::from(pixel[0]);
                    let g = u32::from(pixel[1]);
                    let b = u32::from(pixel[2]);

                    pixel[0] = ((255 * fg_a * 255 + r * bg_a * inv_fg_a) / out_a_times_255) as u8;
                    pixel[1] = ((255 * fg_a * 255 + g * bg_a * inv_fg_a) / out_a_times_255) as u8;
                    pixel[2] = ((255 * fg_a * 255 + b * bg_a * inv_fg_a) / out_a_times_255) as u8;
                    pixel[3] = out_a as u8;
                }
            }
        }

        pos.x += letter.advance_width;
    }
}

pub fn measure_print_number(print_number: &[u8]) -> i32 {
    let mut width = 0;
    for &b in print_number {
        let letter = match b {
            b'#' => &LETTERS.hash,
            b'0'..=b'9' => &LETTERS.digits[(b - b'0') as usize],
            _ => continue,
        };
        width += letter.advance_width;
    }
    width
}
