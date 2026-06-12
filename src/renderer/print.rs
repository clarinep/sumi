use std::sync::LazyLock;

use fontdue::{Font, FontSettings};

use super::pixels::{Point, RawCardImage};

const TEXT_SIZE: f32 = 60.0;

// normal font lib takes a while to render a letter.
// so now we do this.
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

// -- Load font sekali doang
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
        digits: [
            render_char('0'),
            render_char('1'),
            render_char('2'),
            render_char('3'),
            render_char('4'),
            render_char('5'),
            render_char('6'),
            render_char('7'),
            render_char('8'),
            render_char('9'),
        ],
    }
});

pub fn init_font() {
    let _ = &*LETTERS;
}

/// here we manually do alpha blending of the fonts to the image pixel buffer.
/// we do this manually instead of using a image processing library
/// because it is a bitty faster and avoids useless overhead.
/// we are simply manipulating the byte array for some tiny peformance gain
#[inline]
#[allow(clippy::many_single_char_names)]
pub fn draw_print_number(canvas: &mut RawCardImage, print_number: &[u8], mut pos: Point<i32>) {
    let canvas_width = canvas.size.width.cast_signed();
    let canvas_height = canvas.size.height.cast_signed();
    let canvas_buf = &mut canvas.pixels;

    for b in print_number.iter().copied() {
        // look up the letter
        // we only support 1-9 and #
        let letter = match b {
            b'#' => &LETTERS.hash,
            b'0'..=b'9' => &LETTERS.digits[(b - b'0') as usize],
            _ => continue,
        };

        // pre cast letter w h to i32 to avoid repeated casting
        let letter_width = letter.width.cast_signed();
        let letter_height = letter.height.cast_signed();

        // count the starting x and y coords for letter on the canvas
        let draw_y = pos.y + letter.offset_y;

        for draw_y_offset in 0..letter_height {
            let canvas_y = draw_y + draw_y_offset;

            // skip row if its outside canvas
            if canvas_y < 0 || canvas_y >= canvas_height {
                continue;
            }

            // count and make sure we dont draw outside canvas
            let draw_x_start = 0.max(-(pos.x + letter.offset_x));
            let draw_x_end = letter_width.min(canvas_width - (pos.x + letter.offset_x));

            // skip if the letter horizontally is outside canvas
            if draw_x_start >= draw_x_end {
                continue;
            }

            // canvas is rgba so its 4 bytes per pixel, coverage is 1 byte per pixel.
            let canvas_pixel_start = usize::try_from(
                (canvas_y * canvas_width + (pos.x + letter.offset_x + draw_x_start)) * 4,
            )
            .unwrap();
            let letter_pixel_start =
                usize::try_from(draw_y_offset * letter_width + draw_x_start).unwrap();

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

                    // Unpremultiplied RGBA blending:
                    // out_a = fg_a + bg_a * (255 - fg_a) / 255
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

        // up the x coord for the next letter.
        pos.x += letter.advance_width;
    }
}

pub fn measure_print_number(print_number: &[u8]) -> i32 {
    let mut width = 0;
    for b in print_number.iter().copied() {
        let letter = match b {
            b'#' => &LETTERS.hash,
            b'0'..=b'9' => &LETTERS.digits[(b - b'0') as usize],
            _ => continue,
        };
        width += letter.advance_width;
    }
    width
}
