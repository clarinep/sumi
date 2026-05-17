use std::{
    fmt::{Debug, Formatter, Result as FmtResult},
    sync::LazyLock,
    time::Instant,
};

use bytes::Bytes;
use fontdue::{Font, FontSettings};
use itoa::Buffer as ItoaBuffer;

use crate::renderer::{encoding::encode_webp, error::RenderError};

/// raw uncompressed rgba image
/// this will act as a container that will replace our previous `image::RgbaImage`.
pub struct RawCardImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
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

const TEXT_SIZE: f32 = 60.0;
const TEXT_PADDING_FROM_EDGE: i32 = 190;
const PADDING_BETWEEN_CARDS: u32 = 20;
const TEXT_PADDING_FROM_BOTTOM: i32 = 80;

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
        let metric_width = u32::try_from(metrics.width).unwrap_or(0);
        let metric_height = u32::try_from(metrics.height).unwrap_or(0);
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
#[allow(clippy::many_single_char_names)]
fn draw_text(canvas: &mut RawCardImage, text: &[u8], mut x: i32, y: i32) {
    let canvas_width = i32::try_from(canvas.width).unwrap_or(0);
    let canvas_height = i32::try_from(canvas.height).unwrap_or(0);
    let canvas_buf = &mut canvas.pixels;

    for &b in text {
        // look up the letter
        // we only support 1-9 and #
        let letter = match b {
            b'#' => &LETTERS.hash,
            b'0'..=b'9' => &LETTERS.digits[(b - b'0') as usize],
            _ => continue,
        };

        // pre cast letter w h to i32 to avoid repeated casting
        let letter_width = i32::try_from(letter.width).unwrap_or(0);
        let letter_height = i32::try_from(letter.height).unwrap_or(0);

        // count the starting x and y coords for letter on the canvas
        let draw_y = y + letter.offset_y;

        for draw_y_offset in 0..letter_height {
            let canvas_y = draw_y + draw_y_offset;

            // skip row if its outside canvas
            if canvas_y < 0 || canvas_y >= canvas_height {
                continue;
            }

            // count and make sure we dont draw outside canvas
            let draw_x_start = if x + letter.offset_x < 0 { -(x + letter.offset_x) } else { 0 };
            let draw_x_end = if x + letter.offset_x + letter_width > canvas_width {
                canvas_width - (x + letter.offset_x)
            } else {
                letter_width
            };

            // skip if the letter horizontally is outside canvas
            if draw_x_start >= draw_x_end {
                continue;
            }

            // canvas is rgba so its 4 bytes per pixel, coverage is 1 byte per pixel.
            let canvas_pixel_start =
                usize::try_from(canvas_y * canvas_width + (x + letter.offset_x + draw_x_start))
                    .unwrap_or(0)
                    * 4;
            let letter_pixel_start =
                usize::try_from(draw_y_offset * letter_width + draw_x_start).unwrap_or(0);

            let count = usize::try_from(draw_x_end - draw_x_start).unwrap_or(0);
            let target_pixels = &mut canvas_buf[canvas_pixel_start..canvas_pixel_start + count * 4];
            let glyph_row = &letter.coverage[letter_pixel_start..letter_pixel_start + count];

            for (pixel, &coverage) in target_pixels.chunks_exact_mut(4).zip(glyph_row) {
                if coverage == 255 {
                    pixel[0] = 255;
                    pixel[1] = 255;
                    pixel[2] = 255;
                    pixel[3] = 255;
                } else if coverage > 0 {
                    let alpha = u32::from(coverage);
                    let inv_alpha = 255 - alpha;

                    let r = u32::from(pixel[0]);
                    let g = u32::from(pixel[1]);
                    let b = u32::from(pixel[2]);
                    let a = u32::from(pixel[3]);

                    // divide by 256 using bitshift.
                    // this will make it 254 instead of 255 but is mathematically much faster.
                    // -- Fixing next version
                    pixel[0] = (alpha + ((r * inv_alpha) >> 8)) as u8;
                    pixel[1] = (alpha + ((g * inv_alpha) >> 8)) as u8;
                    pixel[2] = (alpha + ((b * inv_alpha) >> 8)) as u8;
                    pixel[3] = (alpha + ((a * inv_alpha) >> 8)) as u8;
                }
            }
        }

        // up the x coord for the next letter.
        x += letter.advance_width;
    }
}

fn copy_card_pixels(
    buffer: &mut [u8],
    card: &RawCardImage,
    total_width: u32,
    start_x: u32,
    start_y: u32,
) {
    let card_width = card.width as usize;
    let card_height = card.height as usize;
    let card_row_bytes = card_width * 4;
    let total_row_bytes = (total_width * 4) as usize;

    let mut start_index = ((start_y * total_width + start_x) * 4) as usize;

    for row in 0..card_height {
        let card_start = row * card_row_bytes;
        buffer[start_index..start_index + card_row_bytes]
            .copy_from_slice(&card.pixels[card_start..card_start + card_row_bytes]);
        start_index += total_row_bytes;
    }
}

fn format_print_num(print_num: u32, buf: &mut [u8; 16]) -> &[u8] {
    let mut itoa_buf = ItoaBuffer::new();
    let num_str = itoa_buf.format(print_num);
    buf[0] = b'#';
    buf[1..=num_str.len()].copy_from_slice(num_str.as_bytes());
    &buf[..=num_str.len()]
}

/// combine two card images and add print numbers = drop image
/// we manually copy pixel rows from the card images. this is much faster
/// than creating a new blank image and using a library to paste the card images to it.
pub fn create_drop_image(
    left_card: &RawCardImage,
    right_card: &RawCardImage,
    left_card_print: u32,
    right_card_print: u32,
) -> Result<Bytes, RenderError> {
    let start_canvas = Instant::now();

    // count the dimensions of our drop image
    let left_width = left_card.width;
    let right_width = right_card.width;

    let total_width = left_width + right_width + PADDING_BETWEEN_CARDS * 3;
    let max_card_height = left_card.height.max(right_card.height);
    let total_height = max_card_height + PADDING_BETWEEN_CARDS * 2;

    // make sure buffer big enough for image (width * height * 4 bytes per pixel).
    let required_len = (total_width * total_height * 4) as usize;

    // We can just use the standard vec! macro.
    // This removes need for unsafe block here and is optimized by compiler.
    let mut buffer: Vec<u8> = vec![0; required_len];

    // count starting position for the left and right card.
    let left_card_x = PADDING_BETWEEN_CARDS;
    let right_card_x = left_width + PADDING_BETWEEN_CARDS * 2;
    let card_y = PADDING_BETWEEN_CARDS;

    // copy pixels from left and right card into buffer.
    copy_card_pixels(&mut buffer, left_card, total_width, left_card_x, card_y);
    copy_card_pixels(&mut buffer, right_card, total_width, right_card_x, card_y);

    // wrap the buffer into RawCardImage so we can pass it to the encoder etc
    let mut final_image = RawCardImage { width: total_width, height: total_height, pixels: buffer };

    // format print numbers into string
    let mut left_text_buf = [0u8; 16];
    let left_text = format_print_num(left_card_print, &mut left_text_buf);

    let mut right_text_buf = [0u8; 16];
    let right_text = format_print_num(right_card_print, &mut right_text_buf);

    let canvas_time = start_canvas.elapsed();

    let start_text = Instant::now();
    // count positions for text and draw it to the image
    let left_text_x = i32::try_from(left_card_x + left_width).unwrap_or(0) - TEXT_PADDING_FROM_EDGE;
    let right_text_x =
        i32::try_from(right_card_x + right_width).unwrap_or(0) - TEXT_PADDING_FROM_EDGE;
    let text_y =
        i32::try_from(total_height).unwrap_or(0) - TEXT_SIZE as i32 - TEXT_PADDING_FROM_BOTTOM;

    draw_text(&mut final_image, left_text, left_text_x, text_y);
    draw_text(&mut final_image, right_text, right_text_x, text_y);

    let text_time = start_text.elapsed();

    let start_encode = Instant::now();
    // encode final drop image to webp
    let result = encode_webp(&final_image);
    let encode_time = start_encode.elapsed();

    log::debug!(
        "pasting={:.3}ms, font={:.3}ms, encode={:.3}ms",
        canvas_time.as_secs_f64() * 1000.0,
        text_time.as_secs_f64() * 1000.0,
        encode_time.as_secs_f64() * 1000.0
    );

    result
}
