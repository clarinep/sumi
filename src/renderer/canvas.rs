use std::time::Instant;

use bytes::Bytes;

use super::{
    encoding::encode_webp,
    error::RenderError,
    pixels::RawCardImage,
    print::{draw_text, measure_text},
};

const TEXT_SIZE: f32 = 60.0;
const TEXT_PADDING_FROM_EDGE: i32 = 190;
const PADDING_BETWEEN_CARDS: u32 = 20;
const TEXT_PADDING_FROM_BOTTOM: i32 = 80;

#[inline]
fn copy_card_pixels(
    buffer: &mut [u8],
    card: &RawCardImage,
    total_width: u32,
    start_x: u32,
    start_y: u32,
) {
    let card_row_bytes = (card.width * 4) as usize;
    let total_row_bytes = (total_width * 4) as usize;
    let start_index = ((start_y * total_width + start_x) * 4) as usize;

    let dest_rows = buffer[start_index..].chunks_exact_mut(total_row_bytes);
    let src_rows = card.pixels.chunks_exact(card_row_bytes);

    for (dest_row, src_row) in dest_rows.zip(src_rows).take(card.height as usize) {
        dest_row[..card_row_bytes].copy_from_slice(src_row);
    }
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
    let mut final_image = RawCardImage {
        width: total_width,
        height: total_height,
        pixels: buffer.into_boxed_slice(),
    };

    let left_text_str = format!("#{left_card_print}");
    let left_text = left_text_str.as_bytes();

    let right_text_str = format!("#{right_card_print}");
    let right_text = right_text_str.as_bytes();

    let canvas_time = start_canvas.elapsed();
    let start_text = Instant::now();

    // count positions for text and draw it to the image
    let left_text_width = measure_text(left_text);
    let right_text_width = measure_text(right_text);

    let ref_width = measure_text(b"#00");
    let right_padding = TEXT_PADDING_FROM_EDGE - ref_width;

    let left_text_x = (left_card_x + left_width).cast_signed() - right_padding - left_text_width;
    let right_text_x =
        (right_card_x + right_width).cast_signed() - right_padding - right_text_width;
    let text_y = total_height.cast_signed() - TEXT_SIZE as i32 - TEXT_PADDING_FROM_BOTTOM;

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
