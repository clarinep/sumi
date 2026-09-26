use std::sync::LazyLock;

use bytes::Bytes;
use crossbeam_queue::ArrayQueue;

use super::{
    PrintNumber,
    encoder::encode_webp,
    error::Result,
    pixels::{Point, RawCardImage},
    print::{TEXT_SIZE, draw_print_number, measure_print_number, ref_number_width},
};

const TEXT_PADDING_FROM_EDGE: i32 = 190;
const PADDING_BETWEEN_CARDS: u32 = 20;
const TEXT_PADDING_FROM_BOTTOM: i32 = 80;

// Combines two card images into destination buffer in a single row-by-row pass.
// Writing both cards to the destination row while it is cached in L1/L2
// avoids sweeping through the 6.5MB buffer twice and eliminates cache misses.
#[inline]
fn composite_cards(
    buffer: &mut [u8],
    left_card: &RawCardImage,
    right_card: &RawCardImage,
    total_width: u32,
    left_x: u32,
    right_x: u32,
    card_y: u32,
) {
    let left_row_bytes = (left_card.size.width * 4) as usize;
    let right_row_bytes = (right_card.size.width * 4) as usize;
    let total_row_bytes = (total_width * 4) as usize;

    let left_x_offset = (left_x * 4) as usize;
    let right_x_offset = (right_x * 4) as usize;

    let common_height = (left_card.size.height.min(right_card.size.height)) as usize;
    let mut dest_offset = (card_y as usize) * total_row_bytes;
    let mut left_src_offset = 0;
    let mut right_src_offset = 0;

    for _ in 0..common_height {
        buffer[dest_offset + left_x_offset..dest_offset + left_x_offset + left_row_bytes]
            .copy_from_slice(&left_card.pixels[left_src_offset..left_src_offset + left_row_bytes]);
        buffer[dest_offset + right_x_offset..dest_offset + right_x_offset + right_row_bytes]
            .copy_from_slice(&right_card.pixels[right_src_offset..right_src_offset + right_row_bytes]);

        dest_offset += total_row_bytes;
        left_src_offset += left_row_bytes;
        right_src_offset += right_row_bytes;
    }

    if left_card.size.height > right_card.size.height {
        let extra = (left_card.size.height - right_card.size.height) as usize;
        for _ in 0..extra {
            buffer[dest_offset + left_x_offset..dest_offset + left_x_offset + left_row_bytes]
                .copy_from_slice(&left_card.pixels[left_src_offset..left_src_offset + left_row_bytes]);
            dest_offset += total_row_bytes;
            left_src_offset += left_row_bytes;
        }
    } else if right_card.size.height > left_card.size.height {
        let extra = (right_card.size.height - left_card.size.height) as usize;
        for _ in 0..extra {
            buffer[dest_offset + right_x_offset..dest_offset + right_x_offset + right_row_bytes]
                .copy_from_slice(&right_card.pixels[right_src_offset..right_src_offset + right_row_bytes]);
            dest_offset += total_row_bytes;
            right_src_offset += right_row_bytes;
        }
    }
}

// Inlined branchless-friendly digit formatting for values 1..=999
#[inline]
fn format_print_number(val: u16, buf: &mut [u8; 8]) -> &[u8] {
    buf[0] = b'#';
    if val < 10 {
        buf[1] = b'0' + val as u8;
        &buf[..2]
    } else if val < 100 {
        buf[1] = b'0' + (val / 10) as u8;
        buf[2] = b'0' + (val % 10) as u8;
        &buf[..3]
    } else {
        buf[1] = b'0' + (val / 100) as u8;
        buf[2] = b'0' + ((val / 10) % 10) as u8;
        buf[3] = b'0' + (val % 10) as u8;
        &buf[..4]
    }
}

static DROP_POOL: LazyLock<ArrayQueue<Vec<u8>>> =
    LazyLock::new(|| ArrayQueue::new(MAX_POOL_BUFFERS));
const MAX_POOL_BUFFERS: usize = 32;

struct BufferGuard {
    buffer: Vec<u8>,
}

impl BufferGuard {
    #[inline]
    fn new(mut buffer: Vec<u8>, required_len: usize) -> Self {
        // Only allocate/zero when buffer is fresh or dimensions change.
        // Reused pool buffers are already required_len; card rows fully overwrite
        // active pixel areas, and margins remain transparent without re-zeroing 6.5MB.
        if buffer.len() != required_len {
            buffer.clear();
            buffer.resize(required_len, 0);
        }
        Self { buffer }
    }
}

impl Drop for BufferGuard {
    #[inline]
    fn drop(&mut self) {
        let buf = std::mem::take(&mut self.buffer);
        let _ = DROP_POOL.push(buf);
    }
}

impl std::ops::Deref for BufferGuard {
    type Target = Vec<u8>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

impl std::ops::DerefMut for BufferGuard {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buffer
    }
}

// combine two card images and add print numbers = drop image
// we manually copy pixel rows from the card images. this is much faster
// than creating a new blank image and using a library to paste the card images to it.
pub(super) fn create_drop_image(
    left_card: &RawCardImage,
    right_card: &RawCardImage,
    left_card_print: PrintNumber,
    right_card_print: PrintNumber,
) -> Result<Bytes> {
    // count the dimensions of our drop image
    let left_width = left_card.size.width;
    let right_width = right_card.size.width;

    let total_width = left_width + right_width + PADDING_BETWEEN_CARDS * 3;
    let max_card_height = left_card.size.height.max(right_card.size.height);
    let total_height = max_card_height + PADDING_BETWEEN_CARDS * 2;

    // make sure buffer big enough for image (width * height * 4 bytes per pixel)
    let required_len = (total_width * total_height * 4) as usize;

    let mut buffer = BufferGuard::new(DROP_POOL.pop().unwrap_or_default(), required_len);

    // count starting position for the left and right card.
    let left_card_x = PADDING_BETWEEN_CARDS;
    let right_card_x = left_width + PADDING_BETWEEN_CARDS * 2;
    let card_y = PADDING_BETWEEN_CARDS;

    // copy pixels from both cards in a single row pass to keep cache lines hot
    composite_cards(
        &mut buffer,
        left_card,
        right_card,
        total_width,
        left_card_x,
        right_card_x,
        card_y,
    );

    let mut left_print_buf = [0u8; 8];
    let left_print = format_print_number(left_card_print.value(), &mut left_print_buf);

    let mut right_print_buf = [0u8; 8];
    let right_print = format_print_number(right_card_print.value(), &mut right_print_buf);

    // count positions for text and draw it to the image
    let left_print_width = measure_print_number(left_print);
    let right_print_width = measure_print_number(right_print);

    let right_padding = TEXT_PADDING_FROM_EDGE - ref_number_width();

    let left_print_x = (left_card_x + left_width).cast_signed() - right_padding - left_print_width;
    let right_print_x =
        (right_card_x + right_width).cast_signed() - right_padding - right_print_width;
    let print_y = total_height.cast_signed() - TEXT_SIZE as i32 - TEXT_PADDING_FROM_BOTTOM;

    draw_print_number(
        total_width,
        total_height,
        &mut buffer[..required_len],
        left_print,
        Point::new(left_print_x, print_y),
    );
    draw_print_number(
        total_width,
        total_height,
        &mut buffer[..required_len],
        right_print,
        Point::new(right_print_x, print_y),
    );

    // encode final drop image to webp
    encode_webp(total_width, total_height, &buffer[..required_len])
}
