//! Arithmetic Boolean range encoder according to RFC 6386 Section 7 and libwebp.
//!
//! Provides the bitstream emission state machine with lookup tables for normalization
//! and carry propagation matching Google libwebp bit_writer_utils.

/// Lookup table for bit shifts during normalization: `kNorm[i] = 8 - log2(i)`.
pub const K_NORM: [u8; 128] = [
    7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
];

/// Lookup table for renormalized ranges: `kNewRange[i] = ((i + 1) << kNorm[i]) - 1`.
pub const K_NEW_RANGE: [u8; 128] = [
    127, 127, 191, 127, 159, 191, 223, 127, 143, 159, 175, 191, 207, 223, 239, 127, 135, 143, 151,
    159, 167, 175, 183, 191, 199, 207, 215, 223, 231, 239, 247, 127, 131, 135, 139, 143, 147, 151,
    155, 159, 163, 167, 171, 175, 179, 183, 187, 191, 195, 199, 203, 207, 211, 215, 219, 223, 227,
    231, 235, 239, 243, 247, 251, 127, 129, 131, 133, 135, 137, 139, 141, 143, 145, 147, 149, 151,
    153, 155, 157, 159, 161, 163, 165, 167, 169, 171, 173, 175, 177, 179, 181, 183, 185, 187, 189,
    191, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 225, 227,
    229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 127,
];

/// RFC 6386 and Google libwebp compliant Arithmetic Boolean Range Encoder.
pub struct BoolEncoder<'a> {
    buffer: &'a mut Vec<u8>,
    range: i32,
    value: i32,
    run: usize,
    nb_bits: i32,
}

impl<'a> BoolEncoder<'a> {
    /// Creates a new [`BoolEncoder`] writing directly to the provided destination buffer.
    #[inline(always)]
    pub fn new(buffer: &'a mut Vec<u8>) -> Self {
        buffer.clear();
        Self {
            buffer,
            range: 254, // 255 - 1
            value: 0,
            run: 0,
            nb_bits: -8,
        }
    }

    #[inline(always)]
    fn flush(&mut self) {
        let s = 8 + self.nb_bits;
        let bits = self.value >> s;
        self.value -= bits << s;
        self.nb_bits -= 8;
        if (bits & 0xFF) != 0xFF {
            if (bits & 0x100) != 0 {
                // Carry propagation over previous byte
                let idx = self.buffer.len();
                if idx > 0 {
                    self.buffer[idx - 1] = self.buffer[idx - 1].wrapping_add(1);
                }
            }
            if self.run > 0 {
                let fill_val = if (bits & 0x100) != 0 { 0x00 } else { 0xFF };
                for _ in 0..self.run {
                    self.buffer.push(fill_val);
                }
                self.run = 0;
            }
            self.buffer.push((bits & 0xFF) as u8);
        } else {
            self.run += 1;
        }
    }

    /// Encodes a single boolean with the specified probability (1..=255).
    #[inline(always)]
    pub fn put_bit(&mut self, bit: bool, prob: u8) {
        let split = (self.range * (prob as i32)) >> 8;
        if bit {
            self.value += split + 1;
            self.range -= split + 1;
        } else {
            self.range = split;
        }
        if self.range < 127 {
            let shift = K_NORM[self.range as usize] as i32;
            self.range = K_NEW_RANGE[self.range as usize] as i32;
            self.value <<= shift;
            self.nb_bits += shift;
            if self.nb_bits > 0 {
                self.flush();
            }
        }
    }

    /// Encodes an equiprobable bit (`prob = 128`) without 8-bit multiplication.
    #[inline(always)]
    pub fn put_bit_equi(&mut self, bit: bool) {
        let split = self.range >> 1;
        if bit {
            self.value += split + 1;
            self.range -= split + 1;
        } else {
            self.range = split;
        }
        if self.range < 127 {
            self.range = K_NEW_RANGE[self.range as usize] as i32;
            self.value <<= 1;
            self.nb_bits += 1;
            if self.nb_bits > 0 {
                self.flush();
            }
        }
    }

    /// Encodes a fixed-width unsigned literal value.
    #[inline(always)]
    pub fn put_literal(&mut self, data: u32, bits: usize) {
        for bit_idx in (0..bits).rev() {
            self.put_bit_equi((data & (1 << bit_idx)) != 0);
        }
    }

    /// Flushes remaining bits into the buffer to produce a valid RFC 6386 bitstream.
    #[inline(always)]
    pub fn finish(mut self) {
        let pad_bits = (9 - self.nb_bits).max(0) as usize;
        for _ in 0..pad_bits {
            self.put_bit_equi(false);
        }
        self.nb_bits = 0;
        self.flush();
    }
}

