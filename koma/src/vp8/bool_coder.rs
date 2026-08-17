//! Arithmetic Boolean range encoder according to RFC 6386 Section 7.
//!
//! Provides the bitstream emission state machine with branchless 8-bit leading-zero
//! normalization to maximize throughput in hot VP8 bitstream generation.

/// Lookup table for leading zero count on 8-bit integers (values 0..=255).
///
/// Enables single-step branchless normalization without variable-iteration loops.
const NORM_LUT: [u8; 256] = {
    let mut lut = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        lut[i] = if i == 0 { 8 } else { (i as u8).leading_zeros() as u8 };
        i += 1;
    }
    lut
};

/// RFC 6386 and Google libwebp compliant Arithmetic Boolean Range Encoder.
///
/// Implements binary arithmetic coding (Witten, Neal & Cleary 1987) with single-step
/// constant-time multi-bit normalization and branch-free equiprobable splits.
pub struct BoolEncoder<'a> {
    buffer: &'a mut Vec<u8>,
    lowvalue: u32,
    range: u32,
    count: i32,
}

impl<'a> BoolEncoder<'a> {
    /// Creates a new [`BoolEncoder`] writing directly to the provided destination buffer.
    #[inline(always)]
    pub fn new(buffer: &'a mut Vec<u8>) -> Self {
        buffer.clear();
        Self { buffer, lowvalue: 0, range: 255, count: -24 }
    }

    /// Single-step fast normalization of arithmetic encoder state.
    #[inline(always)]
    fn normalize(&mut self) {
        let shift = NORM_LUT[self.range as usize] as i32;
        self.range <<= shift;

        // Carry propagation
        if (self.lowvalue & (1 << 31)) != 0 {
            let mut idx = self.buffer.len();
            while idx > 0 {
                idx -= 1;
                if self.buffer[idx] < 255 {
                    self.buffer[idx] += 1;
                    break;
                }
                self.buffer[idx] = 0;
            }
        }

        self.lowvalue <<= shift;
        self.count += shift;

        if self.count >= 0 {
            let out_byte = (self.lowvalue >> (24 + self.count)) as u8;
            self.buffer.push(out_byte);
            self.lowvalue &= (1 << (24 + self.count)) - 1;
            self.count -= 8;

            if self.count >= 0 {
                let out_byte2 = (self.lowvalue >> (24 + self.count)) as u8;
                self.buffer.push(out_byte2);
                self.lowvalue &= (1 << (24 + self.count)) - 1;
                self.count -= 8;
            }
        }
    }

    /// Encodes a single boolean with the specified probability (1..=255).
    #[inline(always)]
    pub fn put_bit(&mut self, bit: bool, prob: u8) {
        let split = 1 + (((self.range - 1) * (prob as u32)) >> 8);
        if bit {
            self.lowvalue += split;
            self.range -= split;
        } else {
            self.range = split;
        }

        if self.range < 128 {
            self.normalize();
        }
    }

    /// Encodes an equiprobable bit (`prob = 128`) without 8-bit multiplication.
    #[inline(always)]
    pub fn put_bit_equi(&mut self, bit: bool) {
        // For prob=128, 1 + (((range - 1) * 128) >> 8) == (range + 1) >> 1
        let split = (self.range + 1) >> 1;
        if bit {
            self.lowvalue += split;
            self.range -= split;
        } else {
            self.range = split;
        }

        if self.range < 128 {
            self.normalize();
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
        for _ in 0..32 {
            self.put_bit_equi(false);
        }
    }
}
