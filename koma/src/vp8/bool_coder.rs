//! RFC 6386 Arithmetic Boolean Range Coder.
//!
//! Implements the binary arithmetic entropy encoder specified in RFC 6386 Section 7.
//! VP8's boolean coder uses an 8-bit range `[128, 255]`, 32-bit low value, and a 
//! carry bit mechanism with standard bit-level renormalizations.

/// Standard RFC 6386 Boolean Range Encoder.
#[derive(Debug, Clone)]
pub struct BoolEncoder {
    /// Low value register.
    low: u32,
    /// Range register (always kept in `128..=255` after renormalization).
    range: u32,
    /// Number of bits of precision currently accumulated in `low` before output.
    count: i32,
    /// Destination byte buffer.
    buffer: Vec<u8>,
}

impl Default for BoolEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl BoolEncoder {
    /// Creates a new, initialized boolean range coder with an empty buffer.
    pub fn new() -> Self {
        Self::with_capacity(4096)
    }

    /// Creates a new boolean coder with pre-allocated buffer capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            low: 0,
            range: 255,
            count: -24,
            buffer: Vec::with_capacity(capacity),
        }
    }

    /// Encodes a single boolean value with a given probability (0..255).
    /// `prob` represents the probability of the bit being `0` in units of 1/256.
    #[inline(always)]
    pub fn put_bool(&mut self, val: bool, prob: u8) {
        let split = 1 + (((self.range - 1) * (prob as u32)) >> 8);

        if !val {
            self.range = split;
        } else {
            self.low += split;
            self.range -= split;
        }

        // Renormalize if range falls below 128
        let shift = (self.range as u8).leading_zeros() as i32;
        self.range <<= shift;
        self.count += shift;

        if self.count >= 0 {
            let offset = self.count - shift;
            if (self.low << offset) & 0x8000_0000 != 0 {
                // Carry propagation backward through buffer
                let mut idx = self.buffer.len();
                while idx > 0 {
                    idx -= 1;
                    self.buffer[idx] = self.buffer[idx].wrapping_add(1);
                    if self.buffer[idx] != 0 {
                        break;
                    }
                }
            }

            let mut out = (self.low >> (24 - offset)) as u8;
            self.buffer.push(out);
            self.low <<= 8;
            self.count -= 8;
        }
        self.low <<= shift;
    }

    /// Encodes an equiprobable bit (probability = 128 / 256 = 50%).
    #[inline(always)]
    pub fn put_bit(&mut self, val: bool) {
        self.put_bool(val, 128);
    }

    /// Encodes an unsigned integer `val` of `bits` length using uniform probability (128).
    /// Bits are encoded MSB first.
    pub fn put_uint(&mut self, val: u32, bits: usize) {
        for i in (0..bits).rev() {
            self.put_bit(((val >> i) & 1) != 0);
        }
    }

    /// Encodes a signed integer using sign-magnitude with uniform probability.
    pub fn put_signed(&mut self, val: i32, bits: usize) {
        let mag = val.unsigned_abs();
        self.put_uint(mag, bits);
        if val != 0 {
            self.put_bit(val < 0);
        }
    }

    /// Encodes a value using an arbitrary probability tree.
    pub fn put_tree(&mut self, tree: &[i8], probs: &[u8], mut value: usize) {
        let mut i: usize = 0;
        loop {
            let prob = probs[i >> 1];
            let bit = (value & 1) != 0;
            self.put_bool(bit, prob);
            
            let next = if !bit { tree[i] } else { tree[i + 1] };
            if next <= 0 {
                break;
            }
            i = next as usize;
            value >>= 1;
        }
    }

    /// Flushes remaining bits and pads the bitstream according to RFC 6386 Section 7.4.
    pub fn finish(mut self) -> Vec<u8> {
        let mut shift = 27 - ((self.range as u8).leading_zeros() as i32);
        self.range = self.range.wrapping_shl((self.range as u8).leading_zeros());

        while shift > 0 {
            let mut out = (self.low >> (shift + 8)) as u8;
            if (self.low << (24 - shift)) & 0x8000_0000 != 0 {
                let mut idx = self.buffer.len();
                while idx > 0 {
                    idx -= 1;
                    self.buffer[idx] = self.buffer[idx].wrapping_add(1);
                    if self.buffer[idx] != 0 {
                        break;
                    }
                }
            }
            self.buffer.push(out);
            self.low <<= 8;
            shift -= 8;
        }

        self.buffer
    }
}
