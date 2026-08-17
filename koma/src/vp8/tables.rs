//! RFC 6386 quantization and coefficient probability tables.
//!
//! Provides the standard VP8 lookup tables for DC/AC quantizer mapping,
//! zigzag coefficient ordering, and baseline entropy probabilities.

/// VP8 RFC 6386 uncompressed frame header start code `[0x9D, 0x01, 0x2A]`.
pub const VP8_START_CODE: [u8; 3] = [0x9D, 0x01, 0x2A];

/// 4x4 Zigzag scan coefficient ordering table.
pub const ZIGZAG: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// RFC 6386 Standard DC Quantization Lookup Table (Indices 0..=127).
pub const DC_QLOOKUP: [i16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 10, 11, 12, 13, 14, 15, 16, 17, 17, 18, 19, 20, 20, 21, 21, 22, 22, 23,
    23, 24, 25, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 37, 38, 39, 40, 41, 42, 43, 44,
    45, 46, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    68, 69, 70, 71, 72, 73, 74, 75, 76, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 91,
    93, 95, 96, 98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118, 122, 124, 126, 128, 130,
    132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
];

/// RFC 6386 Standard AC Quantization Lookup Table (Indices 0..=127).
pub const AC_QLOOKUP: [i16; 128] = [
    4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 60, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94,
    96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140,
    143, 146, 149, 152, 155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205,
    209, 213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
];

/// RFC 6386 Section 13.5: Baseline probability of End-Of-Block (EOB).
pub const PROB_EOB: u8 = 214;
/// RFC 6386 Section 13.5: Baseline probability of zero coefficient.
pub const PROB_ZERO: u8 = 180;
/// RFC 6386 Section 13.5: Baseline probability of coefficient value == 1.
pub const PROB_ONE: u8 = 170;
/// RFC 6386 Section 13.5: Baseline probability of coefficient value == 2.
pub const PROB_TWO: u8 = 150;

/// RFC 6386 Section 13.2: Extra bits probabilities for coefficient categories.
pub const PCAT1: [u8; 1] = [159];
pub const PCAT2: [u8; 2] = [165, 145];
pub const PCAT3: [u8; 3] = [173, 148, 140];
pub const PCAT4: [u8; 4] = [176, 155, 140, 135];
pub const PCAT5: [u8; 5] = [180, 157, 141, 134, 130];
pub const PCAT6: [u8; 11] = [254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];
