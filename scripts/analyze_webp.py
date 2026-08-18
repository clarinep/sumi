#!/usr/bin/env python3
"""
WebP Bitstream & Chunk Analyzer for Sumi / Koma
------------------------------------------------
Deeply inspects WebP containers (RIFF, VP8X, ALPH, VP8) to diagnose bitstream validity,
chunk alignment, alpha transparency distribution, and keyframe headers.
"""

import sys
import os
import struct
import glob

def analyze_file(filepath):
    print("=" * 80)
    print(f"ANALYZING WEBP FILE: {filepath}")
    print("=" * 80)

    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        return False

    with open(filepath, "rb") as f:
        data = f.read()

    file_len = len(data)
    print(f"File Size: {file_len} bytes")

    if file_len < 12:
        print("Error: File is too small to contain a valid RIFF/WebP header.")
        return False

    riff_magic = data[:4]
    print(f"RIFF Magic: {riff_magic} ({riff_magic.decode('latin1', 'replace')})")
    if riff_magic != b"RIFF":
        if b"<!DOCTYPE" in data[:50] or b"<html" in data[:50]:
            print("CRITICAL FAILURE: File contains HTML text (SPA fallback / 404 page), not binary WebP!")
        else:
            print("CRITICAL FAILURE: Invalid RIFF header.")
        return False

    riff_size, = struct.unpack("<I", data[4:8])
    webp_magic = data[8:12]
    print(f"RIFF Payload Size: {riff_size} bytes (Expected file size: {riff_size + 8})")
    print(f"WEBP Magic: {webp_magic} ({webp_magic.decode('latin1', 'replace')})")

    if webp_magic != b"WEBP":
        print("CRITICAL FAILURE: Missing 'WEBP' fourcc identifier.")
        return False

    offset = 12
    chunk_index = 0
    canvas_w = 0
    canvas_h = 0
    has_alpha_flag = False

    while offset < file_len:
        chunk_index += 1
        if offset + 8 > file_len:
            print(f"\n[Chunk #{chunk_index}] Truncated chunk header at offset {offset}")
            break

        tag = data[offset:offset+4].decode("latin1", "replace")
        chunk_size, = struct.unpack("<I", data[offset+4:offset+8])
        chunk_data = data[offset+8:offset+8+chunk_size]
        actual_chunk_len = len(chunk_data)

        print(f"\n--- [Chunk #{chunk_index}] FourCC: '{tag}' | Payload Size: {chunk_size} bytes | Offset: {offset} ---")

        if actual_chunk_len < chunk_size:
            print(f"  WARNING: Chunk truncated! Available bytes: {actual_chunk_len}, expected: {chunk_size}")

        if tag == "VP8X":
            if chunk_size >= 10:
                flags, = struct.unpack("<I", chunk_data[:4])
                canvas_w = int.from_bytes(chunk_data[4:7], "little") + 1
                canvas_h = int.from_bytes(chunk_data[7:10], "little") + 1
                has_alpha_flag = bool(flags & 0x10)
                print(f"  Container Type: Extended WebP (VP8X)")
                print(f"  Canvas Dimensions: {canvas_w} x {canvas_h}")
                print(f"  Flags (0x{flags:08x}):")
                print(f"    - ICC Profile:   {bool(flags & 0x20)}")
                print(f"    - Alpha (ALPH):  {has_alpha_flag}")
                print(f"    - EXIF Metadata: {bool(flags & 0x08)}")
                print(f"    - XMP Metadata:  {bool(flags & 0x04)}")
                print(f"    - Animation:     {bool(flags & 0x02)}")
            else:
                print("  ERROR: VP8X payload is smaller than required 10 bytes.")

        elif tag == "ALPH":
            if len(chunk_data) >= 1:
                header_byte = chunk_data[0]
                comp_method = header_byte & 0x03
                filter_method = (header_byte >> 2) & 0x03
                pre_process = (header_byte >> 4) & 0x03
                reserved = (header_byte >> 6) & 0x03

                comp_names = {0: "Uncompressed raw", 1: "Lossless compressed (VP8L stream)"}
                filter_names = {0: "None", 1: "Horizontal", 2: "Vertical", 3: "Gradient"}

                print(f"  Header Byte: 0x{header_byte:02x}")
                print(f"    - Compression: {comp_names.get(comp_method, f'Unknown ({comp_method})')}")
                print(f"    - Filter:      {filter_names.get(filter_method, f'Unknown ({filter_method})')}")
                print(f"    - Preprocess:  {pre_process}")
                print(f"    - Reserved:    {reserved}")

                alpha_bytes = chunk_data[1:]
                total_alpha = len(alpha_bytes)
                zero_count = sum(1 for b in alpha_bytes if b == 0)
                opaque_count = sum(1 for b in alpha_bytes if b == 255)
                translucent_count = total_alpha - zero_count - opaque_count

                print(f"  Alpha Plane Statistics (Total values: {total_alpha}):")
                print(f"    - Fully Transparent (0):   {zero_count:8d} ({zero_count/max(1,total_alpha)*100:.2f}%)")
                print(f"    - Fully Opaque (255):      {opaque_count:8d} ({opaque_count/max(1,total_alpha)*100:.2f}%)")
                print(f"    - Translucent (1-254):     {translucent_count:8d} ({translucent_count/max(1,total_alpha)*100:.2f}%)")

                if comp_method == 0 and total_alpha > 0 and canvas_w > 0 and canvas_h > 0:
                    # Print ASCII density mini-map (16x16 downsample)
                    print("\n  Alpha Transparency Map (O: Opaque, . : Semi, _ : Transparent):")
                    grid_w, grid_h = 40, 15
                    for gy in range(grid_h):
                        row_str = "    "
                        py = int(gy * canvas_h / grid_h)
                        for gx in range(grid_w):
                            px = int(gx * canvas_w / grid_w)
                            idx = py * canvas_w + px
                            if idx < total_alpha:
                                val = alpha_bytes[idx]
                                if val == 255: row_str += "O"
                                elif val > 0:  row_str += "."
                                else:          row_str += "_"
                            else:
                                row_str += " "
                        print(row_str)
            else:
                print("  ERROR: ALPH chunk is empty.")

        elif tag == "VP8 ":
            if len(chunk_data) >= 10:
                tag0, tag1, tag2 = chunk_data[0], chunk_data[1], chunk_data[2]
                is_keyframe = (tag0 & 1) == 0
                version = (tag0 >> 1) & 0x07
                show_frame = (tag0 >> 4) & 0x01
                first_part_size = (tag0 >> 5) | (tag1 << 3) | (tag2 << 11)

                print(f"  VP8 Frame Header:")
                print(f"    - Frame Type:        {'Keyframe (intra)' if is_keyframe else 'Inter-frame (predicted)'}")
                print(f"    - Version:           {version}")
                print(f"    - Show Frame:        {bool(show_frame)}")
                print(f"    - 1st Partition Len: {first_part_size} bytes")

                if is_keyframe:
                    sync_code = chunk_data[3:6]
                    sync_hex = sync_code.hex()
                    is_sync_valid = (sync_code == b'\x9d\x01\x2a')
                    sync_status = "(VALID VP8 Keyframe: 9d012a)" if is_sync_valid else "(INVALID SYNC CODE!)"
                    print(f"    - Sync Code:         0x{sync_hex} {sync_status}")

                    w_raw = int.from_bytes(chunk_data[6:8], "little")
                    h_raw = int.from_bytes(chunk_data[8:10], "little")
                    vp8_w = w_raw & 0x3FFF
                    vp8_h = h_raw & 0x3FFF
                    h_scale = w_raw >> 14
                    v_scale = h_raw >> 14

                    print(f"    - Encoded Frame WxH: {vp8_w} x {vp8_h} (H-scale: {h_scale}, V-scale: {v_scale})")
                    print(f"    - Macroblock Grid:   {(vp8_w + 15) // 16} x {(vp8_h + 15) // 16} MBs")
            else:
                print("  ERROR: VP8 chunk is shorter than 10 bytes.")

        elif tag == "VP8L":
            print("  Lossless WebP Stream (VP8L)")
            if len(chunk_data) >= 5:
                sig = chunk_data[0]
                print(f"  VP8L Signature Byte: 0x{sig:02x} {'(VALID: 0x2f)' if sig == 0x2F else '(INVALID)'}")

        # RIFF chunks are padded to even 2-byte boundaries
        offset += 8 + chunk_size + (chunk_size & 1)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    return True

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target:
        analyze_file(target)
    else:
        # Search for webp files in generated/ or current directory
        patterns = ["generated/*.webp", "*.webp"]
        found = []
        for p in patterns:
            found.extend(glob.glob(p))
        if found:
            for f in found:
                analyze_file(f)
        else:
            print("No .webp files provided or found in generated/*.webp.")
            print("Usage: python3 scripts/analyze_webp.py <path-to-file.webp>")
