//! Library Maintainer Benchmark Suite
//!
//! Shows how a library author would structure benchmarks for their public API:
//! - Groups by module (parsing, collections, encoding)
//! - Regression guards on critical paths
//! - Comparisons between alternative implementations
//!
//! Run with: cargo run --example library_bench -p fluxbench --release

use fluxbench::prelude::*;
use fluxbench::{bench, compare, synthetic, verify};
use std::collections::{BTreeMap, HashMap};
use std::hint::black_box;

// ============================================================================
// Module: Parsing
// ============================================================================

/// Parse integers from a comma-separated string.
#[bench(id = "parse_csv_ints", group = "parsing", tags = "core")]
fn parse_csv_ints(b: &mut Bencher) {
    let input = (0..500)
        .map(|i| i.to_string())
        .collect::<Vec<_>>()
        .join(",");
    b.iter(|| {
        let nums: Vec<i64> = input.split(',').filter_map(|s| s.parse().ok()).collect();
        black_box(nums.len())
    });
}

/// Parse key=value pairs from a config-style string.
#[bench(id = "parse_kv_pairs", group = "parsing", tags = "core")]
fn parse_kv_pairs(b: &mut Bencher) {
    let input: String = (0..200)
        .map(|i| format!("key_{i}=value_{i}"))
        .collect::<Vec<_>>()
        .join("\n");
    b.iter(|| {
        let map: HashMap<&str, &str> = input
            .lines()
            .filter_map(|line| line.split_once('='))
            .collect();
        black_box(map.len())
    });
}

// ============================================================================
// Module: Collections
// ============================================================================

/// HashMap insert + lookup cycle (typical cache pattern).
#[bench(id = "hashmap_cache_cycle", group = "collections", tags = "core")]
fn hashmap_cache_cycle(b: &mut Bencher) {
    b.iter(|| {
        let mut cache: HashMap<u64, u64> = HashMap::with_capacity(256);
        for i in 0u64..256 {
            cache.insert(i, i.wrapping_mul(2654435761));
        }
        let mut hits = 0u64;
        for i in 0u64..256 {
            if cache.contains_key(&i) {
                hits += 1;
            }
        }
        black_box(hits)
    });
}

/// BTreeMap insert + range scan (ordered index pattern).
#[bench(id = "btree_range_scan", group = "collections", tags = "core")]
fn btree_range_scan(b: &mut Bencher) {
    b.iter(|| {
        let mut tree = BTreeMap::new();
        for i in 0u64..256 {
            tree.insert(i, i * 3);
        }
        let sum: u64 = tree.range(50..200).map(|(_, v)| v).sum();
        black_box(sum)
    });
}

/// Vec binary search — sorted array alternative to HashMap.
#[bench(id = "vec_binary_search", group = "collections")]
fn vec_binary_search(b: &mut Bencher) {
    let data: Vec<(u64, u64)> = (0u64..256).map(|i| (i, i * 7)).collect();
    b.iter(|| {
        let mut found = 0u64;
        for key in 0u64..256 {
            if data.binary_search_by_key(&key, |&(k, _)| k).is_ok() {
                found += 1;
            }
        }
        black_box(found)
    });
}

// ============================================================================
// Module: Encoding
// ============================================================================

/// Hex-encode 4 KiB of data.
#[bench(id = "hex_encode_4k", group = "encoding")]
fn hex_encode_4k(b: &mut Bencher) {
    let data: Vec<u8> = (0..4096).map(|i| (i & 0xFF) as u8).collect();
    b.iter(|| {
        let hex: String = data.iter().map(|b| format!("{b:02x}")).collect();
        black_box(hex.len())
    });
}

/// Base64-style encode (simplified — shifts + masks only).
#[bench(id = "b64_encode_4k", group = "encoding")]
fn b64_encode_4k(b: &mut Bencher) {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let data: Vec<u8> = (0..4096).map(|i| (i & 0xFF) as u8).collect();
    b.iter(|| {
        let mut out = Vec::with_capacity(data.len() * 4 / 3 + 4);
        for chunk in data.chunks(3) {
            let b0 = chunk[0] as u32;
            let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
            let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
            let triple = (b0 << 16) | (b1 << 8) | b2;
            out.push(TABLE[((triple >> 18) & 0x3F) as usize]);
            out.push(TABLE[((triple >> 12) & 0x3F) as usize]);
            out.push(TABLE[((triple >> 6) & 0x3F) as usize]);
            out.push(TABLE[(triple & 0x3F) as usize]);
        }
        black_box(out.len())
    });
}

// ============================================================================
// Comparisons
// ============================================================================

#[compare(
    id = "lookup_cmp",
    title = "Lookup Strategy: HashMap vs BTreeMap vs Binary Search",
    benchmarks = ["hashmap_cache_cycle", "btree_range_scan", "vec_binary_search"],
    baseline = "hashmap_cache_cycle",
    metric = "mean"
)]
#[allow(dead_code)]
struct LookupComparison;

#[compare(
    id = "encode_cmp",
    title = "Encoding: Hex vs Base64",
    benchmarks = ["hex_encode_4k", "b64_encode_4k"],
    baseline = "hex_encode_4k",
    metric = "mean"
)]
#[allow(dead_code)]
struct EncodingComparison;

// ============================================================================
// Synthetic metrics
// ============================================================================

/// CSV parsing throughput (500 items per call).
#[synthetic(
    id = "csv_items_per_sec",
    formula = "500 / parse_csv_ints * 1000000000",
    unit = "items/s"
)]
#[allow(dead_code)]
struct CsvThroughput;

/// Hex encoding throughput (4096 bytes per call).
#[synthetic(
    id = "hex_bytes_per_sec",
    formula = "4096 / hex_encode_4k * 1000000000",
    unit = "B/s"
)]
#[allow(dead_code)]
struct HexThroughput;

// ============================================================================
// Regression guards
// ============================================================================

/// Core parsing must stay under 50 us.
#[verify(expr = "parse_csv_ints < 50000", severity = "critical")]
#[allow(dead_code)]
struct CsvParsingBudget;

/// HashMap cache cycle must stay under 20 us.
#[verify(expr = "hashmap_cache_cycle < 20000", severity = "critical")]
#[allow(dead_code)]
struct CacheCycleBudget;

/// HashMap must beat BTreeMap for point lookups.
#[verify(expr = "hashmap_cache_cycle < btree_range_scan", severity = "warning")]
#[allow(dead_code)]
struct HashMapFasterThanBTree;

fn main() {
    if let Err(e) = fluxbench::run() {
        eprintln!("Error: {e}");
        std::process::exit(1);
    }
}
