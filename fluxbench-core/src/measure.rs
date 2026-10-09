//! High-Precision Timing
//!
//! Uses RDTSCP on x86_64 and CNTVCT_EL0 on AArch64 for minimal overhead
//! cycle counting, with fallback to std::time::Instant on other platforms.

use std::sync::OnceLock;
use std::time::Duration;

// ─── Inline cycle counter helpers ────────────────────────────────────────────

/// Read the CPU cycle/tick counter (platform-specific).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn read_cycles() -> u64 {
    // SAFETY: RDTSCP is available on all x86_64 CPUs since ~2006.
    // It is serializing by design — waits for all prior instructions
    // to complete before reading the cycle counter.
    unsafe {
        let mut _aux: u32 = 0;
        std::arch::x86_64::__rdtscp(&mut _aux)
    }
}

/// Read the virtual counter timer on AArch64 (comparable to x86 TSC).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn read_cycles() -> u64 {
    let cnt: u64;
    // SAFETY: CNTVCT_EL0 is accessible from EL0 (userspace) on all
    // AArch64 implementations. It provides a monotonically increasing
    // counter at a fixed frequency (typically the system timer frequency).
    unsafe {
        std::arch::asm!("mrs {}, cntvct_el0", out(reg) cnt, options(nostack, nomem));
    }
    cnt
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
#[inline(always)]
fn read_cycles() -> u64 {
    0
}

/// Whether this platform provides real cycle counters.
pub const HAS_CYCLE_COUNTER: bool = cfg!(target_arch = "x86_64") || cfg!(target_arch = "aarch64");

// ─── Instant ─────────────────────────────────────────────────────────────────

/// High-precision instant for benchmarking
#[derive(Debug, Clone, Copy)]
pub struct Instant {
    instant: std::time::Instant,
    tsc: u64,
}

impl Instant {
    /// Capture current instant
    #[inline(always)]
    pub fn now() -> Self {
        let tsc = read_cycles();
        Self {
            instant: std::time::Instant::now(),
            tsc,
        }
    }

    /// Compute elapsed time since this instant
    #[inline(always)]
    pub fn elapsed(&self) -> Duration {
        self.instant.elapsed()
    }

    /// Raw cycle/tick count (non-zero on x86_64 and aarch64)
    #[inline(always)]
    pub fn cycles(&self) -> u64 {
        self.tsc
    }
}

// ─── Timer ───────────────────────────────────────────────────────────────────

/// Timer for measuring benchmark iterations
pub struct Timer {
    start: Instant,
    cycles_start: u64,
}

impl Timer {
    /// Start a new timer
    #[inline(always)]
    pub fn start() -> Self {
        let cycles_start = read_cycles();
        Self {
            start: Instant::now(),
            cycles_start,
        }
    }

    /// Stop the timer and return elapsed nanoseconds and cycles
    #[inline(always)]
    pub fn stop(&self) -> (u64, u64) {
        let elapsed = self.start.elapsed();
        let nanos = elapsed.as_nanos() as u64;
        let cycles = read_cycles().saturating_sub(self.cycles_start);
        (nanos, cycles)
    }
}

// ─── Timer overhead calibration ──────────────────────────────────────────────

/// Empty timer pairs timed per calibration batch.
const CALIBRATION_PAIRS: usize = 1_000;

/// Calibration batches. The cheapest batch is the least disturbed one.
const CALIBRATION_BATCHES: usize = 20;

/// Cost of one empty `Timer::start` + `Timer::stop` pair: (nanoseconds, cycles).
static TIMER_OVERHEAD: OnceLock<(f64, f64)> = OnceLock::new();

/// Time batches of empty timer pairs and return the cheapest per-pair cost.
///
/// An outer timer spans each whole batch. A coarse clock, such as the 100 ns
/// Windows `Instant`, then still resolves the per-pair cost.
fn calibrate_timer() -> (f64, f64) {
    let mut best_ns = f64::MAX;
    let mut best_cycles = f64::MAX;
    for _ in 0..CALIBRATION_BATCHES {
        let outer = Timer::start();
        for _ in 0..CALIBRATION_PAIRS {
            let timer = Timer::start();
            std::hint::black_box(std::hint::black_box(&timer).stop());
        }
        let (ns, cycles) = outer.stop();
        best_ns = best_ns.min(ns as f64 / CALIBRATION_PAIRS as f64);
        best_cycles = best_cycles.min(cycles as f64 / CALIBRATION_PAIRS as f64);
    }
    (best_ns, best_cycles)
}

/// Nanoseconds one empty timer pair costs.
///
/// The first call calibrates from batches of empty pairs. Later calls read the cached value.
/// The engine subtracts this once per timed region.
pub fn timer_overhead_ns() -> f64 {
    TIMER_OVERHEAD.get_or_init(calibrate_timer).0
}

/// Cycles one empty timer pair costs. It is 0 without a cycle counter.
///
/// It shares the calibration and cache with `timer_overhead_ns`.
pub fn cycles_overhead() -> f64 {
    TIMER_OVERHEAD.get_or_init(calibrate_timer).1
}

/// Set CPU affinity to pin the current thread to a specific core
///
/// This improves TSC stability by avoiding core migrations.
#[cfg(target_os = "linux")]
pub fn pin_to_cpu(cpu: usize) -> Result<(), std::io::Error> {
    use std::mem::MaybeUninit;

    unsafe {
        let mut set = MaybeUninit::<libc::cpu_set_t>::zeroed();
        let set_ref = set.assume_init_mut();

        libc::CPU_ZERO(set_ref);
        libc::CPU_SET(cpu, set_ref);

        let result = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), set_ref);

        if result == 0 {
            Ok(())
        } else {
            Err(std::io::Error::last_os_error())
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_to_cpu(_cpu: usize) -> Result<(), std::io::Error> {
    // CPU pinning not supported on this platform
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instant_elapsed() {
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = start.elapsed();

        // Should be at least 10ms
        assert!(elapsed >= Duration::from_millis(5));
        // Should be less than 100ms (accounting for scheduling)
        assert!(elapsed < Duration::from_millis(100));
    }

    #[test]
    fn test_timer() {
        let timer = Timer::start();
        std::thread::sleep(Duration::from_millis(10));
        let (nanos, _cycles) = timer.stop();

        // Should be at least 5ms in nanos
        assert!(nanos >= 5_000_000);
    }

    #[test]
    fn timer_overhead_calibrated_positive_and_small() {
        let ns = timer_overhead_ns();
        assert!(ns > 0.0, "timer overhead is {ns} ns");
        assert!(ns < 1000.0, "timer overhead is {ns} ns");
        // The cached value is stable across calls.
        assert_eq!(ns, timer_overhead_ns());
        assert!(cycles_overhead() >= 0.0);
    }

    #[test]
    fn test_cycle_counter() {
        if HAS_CYCLE_COUNTER {
            let a = read_cycles();
            let b = read_cycles();
            assert!(b >= a, "cycle counter should be monotonic");
        }
    }
}
