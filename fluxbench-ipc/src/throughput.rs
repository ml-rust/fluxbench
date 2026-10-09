//! Work per routine call, for throughput rates.

use rkyv::{Archive, Deserialize as RkyvDeserialize, Serialize as RkyvSerialize};

/// Work one routine call processes.
///
/// The report divides it by the mean time per call to get a rate per second.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Archive, RkyvSerialize, RkyvDeserialize)]
pub enum Throughput {
    /// Bytes per call. Reports show binary units such as MiB/s.
    Bytes(u64),
    /// Elements per call, such as items, records, or tokens.
    Elements(u64),
}

impl Throughput {
    /// Rate per second at `mean_ns` per call. `None` for a mean at or below 0.
    pub fn per_second(self, mean_ns: f64) -> Option<f64> {
        let amount = match self {
            Throughput::Bytes(n) | Throughput::Elements(n) => n as f64,
        };
        (mean_ns.is_finite() && mean_ns > 0.0).then(|| amount * 1e9 / mean_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rkyv::rancor;

    #[test]
    fn per_second_scales_by_mean() {
        assert_eq!(
            Throughput::Bytes(1024).per_second(1_000.0),
            Some(1_024_000_000.0)
        );
        assert_eq!(Throughput::Elements(4).per_second(2.0), Some(2e9));
    }

    #[test]
    fn per_second_rejects_zero_and_bad_means() {
        assert_eq!(Throughput::Bytes(8).per_second(0.0), None);
        assert_eq!(Throughput::Bytes(8).per_second(-1.0), None);
        assert_eq!(Throughput::Bytes(8).per_second(f64::NAN), None);
        assert_eq!(Throughput::Bytes(8).per_second(f64::INFINITY), None);
    }

    #[test]
    fn rkyv_roundtrip() {
        for original in [Throughput::Bytes(7), Throughput::Elements(u64::MAX)] {
            let bytes = rkyv::to_bytes::<rancor::Error>(&original).unwrap();
            let decoded = rkyv::from_bytes::<Throughput, rancor::Error>(&bytes).unwrap();
            assert_eq!(original, decoded);
        }
    }
}
