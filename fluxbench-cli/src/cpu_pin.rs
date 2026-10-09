//! CPU assignment for isolated workers.
//!
//! The supervisor reads the process affinity mask once and gives each worker one core.
//! CPU 0 takes most IRQs on Linux, so it is skipped unless it is the only core.

/// CPUs the usable pool holds: `allowed` without CPU 0, unless CPU 0 is the only one.
pub(crate) fn usable_cpus(allowed: &[usize]) -> Vec<usize> {
    if allowed.len() <= 1 {
        return allowed.to_vec();
    }
    allowed.iter().copied().filter(|&cpu| cpu != 0).collect()
}

/// Assign one CPU per worker. Worker `i` gets `usable[i % usable.len()]`.
///
/// An empty `allowed` list gives `None` for every worker.
/// More jobs than CPUs makes workers share cores.
pub(crate) fn assign_cpus(allowed: &[usize], jobs: usize) -> Vec<Option<usize>> {
    let usable = usable_cpus(allowed);
    (0..jobs)
        .map(|i| {
            if usable.is_empty() {
                None
            } else {
                usable.get(i % usable.len()).copied()
            }
        })
        .collect()
}

/// Read the CPUs this process may run on. Other platforms and read errors give an empty list.
#[cfg(target_os = "linux")]
pub(crate) fn allowed_cpus() -> Vec<usize> {
    // SAFETY: `set` is zeroed and sized for `sched_getaffinity`. `CPU_ISSET` reads inside it.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize)
            .filter(|&cpu| libc::CPU_ISSET(cpu, &set))
            .collect()
    }
}

/// Read the CPUs this process may run on. Other platforms give an empty list (no pinning).
#[cfg(not(target_os = "linux"))]
pub(crate) fn allowed_cpus() -> Vec<usize> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_cpu_is_used_even_when_zero() {
        assert_eq!(assign_cpus(&[0], 2), vec![Some(0), Some(0)]);
        assert_eq!(assign_cpus(&[5], 1), vec![Some(5)]);
    }

    #[test]
    fn cpu_zero_is_skipped_when_others_exist() {
        assert_eq!(
            assign_cpus(&[0, 1, 2, 3], 3),
            vec![Some(1), Some(2), Some(3)]
        );
    }

    #[test]
    fn assignment_wraps_when_jobs_exceed_cpus() {
        assert_eq!(
            assign_cpus(&[0, 1, 2], 5),
            vec![Some(1), Some(2), Some(1), Some(2), Some(1)]
        );
    }

    #[test]
    fn empty_allowed_list_gives_none() {
        assert_eq!(assign_cpus(&[], 3), vec![None, None, None]);
    }

    #[test]
    fn zero_jobs_gives_empty_list() {
        assert!(assign_cpus(&[1, 2], 0).is_empty());
    }

    #[test]
    fn usable_pool_drops_cpu_zero_only_with_company() {
        assert_eq!(usable_cpus(&[0]), vec![0]);
        assert_eq!(usable_cpus(&[0, 2]), vec![2]);
        assert_eq!(usable_cpus(&[3, 4]), vec![3, 4]);
    }
}
