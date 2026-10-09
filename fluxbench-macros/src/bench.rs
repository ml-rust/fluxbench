//! `#[flux::bench]`: benchmark registration.

use crate::attr;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::ItemFn;

pub fn bench_impl(args: TokenStream2, func: ItemFn) -> Result<TokenStream2, syn::Error> {
    // Parse configuration first (need to know if args is set before validating signature)
    let config = parse_bench_config(args)?;

    // Validate signature
    validate_signature(&func, !config.args.is_empty())?;

    // Generate identifiers
    let fn_name = &func.sig.ident;
    let fn_name_str = fn_name.to_string();

    // Shared config values
    let base_id = config.id.clone().unwrap_or_else(|| fn_name_str.clone());
    let group = config
        .group
        .clone()
        .unwrap_or_else(|| "default".to_string());
    let severity = config
        .severity
        .clone()
        .unwrap_or(quote! { ::fluxbench::Severity::Info });
    let threshold = config.threshold.unwrap_or(0.0);
    let budget_ns = attr::option(config.budget_ns);
    let tags: Vec<_> = config.tags.iter().map(|t| quote! { #t }).collect();

    let depends_on_tokens: Vec<_> = config.depends_on.iter().map(|d| quote! { #d }).collect();

    let warmup_ns_expr = attr::option(config.warmup_ns);
    let measurement_ns_expr = attr::option(config.measurement_ns);
    let samples_expr = attr::option(config.samples);
    let sample_count_expr = attr::option(config.sample_count);
    let iters_per_sample_expr = attr::option(config.iters_per_sample);
    let min_iterations_expr = attr::option(config.min_iterations);
    let max_iterations_expr = attr::option(config.max_iterations);

    // ── Parameterized path: generate one BenchmarkDef per arg value ──────
    if !config.args.is_empty() {
        let is_async = func.sig.asyncness.is_some();

        let mut registrations = Vec::new();
        for (i, arg_expr) in config.args.iter().enumerate() {
            let arg_label = expr_to_label(arg_expr);
            let bench_id = format!("{}@{}", base_id, arg_label);
            let wrapper_name = format_ident!("_flux_wrapper_{}_{}", fn_name, i);

            let runner_block = if is_async {
                let body_call = quote! { #fn_name(bencher, #arg_expr) };
                generate_async_runner(&config.async_runtime, body_call)
            } else {
                quote! {
                    #fn_name(bencher, #arg_expr);
                }
            };

            registrations.push(quote! {
                #[doc(hidden)]
                #[allow(non_snake_case)]
                fn #wrapper_name(bencher: &mut ::fluxbench::Bencher) {
                    #runner_block
                }

                ::fluxbench::internal::inventory::submit! {
                    ::fluxbench::BenchmarkDef {
                        id: #bench_id,
                        name: #fn_name_str,
                        group: #group,
                        severity: #severity,
                        threshold: #threshold,
                        budget_ns: #budget_ns,
                        tags: &[#(#tags),*],
                        runner_fn: #wrapper_name,
                        file: file!(),
                        line: line!(),
                        module_path: module_path!(),
                        warmup_ns: #warmup_ns_expr,
                        measurement_ns: #measurement_ns_expr,
                        samples: #samples_expr,
                        sample_count: #sample_count_expr,
                        iters_per_sample: #iters_per_sample_expr,
                        min_iterations: #min_iterations_expr,
                        max_iterations: #max_iterations_expr,
                        depends_on: &[#(#depends_on_tokens),*],
                    }
                }
            });
        }

        return Ok(quote! {
            #func
            #(#registrations)*
        });
    }

    // ── Normal (non-parameterized) path ──────────────────────────────────
    let wrapper_name = format_ident!("_flux_wrapper_{}", fn_name);

    let is_async = func.sig.asyncness.is_some();
    let runner_block = if is_async {
        generate_async_runner(&config.async_runtime, quote! { #fn_name(bencher) })
    } else {
        quote! {
            #fn_name(bencher);
        }
    };

    Ok(quote! {
        #func

        #[doc(hidden)]
        #[allow(non_snake_case)]
        fn #wrapper_name(bencher: &mut ::fluxbench::Bencher) {
            #runner_block
        }

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::BenchmarkDef {
                id: #base_id,
                name: #fn_name_str,
                group: #group,
                severity: #severity,
                threshold: #threshold,
                budget_ns: #budget_ns,
                tags: &[#(#tags),*],
                runner_fn: #wrapper_name,
                file: file!(),
                line: line!(),
                module_path: module_path!(),
                warmup_ns: #warmup_ns_expr,
                measurement_ns: #measurement_ns_expr,
                samples: #samples_expr,
                sample_count: #sample_count_expr,
                iters_per_sample: #iters_per_sample_expr,
                min_iterations: #min_iterations_expr,
                max_iterations: #max_iterations_expr,
                depends_on: &[#(#depends_on_tokens),*],
            }
        }
    })
}

/// Generate the wrapper body for an async benchmark.
///
/// `body_call` creates the body future, for example `bench(bencher, arg)`.
/// The wrapper builds one runtime and attaches it to the bencher.
/// `drive_async_body` runs the body outside `block_on`, so `iter_async` can block per batch.
/// A build error or a context error goes to the bencher. Nothing panics.
fn generate_async_runner(config: &AsyncRuntimeConfig, body_call: TokenStream2) -> TokenStream2 {
    let (runtime_builder, runtime_config) = async_runtime_tokens(config);

    quote! {
        let __flux_rt = match ::fluxbench::internal::tokio::runtime::Builder::#runtime_builder
            #runtime_config
            .build()
        {
            ::std::result::Result::Ok(__flux_rt) => ::std::sync::Arc::new(__flux_rt),
            ::std::result::Result::Err(__flux_error) => {
                bencher.attach_runtime_error(__flux_error.to_string());
                return;
            }
        };
        bencher.attach_runtime(::std::sync::Arc::clone(&__flux_rt));
        if let ::std::result::Result::Err(__flux_error) =
            ::fluxbench::internal::drive_async_body(&__flux_rt, #body_call)
        {
            bencher.record_error(__flux_error);
        }
    }
}

#[derive(Debug, Clone)]
enum AsyncRuntimeConfig {
    CurrentThread {
        enable_time: bool,
        enable_io: bool,
    },
    MultiThread {
        worker_threads: Option<usize>,
        enable_time: bool,
        enable_io: bool,
    },
}

impl Default for AsyncRuntimeConfig {
    fn default() -> Self {
        AsyncRuntimeConfig::CurrentThread {
            enable_time: true,
            enable_io: true,
        }
    }
}

#[derive(Default)]
struct BenchConfig {
    id: Option<String>,
    group: Option<String>,
    /// `Severity` path tokens
    severity: Option<TokenStream2>,
    threshold: Option<f64>,
    budget_ns: Option<u64>,
    tags: Vec<String>,
    async_runtime: AsyncRuntimeConfig,
    warmup_ns: Option<u64>,
    measurement_ns: Option<u64>,
    samples: Option<u64>,
    sample_count: Option<u64>,
    iters_per_sample: Option<u64>,
    min_iterations: Option<u64>,
    max_iterations: Option<u64>,
    /// Parameterized benchmark args: `args = [10, 100, 1000]`
    args: Vec<syn::Expr>,
    /// Dependencies: `depends_on = ["bench_a", "bench_b"]`
    depends_on: Vec<String>,
}

fn parse_bench_config(args: TokenStream2) -> Result<BenchConfig, syn::Error> {
    let mut config = BenchConfig::default();
    let mut runtime_type: Option<syn::LitStr> = None;
    let mut worker_threads: Option<usize> = None;
    let mut enable_time = true;
    let mut enable_io = true;

    if args.is_empty() {
        return Ok(config);
    }

    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        match name.as_str() {
            "id" => config.id = Some(attr::string(&meta)?),
            "group" => config.group = Some(attr::string(&meta)?),
            "severity" => config.severity = Some(attr::severity(&meta)?),
            "threshold" => config.threshold = Some(attr::non_negative(&meta, "threshold")?),
            "budget" => {
                let budget = attr::duration(&meta)?;
                if budget == 0 {
                    return Err(meta.error("`budget` must be above 0, such as \"50us\""));
                }
                config.budget_ns = Some(budget);
            }
            "runtime" => runtime_type = Some(meta.value()?.parse()?),
            "worker_threads" => worker_threads = Some(attr::worker_threads(&meta)?),
            "enable_time" => enable_time = attr::bool(&meta)?,
            "enable_io" => enable_io = attr::bool(&meta)?,
            "tags" => config.tags = attr::tags(&meta)?,
            "iterations" => {
                return Err(syn::Error::new_spanned(
                    &meta.path,
                    "`iterations` is not supported. Use `iters_per_sample`.",
                ));
            }
            "warmup" => config.warmup_ns = Some(attr::duration(&meta)?),
            "measurement" => config.measurement_ns = Some(attr::duration(&meta)?),
            "samples" => config.samples = attr::int(&meta)?,
            "sample_count" => config.sample_count = attr::int(&meta)?,
            "iters_per_sample" => config.iters_per_sample = attr::int(&meta)?,
            "min_iterations" => config.min_iterations = attr::int(&meta)?,
            "max_iterations" => config.max_iterations = attr::int(&meta)?,
            "args" => config.args = attr::expr_array(&meta)?,
            "depends_on" => config.depends_on = attr::string_array(&meta)?,
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });

    syn::parse::Parser::parse2(parser, args)?;

    config.async_runtime = match runtime_type.as_ref().map(|lit| (lit, lit.value())) {
        Some((_, ref kind)) if kind == "multi_thread" || kind == "multi-thread" => {
            AsyncRuntimeConfig::MultiThread {
                worker_threads,
                enable_time,
                enable_io,
            }
        }
        Some((lit, kind)) if kind != "current_thread" && kind != "current-thread" => {
            return Err(syn::Error::new_spanned(
                lit,
                format!("unknown runtime `{kind}`. Use \"current_thread\" or \"multi_thread\"."),
            ));
        }
        _ => {
            if worker_threads.is_some() {
                return Err(syn::Error::new(
                    proc_macro2::Span::call_site(),
                    "`worker_threads` needs `runtime = \"multi_thread\"`",
                ));
            }
            AsyncRuntimeConfig::CurrentThread {
                enable_time,
                enable_io,
            }
        }
    };

    Ok(config)
}

fn validate_signature(func: &ItemFn, has_args: bool) -> syn::Result<()> {
    let expected = if has_args { 2 } else { 1 };
    if func.sig.inputs.len() != expected {
        let msg = if has_args {
            "FluxBench: Parameterized benchmark must take two arguments: `&mut Bencher` and the parameter"
        } else {
            "FluxBench: Function must take exactly one argument: `&mut Bencher`"
        };
        return Err(syn::Error::new_spanned(&func.sig, msg));
    }
    Ok(())
}

/// Convert an expression to a safe, human-readable label for benchmark IDs.
///
/// Collapses whitespace and escapes path-unsafe characters so IDs are valid
/// in file paths, URLs, and don't conflict with the `base_id/label` separator.
///
/// Examples: `10` → `"10"`, `1000` → `"1000"`, `vec![1, 2]` → `"vec![1,2]"`
fn expr_to_label(expr: &syn::Expr) -> String {
    let s = quote!(#expr).to_string();
    // Collapse all whitespace
    let collapsed: String = s.split_whitespace().collect::<Vec<_>>().join("");
    // Escape characters that are unsafe in IDs, paths, or URLs
    collapsed
        .chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '?' | '*' | '"' | '<' | '>' | '|' => '_',
            c => c,
        })
        .collect()
}

/// Extract async runtime builder tokens from config (shared between normal and parameterized paths)
fn async_runtime_tokens(config: &AsyncRuntimeConfig) -> (TokenStream2, TokenStream2) {
    match config {
        AsyncRuntimeConfig::CurrentThread {
            enable_time,
            enable_io,
        } => {
            let time = if *enable_time {
                quote! { .enable_time() }
            } else {
                quote! {}
            };
            let io = if *enable_io {
                quote! { .enable_io() }
            } else {
                quote! {}
            };
            (quote! { new_current_thread() }, quote! { #time #io })
        }
        AsyncRuntimeConfig::MultiThread {
            worker_threads,
            enable_time,
            enable_io,
        } => {
            let workers = worker_threads
                .map(|n| quote! { .worker_threads(#n) })
                .unwrap_or(quote! {});
            let time = if *enable_time {
                quote! { .enable_time() }
            } else {
                quote! {}
            };
            let io = if *enable_io {
                quote! { .enable_io() }
            } else {
                quote! {}
            };
            (quote! { new_multi_thread() }, quote! { #workers #time #io })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_err(tokens: TokenStream2) -> String {
        parse_bench_config(tokens)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    }

    #[test]
    fn bench_config_parses_sample_count_and_iters_per_sample() {
        let config = parse_bench_config(quote!(sample_count = 250, iters_per_sample = 10)).unwrap();
        assert_eq!(config.sample_count, Some(250));
        assert_eq!(config.iters_per_sample, Some(10));
        assert_eq!(config.samples, None);
    }

    #[test]
    fn bench_config_rejects_iterations() {
        assert_eq!(
            parse_err(quote!(iterations = 5)),
            "`iterations` is not supported. Use `iters_per_sample`."
        );
    }

    #[test]
    fn bench_config_rejects_zero_worker_threads() {
        assert_eq!(
            parse_err(quote!(runtime = "multi_thread", worker_threads = 0)),
            "`worker_threads` must be at least 1"
        );
    }

    #[test]
    fn bench_config_accepts_one_worker_thread() {
        let config =
            parse_bench_config(quote!(runtime = "multi_thread", worker_threads = 1)).unwrap();
        assert!(matches!(
            config.async_runtime,
            AsyncRuntimeConfig::MultiThread {
                worker_threads: Some(1),
                ..
            }
        ));
    }

    #[test]
    fn bench_config_rejects_unknown_runtime_and_stray_worker_threads() {
        assert!(parse_err(quote!(runtime = "tokio")).contains("unknown runtime `tokio`"));
        assert!(parse_err(quote!(worker_threads = 2)).contains("needs `runtime"));
        assert!(
            parse_err(quote!(runtime = "current_thread", worker_threads = 2))
                .contains("needs `runtime")
        );
    }

    #[test]
    fn bench_config_parses_budget() {
        let config = parse_bench_config(quote!(budget = "50us")).unwrap();
        assert_eq!(config.budget_ns, Some(50_000));
        let config = parse_bench_config(quote!(budget = "1.5ms")).unwrap();
        assert_eq!(config.budget_ns, Some(1_500_000));
    }

    #[test]
    fn bench_config_rejects_bad_budget() {
        assert!(parse_err(quote!(budget = "fast")).contains("invalid duration \"fast\""));
        assert!(parse_err(quote!(budget = "100")).contains("it has no unit"));
        assert!(parse_err(quote!(budget = "0ns")).contains("must be above 0"));
    }

    #[test]
    fn bench_config_rejects_bad_warmup_and_measurement() {
        assert!(parse_err(quote!(warmup = "5")).contains("invalid duration"));
        assert!(parse_err(quote!(measurement = "soon")).contains("invalid duration"));
    }

    #[test]
    fn bench_config_rejects_unknown_severity() {
        assert!(parse_err(quote!(severity = "fatal")).contains("unknown severity `fatal`"));
    }

    #[test]
    fn bench_config_accepts_integer_threshold() {
        let config = parse_bench_config(quote!(threshold = 5)).unwrap();
        assert_eq!(config.threshold, Some(5.0));
    }

    #[test]
    fn async_runner_drives_body_without_block_on_or_expect() {
        let config = AsyncRuntimeConfig::default();
        let tokens = generate_async_runner(&config, quote! { bench(bencher) }).to_string();
        assert!(tokens.contains("drive_async_body"));
        assert!(tokens.contains("attach_runtime"));
        assert!(tokens.contains("attach_runtime_error"));
        assert!(tokens.contains("new_current_thread"));
        assert!(!tokens.contains("block_on"));
        assert!(!tokens.contains("expect"));
    }
}
