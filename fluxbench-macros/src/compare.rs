//! `#[flux::compare]`: comparison tables and series.

use crate::attr;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::ItemStruct;

/// Metrics a comparison can use. Matches `get_metric_value` in fluxbench-cli.
const COMPARE_METRICS: &[&str] = &[
    "mean",
    "median",
    "min",
    "max",
    "p50",
    "p90",
    "p95",
    "p99",
    "p999",
    "throughput_bytes",
    "throughput_elements",
];

pub fn compare_impl(args: TokenStream2, input: ItemStruct) -> Result<TokenStream2, syn::Error> {
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    let mut id = struct_name_str.clone();
    let mut title = struct_name_str.clone();
    let mut benchmarks: Vec<String> = Vec::new();
    let mut baseline: Option<String> = None;
    let mut metric = "mean".to_string();
    let mut group: Option<String> = None;
    let mut x_value: Option<String> = None;
    let mut series: Option<Vec<String>> = None;

    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        match name.as_str() {
            "id" => id = attr::string(&meta)?,
            "title" => title = attr::string(&meta)?,
            "benchmarks" => benchmarks = attr::string_array(&meta)?,
            "baseline" => baseline = Some(attr::string(&meta)?),
            "metric" => {
                let value: syn::LitStr = meta.value()?.parse()?;
                if !COMPARE_METRICS.contains(&value.value().as_str()) {
                    return Err(syn::Error::new_spanned(
                        &value,
                        format!(
                            "unknown compare metric `{}`. Use one of: {}.",
                            value.value(),
                            COMPARE_METRICS.join(", ")
                        ),
                    ));
                }
                metric = value.value();
            }
            "group" => group = Some(attr::string(&meta)?),
            "x" => x_value = Some(attr::string(&meta)?),
            "series" => series = Some(attr::string_array(&meta)?),
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });

    syn::parse::Parser::parse2(parser, args)?;

    if benchmarks.is_empty() {
        return Err(syn::Error::new_spanned(
            &input,
            "compare requires at least one benchmark in 'benchmarks' list",
        ));
    }

    let baseline_expr = attr::option(baseline);
    let group_expr = attr::option(group);
    let x_expr = attr::option(x_value);
    let series_expr = match &series {
        Some(s) => quote! { Some(&[#(#s),*]) },
        None => quote! { None },
    };

    Ok(quote! {
        #input

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::CompareDef {
                id: #id,
                title: #title,
                benchmarks: &[#(#benchmarks),*],
                baseline: #baseline_expr,
                metric: #metric,
                group: #group_expr,
                x: #x_expr,
                series: #series_expr,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compare_rejects_unknown_metric() {
        let item: ItemStruct = syn::parse_quote!(
            struct Cmp;
        );
        let err = compare_impl(quote!(benchmarks = ["a"], metric = "fastest"), item)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default();
        assert!(err.contains("unknown compare metric `fastest`"), "{err}");
    }

    #[test]
    fn compare_accepts_work_rate_metrics() {
        for metric in ["throughput_bytes", "throughput_elements"] {
            let item: ItemStruct = syn::parse_quote!(
                struct Cmp;
            );
            let args = quote!(benchmarks = ["a", "b"], metric = #metric);
            assert!(compare_impl(args, item).is_ok(), "{metric}");
        }
    }
}
