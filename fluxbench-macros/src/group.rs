//! `#[flux::group]`: group description, tags, parent, and run config.

use crate::attr;
use crate::throughput::{self, ThroughputAttr};
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::ItemStruct;

/// Run config keys a group sets for its benchmarks and child groups.
#[derive(Debug, Default, PartialEq, Eq)]
struct GroupRunConfig {
    warmup_ns: Option<u64>,
    measurement_ns: Option<u64>,
    sample_count: Option<u64>,
    iters_per_sample: Option<u64>,
    min_iterations: Option<u64>,
    max_iterations: Option<u64>,
    throughput: Option<ThroughputAttr>,
}

impl GroupRunConfig {
    /// `::fluxbench::GroupConfig { .. }` tokens.
    fn tokens(&self) -> TokenStream2 {
        let warmup_ns = attr::option(self.warmup_ns);
        let measurement_ns = attr::option(self.measurement_ns);
        let sample_count = attr::option(self.sample_count);
        let iters_per_sample = attr::option(self.iters_per_sample);
        let min_iterations = attr::option(self.min_iterations);
        let max_iterations = attr::option(self.max_iterations);
        let throughput = throughput::tokens(self.throughput);
        quote! {
            ::fluxbench::GroupConfig {
                warmup_ns: #warmup_ns,
                measurement_ns: #measurement_ns,
                sample_count: #sample_count,
                iters_per_sample: #iters_per_sample,
                min_iterations: #min_iterations,
                max_iterations: #max_iterations,
                throughput: #throughput,
            }
        }
    }
}

/// Parsed `#[flux::group]` arguments.
#[derive(Debug, Default)]
struct GroupArgs {
    id: Option<String>,
    description: String,
    tags: Vec<String>,
    parent: Option<String>,
    run: GroupRunConfig,
}

fn parse_group_args(args: TokenStream2) -> syn::Result<GroupArgs> {
    let mut parsed = GroupArgs::default();
    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        let run = &mut parsed.run;
        match name.as_str() {
            "id" => parsed.id = Some(attr::string(&meta)?),
            "description" | "desc" => parsed.description = attr::string(&meta)?,
            "tags" => parsed.tags = attr::tags(&meta)?,
            "parent" => parsed.parent = Some(attr::string(&meta)?),
            "warmup" => run.warmup_ns = Some(attr::duration(&meta)?),
            "measurement" => run.measurement_ns = Some(attr::duration(&meta)?),
            "sample_count" => run.sample_count = attr::int(&meta)?,
            "iters_per_sample" => run.iters_per_sample = attr::int(&meta)?,
            "min_iterations" => run.min_iterations = attr::int(&meta)?,
            "max_iterations" => run.max_iterations = attr::int(&meta)?,
            key if throughput::is_key(key) => throughput::parse(&meta, key, &mut run.throughput)?,
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });
    syn::parse::Parser::parse2(parser, args)?;
    Ok(parsed)
}

pub fn group_impl(args: TokenStream2, input: ItemStruct) -> Result<TokenStream2, syn::Error> {
    let parsed = parse_group_args(args)?;
    let id = parsed.id.unwrap_or_else(|| input.ident.to_string());

    if parsed.parent.as_deref() == Some(id.as_str()) {
        return Err(syn::Error::new_spanned(
            &input,
            format!("group `{id}` cannot be its own parent"),
        ));
    }

    let description = parsed.description;
    let tags_tokens: Vec<_> = parsed.tags.iter().map(|t| quote! { #t }).collect();
    let parent_expr = attr::option(parsed.parent);
    let config = parsed.run.tokens();

    Ok(quote! {
        #input

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::GroupDef {
                id: #id,
                description: #description,
                tags: &[#(#tags_tokens),*],
                parent: #parent_expr,
                config: #config,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item() -> ItemStruct {
        syn::parse_quote!(
            struct Def;
        )
    }

    fn err(args: TokenStream2) -> String {
        group_impl(args, item())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    }

    #[test]
    fn group_rejects_self_parent() {
        let e = err(quote!(id = "g", parent = "g"));
        assert!(e.contains("cannot be its own parent"), "{e}");
    }

    #[test]
    fn group_parses_run_config() {
        let parsed = parse_group_args(quote!(
            id = "io",
            sample_count = 30,
            measurement = "2s",
            warmup = "500ms",
            iters_per_sample = 4,
            min_iterations = 10,
            max_iterations = 1000,
            throughput_bytes = 4096
        ))
        .unwrap();
        assert_eq!(
            parsed.run,
            GroupRunConfig {
                warmup_ns: Some(500_000_000),
                measurement_ns: Some(2_000_000_000),
                sample_count: Some(30),
                iters_per_sample: Some(4),
                min_iterations: Some(10),
                max_iterations: Some(1000),
                throughput: Some(ThroughputAttr::Bytes(4096)),
            }
        );
    }

    #[test]
    fn group_without_run_keys_sets_nothing() {
        let tokens = group_impl(quote!(id = "g"), item()).unwrap().to_string();
        assert!(
            tokens.contains("config : :: fluxbench :: GroupConfig"),
            "{tokens}"
        );
        assert!(tokens.contains("warmup_ns : None"), "{tokens}");
        assert!(tokens.contains("throughput : None"), "{tokens}");
    }

    #[test]
    fn group_uses_the_strict_duration_parser() {
        assert!(err(quote!(warmup = "5")).contains("invalid duration"));
        assert!(err(quote!(measurement = "soon")).contains("invalid duration"));
    }

    #[test]
    fn group_rejects_both_throughput_keys() {
        let e = err(quote!(throughput_bytes = 1, throughput_elements = 1));
        assert!(e.contains("mutually exclusive"), "{e}");
    }

    #[test]
    fn group_rejects_unknown_key() {
        assert!(err(quote!(samples = 5)).contains("unknown attribute: samples"));
    }
}
