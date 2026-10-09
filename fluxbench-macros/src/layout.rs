//! `#[flux::group]`, `#[flux::report]`, and `#[flux::chart]`: HTML report structure.

use crate::attr;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::ItemStruct;

pub fn group_impl(args: TokenStream2, input: ItemStruct) -> Result<TokenStream2, syn::Error> {
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    let mut id = struct_name_str.clone();
    let mut description = String::new();
    let mut tags: Vec<String> = Vec::new();
    let mut parent: Option<String> = None;

    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        match name.as_str() {
            "id" => id = attr::string(&meta)?,
            "description" | "desc" => description = attr::string(&meta)?,
            "tags" => tags = attr::tags(&meta)?,
            "parent" => parent = Some(attr::string(&meta)?),
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });

    syn::parse::Parser::parse2(parser, args)?;

    if parent.as_deref() == Some(id.as_str()) {
        return Err(syn::Error::new_spanned(
            &input,
            format!("group `{id}` cannot be its own parent"),
        ));
    }

    let tags_tokens: Vec<_> = tags.iter().map(|t| quote! { #t }).collect();
    let parent_expr = attr::option(parent);

    Ok(quote! {
        #input

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::GroupDef {
                id: #id,
                description: #description,
                tags: &[#(#tags_tokens),*],
                parent: #parent_expr,
            }
        }
    })
}

pub fn report_impl(args: TokenStream2, input: ItemStruct) -> Result<TokenStream2, syn::Error> {
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    let mut title = struct_name_str.clone();
    let mut description = String::new();
    let mut layout = (2u32, 2u32);

    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        match name.as_str() {
            "title" => title = attr::string(&meta)?,
            "description" | "desc" => description = attr::string(&meta)?,
            "layout" => {
                let value: syn::LitStr = meta.value()?.parse()?;
                layout = parse_grid_layout(&value.value()).ok_or_else(|| {
                    syn::Error::new_spanned(
                        &value,
                        format!(
                            "invalid layout \"{}\". Use \"grid(rows, cols)\" or \"RxC\" with values >= 1, such as \"grid(2, 2)\".",
                            value.value()
                        ),
                    )
                })?;
            }
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });

    syn::parse::Parser::parse2(parser, args)?;

    let (rows, cols) = layout;

    Ok(quote! {
        #input

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::ReportDef {
                title: #title,
                description: #description,
                layout: (#rows, #cols),
            }
        }
    })
}

/// Parse a grid layout such as "grid(2, 2)" or "2x2". Rows and cols must be >= 1.
fn parse_grid_layout(s: &str) -> Option<(u32, u32)> {
    let s = s.trim();
    let (rows, cols) =
        if let Some(inner) = s.strip_prefix("grid(").and_then(|s| s.strip_suffix(')')) {
            inner.split_once(',')?
        } else {
            s.split_once('x')?
        };
    let rows: u32 = rows.trim().parse().ok()?;
    let cols: u32 = cols.trim().parse().ok()?;
    (rows >= 1 && cols >= 1).then_some((rows, cols))
}

pub fn chart_impl(args: TokenStream2, input: ItemStruct) -> Result<TokenStream2, syn::Error> {
    let mut title = input.ident.to_string();
    let mut kind = quote! { ::fluxbench::ChartType::Bar };
    let mut metric = "mean".to_string();
    let mut benchmarks: Vec<String> = Vec::new();
    let mut target: Option<f64> = None;
    let mut row: Option<u32> = None;
    let mut col: Option<u32> = None;

    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        match name.as_str() {
            "title" => title = attr::string(&meta)?,
            "kind" => {
                let value: syn::LitStr = meta.value()?.parse()?;
                kind = match value.value().as_str() {
                    "bar" => quote! { ::fluxbench::ChartType::Bar },
                    "line" => quote! { ::fluxbench::ChartType::Line },
                    other => {
                        return Err(syn::Error::new_spanned(
                            &value,
                            format!("unknown chart kind `{other}`. Use \"bar\" or \"line\"."),
                        ));
                    }
                };
            }
            "metric" => metric = attr::string(&meta)?,
            "benchmarks" => benchmarks = attr::string_array(&meta)?,
            "target" => {
                let value = attr::number(&meta)?;
                if !value.is_finite() {
                    return Err(meta.error("`target` must be a finite number"));
                }
                target = Some(value);
            }
            "row" => row = Some(grid_index(&meta, "row")?),
            "col" => col = Some(grid_index(&meta, "col")?),
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });

    syn::parse::Parser::parse2(parser, args)?;

    if benchmarks.is_empty() {
        return Err(syn::Error::new_spanned(
            &input,
            "chart needs `benchmarks`, such as `benchmarks = [\"bench_a\", \"bench_b\"]`",
        ));
    }
    let position = match (row, col) {
        (Some(r), Some(c)) => quote! { Some((#r, #c)) },
        (None, None) => quote! { None },
        _ => {
            return Err(syn::Error::new_spanned(
                &input,
                "chart needs both `row` and `col`, or neither",
            ));
        }
    };
    let target_expr = attr::option(target);

    Ok(quote! {
        #input

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::ChartDef {
                title: #title,
                chart_type: #kind,
                metric: #metric,
                benchmarks: &[#(#benchmarks),*],
                target_line: #target_expr,
                position: #position,
            }
        }
    })
}

/// Parse a 1-based grid index.
fn grid_index(meta: &syn::meta::ParseNestedMeta, key: &str) -> syn::Result<u32> {
    let value: Option<u32> = attr::int(meta)?;
    match value {
        Some(v) if v >= 1 => Ok(v),
        _ => Err(meta.error(format!("`{key}` must be an integer >= 1"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item() -> ItemStruct {
        syn::parse_quote!(
            struct Def;
        )
    }

    fn err(result: Result<TokenStream2, syn::Error>) -> String {
        result.err().map(|e| e.to_string()).unwrap_or_default()
    }

    #[test]
    fn grid_layout_forms() {
        assert_eq!(parse_grid_layout("grid(2, 3)"), Some((2, 3)));
        assert_eq!(parse_grid_layout("1x4"), Some((1, 4)));
        assert_eq!(parse_grid_layout("grid(0, 2)"), None);
        assert_eq!(parse_grid_layout("wide"), None);
    }

    #[test]
    fn report_rejects_bad_layout() {
        let e = err(report_impl(quote!(layout = "wide"), item()));
        assert!(e.contains("invalid layout \"wide\""), "{e}");
    }

    #[test]
    fn report_carries_description() {
        let tokens = report_impl(quote!(title = "T", description = "D"), item())
            .unwrap()
            .to_string();
        assert!(tokens.contains("description : \"D\""), "{tokens}");
    }

    #[test]
    fn chart_parses_every_key() {
        let tokens = chart_impl(
            quote!(
                title = "Latency",
                kind = "line",
                metric = "p99",
                benchmarks = ["a", "b"],
                target = 500,
                row = 1,
                col = 2
            ),
            item(),
        )
        .unwrap()
        .to_string();
        assert!(tokens.contains("ChartType :: Line"), "{tokens}");
        assert!(tokens.contains("metric : \"p99\""), "{tokens}");
        assert!(
            tokens.contains("1u32") && tokens.contains("2u32"),
            "{tokens}"
        );
    }

    #[test]
    fn chart_rejects_bad_input() {
        assert!(err(chart_impl(quote!(title = "x"), item())).contains("needs `benchmarks`"));
        assert!(
            err(chart_impl(quote!(benchmarks = ["a"], kind = "pie"), item()))
                .contains("unknown chart kind `pie`")
        );
        assert!(err(chart_impl(quote!(benchmarks = ["a"], row = 1), item())).contains("both"));
        assert!(
            err(chart_impl(
                quote!(benchmarks = ["a"], row = 0, col = 1),
                item()
            ))
            .contains(">= 1")
        );
    }

    #[test]
    fn group_rejects_self_parent() {
        let e = err(group_impl(quote!(id = "g", parent = "g"), item()));
        assert!(e.contains("cannot be its own parent"), "{e}");
    }
}
