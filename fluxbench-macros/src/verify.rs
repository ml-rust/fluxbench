//! `#[flux::verify]` and `#[flux::synthetic]`: assertions and computed metrics.

use crate::attr;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::ItemStruct;

pub fn verify_impl(args: TokenStream2, input: ItemStruct) -> Result<TokenStream2, syn::Error> {
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    let mut expr = String::new();
    let mut severity = quote! { ::fluxbench::Severity::Critical };
    let mut margin = 0.0f64;

    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        match name.as_str() {
            "expr" => expr = attr::string(&meta)?,
            "severity" => severity = attr::severity(&meta)?,
            "margin" => margin = attr::non_negative(&meta, "margin")?,
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });

    syn::parse::Parser::parse2(parser, args)?;

    if expr.trim().is_empty() {
        return Err(syn::Error::new_spanned(
            &input,
            "verify needs `expr`, such as `expr = \"bench_a < 5000\"`",
        ));
    }

    Ok(quote! {
        #input

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::VerifyDef {
                id: #struct_name_str,
                expression: #expr,
                severity: #severity,
                margin: #margin,
            }
        }
    })
}

pub fn synthetic_impl(args: TokenStream2, input: ItemStruct) -> Result<TokenStream2, syn::Error> {
    let struct_name = &input.ident;
    let struct_name_str = struct_name.to_string();

    let mut id = struct_name_str.clone();
    let mut formula = String::new();
    let mut unit: Option<String> = None;

    let parser = syn::meta::parser(|meta| {
        let name = attr::name(&meta);
        match name.as_str() {
            "id" => id = attr::string(&meta)?,
            "formula" | "expr" => formula = attr::string(&meta)?,
            "unit" => unit = Some(attr::string(&meta)?),
            _ => return Err(attr::unknown(&meta, &name)),
        }
        Ok(())
    });

    syn::parse::Parser::parse2(parser, args)?;

    if formula.trim().is_empty() {
        return Err(syn::Error::new_spanned(
            &input,
            "synthetic needs `formula`, such as `formula = \"bench_a / bench_b\"`",
        ));
    }

    let unit_expr = attr::option(unit);

    Ok(quote! {
        #input

        ::fluxbench::internal::inventory::submit! {
            ::fluxbench::SyntheticDef {
                id: #id,
                formula: #formula,
                unit: #unit_expr,
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item() -> ItemStruct {
        syn::parse_quote!(
            struct Check;
        )
    }

    fn verify_err(args: TokenStream2) -> String {
        verify_impl(args, item())
            .err()
            .map(|e| e.to_string())
            .unwrap_or_default()
    }

    #[test]
    fn margin_accepts_int_and_float() {
        let tokens = verify_impl(quote!(expr = "a < b", margin = 5), item())
            .unwrap()
            .to_string();
        assert!(tokens.contains("margin : 5f64"), "{tokens}");
        assert!(verify_impl(quote!(expr = "a < b", margin = 2.5), item()).is_ok());
    }

    #[test]
    fn margin_rejects_negative() {
        assert!(verify_impl(quote!(expr = "a < b", margin = -1), item()).is_err());
    }

    #[test]
    fn verify_rejects_unknown_severity_and_missing_expr() {
        assert!(verify_err(quote!(expr = "a < b", severity = "high")).contains("unknown severity"));
        assert!(verify_err(quote!(severity = "info")).contains("needs `expr`"));
    }

    #[test]
    fn verify_rejects_dropped_bench_key() {
        assert!(
            verify_err(quote!(expr = "a < b", bench = "a")).contains("unknown attribute: bench")
        );
    }
}
