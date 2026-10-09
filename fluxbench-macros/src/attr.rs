//! Attribute parsing helpers shared by every macro.

use crate::duration::parse_duration;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::meta::ParseNestedMeta;

/// Get the attribute name as a string
pub fn name(meta: &ParseNestedMeta) -> String {
    meta.path
        .get_ident()
        .map(|i| i.to_string())
        .unwrap_or_default()
}

/// Parse a string literal attribute: `attr = "value"`
pub fn string(meta: &ParseNestedMeta) -> syn::Result<String> {
    let value: syn::LitStr = meta.value()?.parse()?;
    Ok(value.value())
}

/// Parse a number attribute: `attr = 1.5` or `attr = 2`.
pub fn number(meta: &ParseNestedMeta) -> syn::Result<f64> {
    let lit: syn::Lit = meta.value()?.parse()?;
    match &lit {
        syn::Lit::Float(f) => f.base10_parse(),
        syn::Lit::Int(i) => i.base10_parse(),
        _ => Err(syn::Error::new_spanned(
            &lit,
            "expected a number, such as `5` or `2.5`",
        )),
    }
}

/// Parse a number that must be finite and at least 0.
pub fn non_negative(meta: &ParseNestedMeta, key: &str) -> syn::Result<f64> {
    let value = number(meta)?;
    if !value.is_finite() || value < 0.0 {
        return Err(meta.error(format!("`{key}` must be a finite number >= 0, got {value}")));
    }
    Ok(value)
}

/// Parse an integer literal attribute: `attr = 42`
pub fn int<T>(meta: &ParseNestedMeta) -> syn::Result<Option<T>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value: syn::LitInt = meta.value()?.parse()?;
    Ok(Some(value.base10_parse()?))
}

/// Parse `worker_threads = N`. N must be at least 1.
pub fn worker_threads(meta: &ParseNestedMeta) -> syn::Result<usize> {
    let value: syn::LitInt = meta.value()?.parse()?;
    let threads: usize = value.base10_parse()?;
    if threads == 0 {
        return Err(syn::Error::new_spanned(
            &value,
            "`worker_threads` must be at least 1",
        ));
    }
    Ok(threads)
}

/// Parse a boolean literal attribute: `attr = true`
pub fn bool(meta: &ParseNestedMeta) -> syn::Result<bool> {
    let value: syn::LitBool = meta.value()?.parse()?;
    Ok(value.value())
}

/// Parse a comma-separated string as tags: `tags = "a, b, c"`
pub fn tags(meta: &ParseNestedMeta) -> syn::Result<Vec<String>> {
    let value: syn::LitStr = meta.value()?.parse()?;
    Ok(value
        .value()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect())
}

/// Parse a bracketed array of strings: `items = ["a", "b", "c"]`
pub fn string_array(meta: &ParseNestedMeta) -> syn::Result<Vec<String>> {
    meta.value()?;
    let content;
    syn::bracketed!(content in meta.input);
    let items: syn::punctuated::Punctuated<syn::LitStr, syn::Token![,]> =
        syn::punctuated::Punctuated::parse_terminated(&content)?;
    Ok(items.iter().map(|s| s.value()).collect())
}

/// Parse a bracketed array of expressions: `args = [10, 100, 1000]`
pub fn expr_array(meta: &ParseNestedMeta) -> syn::Result<Vec<syn::Expr>> {
    meta.value()?;
    let content;
    syn::bracketed!(content in meta.input);
    let items: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]> =
        syn::punctuated::Punctuated::parse_terminated(&content)?;
    Ok(items.into_iter().collect())
}

/// Parse a duration string attribute such as `"50us"` into nanoseconds.
pub fn duration(meta: &ParseNestedMeta) -> syn::Result<u64> {
    let value: syn::LitStr = meta.value()?.parse()?;
    parse_duration(&value.value()).map_err(|e| syn::Error::new_spanned(&value, e))
}

/// Parse `severity = "critical" | "warning" | "info"` into a `Severity` path.
pub fn severity(meta: &ParseNestedMeta) -> syn::Result<TokenStream2> {
    let value: syn::LitStr = meta.value()?.parse()?;
    match value.value().as_str() {
        "critical" => Ok(quote! { ::fluxbench::Severity::Critical }),
        "warning" => Ok(quote! { ::fluxbench::Severity::Warning }),
        "info" => Ok(quote! { ::fluxbench::Severity::Info }),
        other => Err(syn::Error::new_spanned(
            &value,
            format!("unknown severity `{other}`. Use \"critical\", \"warning\", or \"info\"."),
        )),
    }
}

/// Create an unknown attribute error
pub fn unknown(meta: &ParseNestedMeta, name: &str) -> syn::Error {
    meta.error(format!("unknown attribute: {}", name))
}

/// `Some(value)` or `None` tokens for an optional literal.
pub fn option<T: quote::ToTokens>(value: Option<T>) -> TokenStream2 {
    match value {
        Some(v) => quote! { Some(#v) },
        None => quote! { None },
    }
}
