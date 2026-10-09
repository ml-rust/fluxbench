//! `throughput_bytes = N` and `throughput_elements = N` on `#[flux::bench]` and `#[flux::group]`.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::meta::ParseNestedMeta;

/// Parsed throughput attribute: work per call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThroughputAttr {
    /// `throughput_bytes = N`
    Bytes(u64),
    /// `throughput_elements = N`
    Elements(u64),
}

/// `true` for the attribute keys this module parses.
pub fn is_key(name: &str) -> bool {
    matches!(name, "throughput_bytes" | "throughput_elements")
}

/// Parse `key = N` into `slot`. N must be above 0.
///
/// A second throughput key on the same item is an error: the two keys are mutually exclusive.
pub fn parse(
    meta: &ParseNestedMeta,
    key: &str,
    slot: &mut Option<ThroughputAttr>,
) -> syn::Result<()> {
    let value: syn::LitInt = meta.value()?.parse()?;
    let amount: u64 = value.base10_parse()?;
    if amount == 0 {
        return Err(syn::Error::new_spanned(
            &value,
            format!("`{key}` must be above 0"),
        ));
    }
    if slot.is_some() {
        return Err(meta.error(
            "`throughput_bytes` and `throughput_elements` are mutually exclusive. Set one of them once.",
        ));
    }
    *slot = Some(if key == "throughput_bytes" {
        ThroughputAttr::Bytes(amount)
    } else {
        ThroughputAttr::Elements(amount)
    });
    Ok(())
}

/// `Some(::fluxbench::Throughput::..)` or `None` tokens.
pub fn tokens(value: Option<ThroughputAttr>) -> TokenStream2 {
    match value {
        Some(ThroughputAttr::Bytes(n)) => quote! { Some(::fluxbench::Throughput::Bytes(#n)) },
        Some(ThroughputAttr::Elements(n)) => {
            quote! { Some(::fluxbench::Throughput::Elements(#n)) }
        }
        None => quote! { None },
    }
}
