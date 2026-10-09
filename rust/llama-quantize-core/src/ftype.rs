use crate::cnum::stoi;

#[derive(Debug, PartialEq)]
pub struct QuantOption {
    pub name: &'static str,
    pub ftype: i32,
    pub desc: &'static str,
}

const fn q(name: &'static str, ftype: i32, desc: &'static str) -> QuantOption {
    QuantOption { name, ftype, desc }
}

pub const ALL_F32: i32 = 0;

pub const QUANT_OPTIONS: &[QuantOption] = &[
    q("Q1_0", 40, " 1.125 bpw quantization"),
    q("Q2_0", 41, " 2.25 bpw quantization (group 64)"),
    q("Q4_0", 2, " 4.34G, +0.4685 ppl @ Llama-3-8B"),
    q("Q4_1", 3, " 4.78G, +0.4511 ppl @ Llama-3-8B"),
    q("MXFP4_MOE", 38, " MXFP4 MoE"),
    q("Q5_0", 8, " 5.21G, +0.1316 ppl @ Llama-3-8B"),
    q("Q5_1", 9, " 5.65G, +0.1062 ppl @ Llama-3-8B"),
    q("IQ2_XXS", 19, " 2.06 bpw quantization"),
    q("IQ2_XS", 20, " 2.31 bpw quantization"),
    q("IQ2_S", 28, " 2.5  bpw quantization"),
    q("IQ2_M", 29, " 2.7  bpw quantization"),
    q("IQ1_S", 24, " 1.56 bpw quantization"),
    q("IQ1_M", 31, " 1.75 bpw quantization"),
    q("TQ1_0", 36, " 1.69 bpw ternarization"),
    q("TQ2_0", 37, " 2.06 bpw ternarization"),
    q("Q2_K", 10, " 2.96G, +3.5199 ppl @ Llama-3-8B"),
    q("Q2_K_S", 21, " 2.96G, +3.1836 ppl @ Llama-3-8B"),
    q("IQ3_XXS", 23, " 3.06 bpw quantization"),
    q("IQ3_S", 26, " 3.44 bpw quantization"),
    q("IQ3_M", 27, " 3.66 bpw quantization mix"),
    q("Q3_K", 12, "alias for Q3_K_M"),
    q("IQ3_XS", 22, " 3.3 bpw quantization"),
    q("Q3_K_S", 11, " 3.41G, +1.6321 ppl @ Llama-3-8B"),
    q("Q3_K_M", 12, " 3.74G, +0.6569 ppl @ Llama-3-8B"),
    q("Q3_K_L", 13, " 4.03G, +0.5562 ppl @ Llama-3-8B"),
    q("IQ4_NL", 25, " 4.50 bpw non-linear quantization"),
    q("IQ4_XS", 30, " 4.25 bpw non-linear quantization"),
    q("Q4_K", 15, "alias for Q4_K_M"),
    q("Q4_K_S", 14, " 4.37G, +0.2689 ppl @ Llama-3-8B"),
    q("Q4_K_M", 15, " 4.58G, +0.1754 ppl @ Llama-3-8B"),
    q("Q5_K", 17, "alias for Q5_K_M"),
    q("Q5_K_S", 16, " 5.21G, +0.1049 ppl @ Llama-3-8B"),
    q("Q5_K_M", 17, " 5.33G, +0.0569 ppl @ Llama-3-8B"),
    q("Q6_K", 18, " 6.14G, +0.0217 ppl @ Llama-3-8B"),
    q("Q8_0", 7, " 7.96G, +0.0026 ppl @ Llama-3-8B"),
    q("F16", 1, "14.00G, +0.0020 ppl @ Mistral-7B"),
    q("BF16", 32, "14.00G, -0.0050 ppl @ Mistral-7B"),
    q("F32", ALL_F32, "26.00G              @ 7B"),
    q("COPY", ALL_F32, "only copy tensors, no quantizing"),
];

pub fn striequals(a: &[u8], b: &[u8]) -> bool {
    a.eq_ignore_ascii_case(b)
}

pub fn try_parse_ftype(input: &[u8]) -> Option<&'static QuantOption> {
    let upper = input.to_ascii_uppercase();
    if let Some(it) = QUANT_OPTIONS
        .iter()
        .find(|it| striequals(it.name.as_bytes(), &upper))
    {
        return Some(it);
    }
    let value = stoi(&upper)?;
    QUANT_OPTIONS.iter().find(|it| it.ftype == value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ftype_lookup_keeps_table_order() {
        assert_eq!(try_parse_ftype(b"q4_k_m").unwrap().name, "Q4_K_M");
        assert_eq!(try_parse_ftype(b"15").unwrap().name, "Q4_K");
        assert_eq!(try_parse_ftype(b"0").unwrap().name, "F32");
        assert_eq!(try_parse_ftype(b"copy").unwrap().name, "COPY");
        assert_eq!(try_parse_ftype(b" 7x").unwrap().name, "Q8_0");
        assert!(try_parse_ftype(b"4").is_none());
        assert!(try_parse_ftype(b"BAD").is_none());
    }
}
