use crate::grisu_table::CACHED_POWERS;

const ALPHA: i32 = -60;
const CACHED_POWERS_MIN_DEC_EXP: i32 = -300;
const CACHED_POWERS_DEC_STEP: i32 = 8;

#[derive(Clone, Copy)]
struct DiyFp {
    f: u64,
    e: i32,
}

impl DiyFp {
    fn sub(x: DiyFp, y: DiyFp) -> DiyFp {
        DiyFp {
            f: x.f.wrapping_sub(y.f),
            e: x.e,
        }
    }

    fn mul(x: DiyFp, y: DiyFp) -> DiyFp {
        let u_lo = x.f & 0xFFFF_FFFF;
        let u_hi = x.f >> 32;
        let v_lo = y.f & 0xFFFF_FFFF;
        let v_hi = y.f >> 32;

        let p0 = u_lo * v_lo;
        let p1 = u_lo * v_hi;
        let p2 = u_hi * v_lo;
        let p3 = u_hi * v_hi;

        let p0_hi = p0 >> 32;
        let p1_lo = p1 & 0xFFFF_FFFF;
        let p1_hi = p1 >> 32;
        let p2_lo = p2 & 0xFFFF_FFFF;
        let p2_hi = p2 >> 32;

        let mut q = p0_hi + p1_lo + p2_lo;
        q += 1u64 << (64 - 32 - 1);

        DiyFp {
            f: p3 + p2_hi + p1_hi + (q >> 32),
            e: x.e + y.e + 64,
        }
    }

    fn normalize(mut x: DiyFp) -> DiyFp {
        while (x.f >> 63) == 0 {
            x.f <<= 1;
            x.e -= 1;
        }
        x
    }

    fn normalize_to(x: DiyFp, target_exponent: i32) -> DiyFp {
        let delta = x.e - target_exponent;
        DiyFp {
            f: x.f << delta,
            e: target_exponent,
        }
    }
}

struct Boundaries {
    w: DiyFp,
    minus: DiyFp,
    plus: DiyFp,
}

fn compute_boundaries(value: f64) -> Boundaries {
    const PRECISION: i32 = 53;
    const BIAS: i32 = 0x3FF + (PRECISION - 1);
    const MIN_EXP: i32 = 1 - BIAS;
    const HIDDEN_BIT: u64 = 1u64 << (PRECISION - 1);

    let bits = value.to_bits();
    let exponent = bits >> (PRECISION - 1);
    let fraction = bits & (HIDDEN_BIT - 1);
    let is_denormal = exponent == 0;
    let v = if is_denormal {
        DiyFp {
            f: fraction,
            e: MIN_EXP,
        }
    } else {
        DiyFp {
            f: fraction + HIDDEN_BIT,
            e: exponent as i32 - BIAS,
        }
    };

    let m_plus = DiyFp {
        f: (2 * v.f) + 1,
        e: v.e - 1,
    };
    let lower_boundary_is_closer = fraction == 0 && exponent > 1;
    let m_minus = if lower_boundary_is_closer {
        DiyFp {
            f: (4 * v.f) - 1,
            e: v.e - 2,
        }
    } else {
        DiyFp {
            f: (2 * v.f) - 1,
            e: v.e - 1,
        }
    };

    let w_plus = DiyFp::normalize(m_plus);
    let w_minus = DiyFp::normalize_to(m_minus, w_plus.e);

    Boundaries {
        w: DiyFp::normalize(v),
        minus: w_minus,
        plus: w_plus,
    }
}

fn cached_power_for_binary_exponent(e: i32) -> &'static crate::grisu_table::CachedPower {
    let f = ALPHA - e - 1;
    let k = ((f * 78913) / (1 << 18)) + i32::from(f > 0);
    let index =
        (-CACHED_POWERS_MIN_DEC_EXP + k + (CACHED_POWERS_DEC_STEP - 1)) / CACHED_POWERS_DEC_STEP;
    &CACHED_POWERS[index as usize]
}

fn find_largest_pow10(n: u32) -> (i32, u32) {
    const TABLE: [(u32, i32); 9] = [
        (1_000_000_000, 10),
        (100_000_000, 9),
        (10_000_000, 8),
        (1_000_000, 7),
        (100_000, 6),
        (10_000, 5),
        (1_000, 4),
        (100, 3),
        (10, 2),
    ];
    for (pow10, k) in TABLE {
        if n >= pow10 {
            return (k, pow10);
        }
    }
    (1, 1)
}

fn grisu2_round(buf: &mut [u8], dist: u64, delta: u64, mut rest: u64, ten_k: u64) {
    let last = buf.len() - 1;
    while rest < dist
        && delta - rest >= ten_k
        && (rest + ten_k < dist || dist - rest > rest + ten_k - dist)
    {
        buf[last] -= 1;
        rest += ten_k;
    }
}

fn grisu2_digit_gen(
    buffer: &mut Vec<u8>,
    decimal_exponent: &mut i32,
    m_minus: DiyFp,
    w: DiyFp,
    m_plus: DiyFp,
) {
    let mut delta = DiyFp::sub(m_plus, m_minus).f;
    let mut dist = DiyFp::sub(m_plus, w).f;

    let one_e = m_plus.e;
    let one_f = 1u64 << -one_e;

    let mut p1 = (m_plus.f >> -one_e) as u32;
    let mut p2 = m_plus.f & (one_f - 1);

    let (k, mut pow10) = find_largest_pow10(p1);

    let mut n = k;
    while n > 0 {
        let d = p1 / pow10;
        let r = p1 % pow10;
        buffer.push(b'0' + d as u8);
        p1 = r;
        n -= 1;

        let rest = (u64::from(p1) << -one_e) + p2;
        if rest <= delta {
            *decimal_exponent += n;
            let ten_n = u64::from(pow10) << -one_e;
            grisu2_round(buffer, dist, delta, rest, ten_n);
            return;
        }

        pow10 /= 10;
    }

    let mut m = 0;
    loop {
        p2 *= 10;
        let d = p2 >> -one_e;
        let r = p2 & (one_f - 1);
        buffer.push(b'0' + d as u8);
        p2 = r;
        m += 1;

        delta *= 10;
        dist *= 10;
        if p2 <= delta {
            break;
        }
    }

    *decimal_exponent -= m;
    grisu2_round(buffer, dist, delta, p2, one_f);
}

pub fn grisu2(value: f64) -> (Vec<u8>, i32) {
    let boundaries = compute_boundaries(value);
    let cached = cached_power_for_binary_exponent(boundaries.plus.e);
    let c_minus_k = DiyFp {
        f: cached.f,
        e: cached.e,
    };

    let w = DiyFp::mul(boundaries.w, c_minus_k);
    let w_minus = DiyFp::mul(boundaries.minus, c_minus_k);
    let w_plus = DiyFp::mul(boundaries.plus, c_minus_k);

    let big_m_minus = DiyFp {
        f: w_minus.f + 1,
        e: w_minus.e,
    };
    let big_m_plus = DiyFp {
        f: w_plus.f - 1,
        e: w_plus.e,
    };

    let mut decimal_exponent = -cached.k;
    let mut buffer = Vec::with_capacity(17);
    grisu2_digit_gen(
        &mut buffer,
        &mut decimal_exponent,
        big_m_minus,
        w,
        big_m_plus,
    );
    (buffer, decimal_exponent)
}
