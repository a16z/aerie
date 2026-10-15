//! Small-integer kernels of the forward pass.
//!
//! [`dot_rows`] is the first-core kernel: a row-major `i16` matrix against a
//! few `i16` source vectors at once. It is written so the compiler lowers the
//! lane loop to widening multiply-accumulate (`smlal`/`smlal2` on NEON,
//! `pmaddwd` on x86) without `unsafe`: every accumulator is an `[i32; LANES]`
//! array and the row-by-source register tile is a const generic.
//!
//! Exactness: callers bound `|matrix entry| <= 12` and `|source| <= 2^15`,
//! so one product is below `2^19`, and each `i32` lane sums `width / LANES`
//! products. [`MAX_WIDTH`] keeps that below `2^31`.

/// `i16` lanes per accumulator; two NEON or one AVX2 register.
const LANES: usize = 8;
/// Largest supported row width: `(MAX_WIDTH / LANES) * 12 * 2^15 < 2^31`.
pub(crate) const MAX_WIDTH: usize = 1 << 14;
/// Rows per register tile.
const TILE_ROWS: usize = 4;

/// Dot products of `R` rows against `S` sources, all of the same length; a
/// tail shorter than [`LANES`] is summed directly.
#[inline]
fn tile<const R: usize, const S: usize>(rows: [&[i16]; R], sources: [&[i16]; S]) -> [[i64; S]; R] {
    let split = rows[0].len() / LANES * LANES;
    let heads: [&[[i16; LANES]]; R] = rows.map(|row| row[..split].as_chunks::<LANES>().0);
    let source_heads: [&[[i16; LANES]]; S] =
        sources.map(|source| source[..split].as_chunks::<LANES>().0);
    let chunks = split / LANES;
    let mut acc = [[[0_i32; LANES]; S]; R];
    for chunk in 0..chunks {
        for s in 0..S {
            let rhs = source_heads[s][chunk];
            for r in 0..R {
                let lhs = heads[r][chunk];
                for lane in 0..LANES {
                    acc[r][s][lane] += i32::from(lhs[lane]) * i32::from(rhs[lane]);
                }
            }
        }
    }
    let mut out = acc.map(|row| row.map(|lanes| lanes.iter().map(|&lane| i64::from(lane)).sum()));
    for (r, row) in rows.iter().enumerate() {
        for (s, source) in sources.iter().enumerate() {
            out[r][s] += row[split..]
                .iter()
                .zip(&source[split..])
                .map(|(&a, &b)| i64::from(a) * i64::from(b))
                .sum::<i64>();
        }
    }
    out
}

/// `out[s * out_stride + r] = sum_k matrix[r * width + k] * sources[s][k]`
/// for every row `r` of `matrix` and every source `s`, exactly.
///
/// `width` must be at most [`MAX_WIDTH`]; entries of `matrix` must have
/// magnitude at most 12.
pub(crate) fn dot_rows(
    matrix: &[i16],
    width: usize,
    sources: &[&[i16]],
    out: &mut [i64],
    out_stride: usize,
) {
    assert!(width <= MAX_WIDTH);
    debug_assert!(matrix.len().is_multiple_of(width));
    debug_assert!(sources.iter().all(|source| source.len() == width));
    let rows = matrix.len() / width;
    let row = |r: usize| &matrix[r * width..(r + 1) * width];
    let mut store = |r: usize, s: usize, value: i64| out[s * out_stride + r] = value;
    let mut s = 0;
    while s < sources.len() {
        let pair = sources.len() - s >= 2;
        let mut r = 0;
        while r < rows {
            if rows - r >= TILE_ROWS {
                let block: [&[i16]; TILE_ROWS] = core::array::from_fn(|i| row(r + i));
                if pair {
                    let dots = tile(block, [sources[s], sources[s + 1]]);
                    for (i, dots) in dots.iter().enumerate() {
                        store(r + i, s, dots[0]);
                        store(r + i, s + 1, dots[1]);
                    }
                } else {
                    let dots = tile(block, [sources[s]]);
                    for (i, dots) in dots.iter().enumerate() {
                        store(r + i, s, dots[0]);
                    }
                }
                r += TILE_ROWS;
            } else {
                if pair {
                    let [dots] = tile([row(r)], [sources[s], sources[s + 1]]);
                    store(r, s, dots[0]);
                    store(r, s + 1, dots[1]);
                } else {
                    let [[dot]] = tile([row(r)], [sources[s]]);
                    store(r, s, dot);
                }
                r += 1;
            }
        }
        s += if pair { 2 } else { 1 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference(matrix: &[i16], width: usize, source: &[i16], row: usize) -> i64 {
        matrix[row * width..(row + 1) * width]
            .iter()
            .zip(source)
            .map(|(&a, &b)| i64::from(a) * i64::from(b))
            .sum()
    }

    #[test]
    fn projection_kernel_dot_rows_matches_reference_at_extremes() {
        for (width, rows) in [
            (2048, 1),
            (2048, 3),
            (2048, 4),
            (2048, 7),
            (2048, 31),
            (2048, 43),
            (43, 5),
            (93, 43),
            (7, 2),
        ] {
            let matrix: Vec<i16> = (0..rows * width)
                .map(|index| [12, -12, 0, 1, -1, 7][(index * 7 + index / 5) % 6])
                .collect();
            let sources: Vec<Vec<i16>> = (0..5)
                .map(|s| {
                    (0..width)
                        .map(|k| match (k + s) % 4 {
                            0 => i16::MAX,
                            1 => i16::MIN,
                            2 => (k as i16).wrapping_mul(31),
                            _ => -(k as i16),
                        })
                        .collect()
                })
                .collect();
            // Worst case: every product has the same sign and magnitude 12 * 2^15.
            let mut worst_matrix = vec![-12_i16; rows * width];
            worst_matrix[0] = 12;
            let worst_source = vec![i16::MIN; width];
            for (matrix, sources) in [
                (
                    &matrix,
                    sources.iter().map(Vec::as_slice).collect::<Vec<_>>(),
                ),
                (&worst_matrix, vec![worst_source.as_slice(); 3]),
            ] {
                let mut out = vec![0; sources.len() * rows];
                dot_rows(matrix, width, &sources, &mut out, rows);
                for (s, source) in sources.iter().enumerate() {
                    for row in 0..rows {
                        assert_eq!(out[s * rows + row], reference(matrix, width, source, row));
                    }
                }
            }
        }
    }
}
