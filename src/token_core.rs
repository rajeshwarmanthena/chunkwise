//! Token-aware text chunking using tiktoken BPE encodings.
//!
//! # Design
//!
//! Token counting with BPE encoders is **not** compositional: the token count of
//! `A + B` can differ from `count(A) + count(B)` because BPE merge rules span
//! boundaries. This module therefore **encodes the full candidate chunk** to
//! measure its token length, rather than summing independently computed segment counts.
//!
//! # Tokenizer caching
//!
//! `tiktoken-rs` returns `&'static CoreBPE` singletons initialized once per process.
//! The first call for each encoding (~100 ms) builds internal structures from bundled
//! data; subsequent calls are zero-cost references.
//!
//! # Token boundaries and UTF-8
//!
//! BPE tokens may straddle UTF-8 code-point boundaries when decoded individually.
//! All chunks returned by this module are **substrings of the original source** at
//! verified character (Unicode scalar-value) offsets; no raw token-ID slice is
//! directly decoded and re-emitted without boundary alignment.
//!
//! # Overlap semantics
//!
//! Token overlap is best-effort: when text-boundary constraints (paragraph, line,
//! or word boundaries) prevent placing exactly `chunk_overlap` tokens at the head
//! of the next chunk, the implementation keeps the nearest boundary that does not
//! exceed `chunk_overlap` tokens. Callers must not assume exact token overlap.

use crate::core::{ChunkError, KeepSeparator};
use std::fmt;
use std::str::FromStr;

#[cfg(feature = "tiktoken-rs")]
use tiktoken_rs::{cl100k_base_singleton, o200k_base_singleton, CoreBPE};

// ─── Encoding name ───────────────────────────────────────────────────────────

/// Named encoding supported by this module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// `cl100k_base` — used by `gpt-4`, `gpt-3.5-turbo`, embedding models.
    Cl100kBase,
    /// `o200k_base` — used by `gpt-4o`, `o1`, and later OpenAI models.
    O200kBase,
}

impl fmt::Display for Encoding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Encoding::Cl100kBase => write!(f, "cl100k_base"),
            Encoding::O200kBase => write!(f, "o200k_base"),
        }
    }
}

impl FromStr for Encoding {
    type Err = TokenChunkError;
    fn from_str(s: &str) -> Result<Self, TokenChunkError> {
        match s {
            "cl100k_base" => Ok(Encoding::Cl100kBase),
            "o200k_base" => Ok(Encoding::O200kBase),
            other => Err(TokenChunkError::UnknownEncoding(other.to_string())),
        }
    }
}

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors returned by the token chunking API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenChunkError {
    /// `chunk_size` was zero.
    InvalidSize,
    /// `chunk_overlap >= chunk_size`.
    InvalidOverlap { overlap: usize, size: usize },
    /// Unrecognized encoding name.
    UnknownEncoding(String),
    /// The `tiktoken-rs` feature was not enabled at compile time.
    FeatureNotEnabled,
    /// An internal tokenizer error.
    TokenizerError(String),
}

impl fmt::Display for TokenChunkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenChunkError::InvalidSize => write!(f, "chunk_size must be greater than 0"),
            TokenChunkError::InvalidOverlap { overlap, size } => write!(
                f,
                "chunk_overlap ({overlap}) must be smaller than chunk_size ({size})"
            ),
            TokenChunkError::UnknownEncoding(s) => write!(
                f,
                "unknown encoding '{s}' (supported: 'cl100k_base', 'o200k_base')"
            ),
            TokenChunkError::FeatureNotEnabled => write!(
                f,
                "token chunking requires the 'tiktoken-rs' feature: \
                 rebuild with `cargo build --features tiktoken-rs`"
            ),
            TokenChunkError::TokenizerError(s) => write!(f, "tokenizer error: {s}"),
        }
    }
}

impl std::error::Error for TokenChunkError {}

impl From<ChunkError> for TokenChunkError {
    fn from(e: ChunkError) -> Self {
        match e {
            ChunkError::InvalidSize => TokenChunkError::InvalidSize,
            ChunkError::InvalidOverlap { overlap, size } => {
                TokenChunkError::InvalidOverlap { overlap, size }
            }
            other => TokenChunkError::TokenizerError(other.to_string()),
        }
    }
}

// ─── TokenConfig ─────────────────────────────────────────────────────────────

/// Configuration for token-aware chunking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenConfig {
    /// Maximum number of tokens per chunk (must be > 0).
    pub chunk_size: usize,
    /// Maximum number of tokens of overlap between consecutive chunks (must be < chunk_size).
    pub chunk_overlap: usize,
    /// BPE encoding to use.
    pub encoding: Encoding,
    /// Text splitting strategy.
    pub strategy: TokenStrategy,
    /// Whether to drop whitespace-only chunks (default true).
    pub drop_whitespace: bool,
    /// Where to attach separators for the `Recursive` strategy.
    pub keep_separator: KeepSeparator,
}

impl Default for TokenConfig {
    fn default() -> Self {
        Self {
            chunk_size: 512,
            chunk_overlap: 64,
            encoding: Encoding::Cl100kBase,
            strategy: TokenStrategy::Recursive,
            drop_whitespace: true,
            keep_separator: KeepSeparator::End,
        }
    }
}

impl TokenConfig {
    /// Validate configuration invariants.
    pub fn validate(&self) -> Result<(), TokenChunkError> {
        if self.chunk_size == 0 {
            return Err(TokenChunkError::InvalidSize);
        }
        if self.chunk_overlap >= self.chunk_size {
            return Err(TokenChunkError::InvalidOverlap {
                overlap: self.chunk_overlap,
                size: self.chunk_size,
            });
        }
        Ok(())
    }
}

/// Text splitting strategy used for token chunking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TokenStrategy {
    /// Prefer paragraph (`\n\n`), line (`\n`), and word (` `) boundaries.
    #[default]
    Recursive,
    /// Split at token window limits.
    Fixed,
}

impl fmt::Display for TokenStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenStrategy::Recursive => write!(f, "recursive"),
            TokenStrategy::Fixed => write!(f, "fixed"),
        }
    }
}

impl FromStr for TokenStrategy {
    type Err = TokenChunkError;
    fn from_str(s: &str) -> Result<Self, TokenChunkError> {
        match s {
            "recursive" => Ok(TokenStrategy::Recursive),
            "fixed" => Ok(TokenStrategy::Fixed),
            other => Err(TokenChunkError::UnknownEncoding(format!(
                "unknown token strategy '{other}' (expected 'recursive' or 'fixed')"
            ))),
        }
    }
}

// ─── TokenChunk ──────────────────────────────────────────────────────────────

/// A chunk produced by token-aware splitting.
///
/// `start` and `end` are **character (Unicode scalar value) offsets** into the
/// original source string, guaranteeing `source[chunk.start:chunk.end] == chunk.text`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenChunk {
    /// The chunk text.
    pub text: String,
    /// Inclusive character offset into the source.
    pub start: usize,
    /// Exclusive character offset into the source.
    pub end: usize,
    /// Zero-based position within the chunk list for this document.
    pub index: usize,
    /// Exact token count of this chunk as measured by the configured BPE encoder.
    pub token_count: usize,
}

// ─── Implementation ──────────────────────────────────────────────────────────

#[cfg(feature = "tiktoken-rs")]
mod impl_tokens {
    use super::*;

    const SEPARATORS: [&str; 3] = ["\n\n", "\n", " "];

    pub(super) fn get_bpe(enc: Encoding) -> &'static CoreBPE {
        match enc {
            Encoding::Cl100kBase => cl100k_base_singleton(),
            Encoding::O200kBase => o200k_base_singleton(),
        }
    }

    pub(super) fn count_tokens(bpe: &CoreBPE, text: &str) -> usize {
        bpe.encode_ordinary(text).len()
    }

    pub(super) fn split_into_segments(
        text: &str,
        max_tokens: usize,
        keep_sep: KeepSeparator,
        bpe: &CoreBPE,
    ) -> Vec<(usize, usize)> {
        let mut spans = Vec::new();
        refine_tokens(
            text,
            0,
            text.len(),
            0,
            max_tokens,
            keep_sep,
            bpe,
            &mut spans,
        );
        spans
    }

    #[allow(clippy::too_many_arguments)]
    fn refine_tokens(
        text: &str,
        start: usize,
        end: usize,
        level: usize,
        max_tokens: usize,
        keep_sep: KeepSeparator,
        bpe: &CoreBPE,
        out: &mut Vec<(usize, usize)>,
    ) {
        if start >= end {
            return;
        }
        let region = &text[start..end];
        if count_tokens(bpe, region) <= max_tokens {
            out.push((start, end));
            return;
        }
        if level >= SEPARATORS.len() {
            hard_split_tokens(text, start, end, max_tokens, bpe, out);
            return;
        }

        let sep = SEPARATORS[level];
        let mut piece_start = start;
        let mut pending_sep_start: Option<usize> = None;

        for (i, m) in region.match_indices(sep) {
            let abs_match_start = start + i;
            let abs_match_end = abs_match_start + m.len();

            match keep_sep {
                KeepSeparator::End => {
                    let piece_end = abs_match_end;
                    if piece_end > piece_start {
                        refine_tokens(
                            text,
                            piece_start,
                            piece_end,
                            level + 1,
                            max_tokens,
                            keep_sep,
                            bpe,
                            out,
                        );
                    }
                    piece_start = piece_end;
                }
                KeepSeparator::Start => {
                    let piece_end = abs_match_start;
                    if let Some(ps) = pending_sep_start {
                        if ps < piece_end {
                            refine_tokens(
                                text,
                                ps,
                                piece_end,
                                level + 1,
                                max_tokens,
                                keep_sep,
                                bpe,
                                out,
                            );
                        } else if ps == piece_end {
                            out.push((ps, abs_match_start));
                        }
                    } else if piece_end > piece_start {
                        refine_tokens(
                            text,
                            piece_start,
                            piece_end,
                            level + 1,
                            max_tokens,
                            keep_sep,
                            bpe,
                            out,
                        );
                    }
                    pending_sep_start = Some(abs_match_start);
                    piece_start = abs_match_end;
                }
                KeepSeparator::Discard => {
                    let piece_end = abs_match_start;
                    if piece_end > piece_start {
                        refine_tokens(
                            text,
                            piece_start,
                            piece_end,
                            level + 1,
                            max_tokens,
                            keep_sep,
                            bpe,
                            out,
                        );
                    }
                    piece_start = abs_match_end;
                }
            }
        }

        match keep_sep {
            KeepSeparator::Start => {
                if let Some(ps) = pending_sep_start {
                    if ps < end {
                        refine_tokens(text, ps, end, level + 1, max_tokens, keep_sep, bpe, out);
                    }
                } else if piece_start < end {
                    refine_tokens(
                        text,
                        piece_start,
                        end,
                        level + 1,
                        max_tokens,
                        keep_sep,
                        bpe,
                        out,
                    );
                }
            }
            _ => {
                if piece_start < end {
                    refine_tokens(
                        text,
                        piece_start,
                        end,
                        level + 1,
                        max_tokens,
                        keep_sep,
                        bpe,
                        out,
                    );
                }
            }
        }
    }

    fn hard_split_tokens(
        text: &str,
        start: usize,
        end: usize,
        max_tokens: usize,
        bpe: &CoreBPE,
        out: &mut Vec<(usize, usize)>,
    ) {
        let region = &text[start..end];
        let char_indices: Vec<(usize, char)> = region.char_indices().collect();
        let n_chars = char_indices.len();
        if n_chars == 0 {
            return;
        }

        let mut char_idx = 0;
        while char_idx < n_chars {
            let seg_start_byte = start + char_indices[char_idx].0;
            let one_char_end_byte = if char_idx + 1 < n_chars {
                start + char_indices[char_idx + 1].0
            } else {
                end
            };

            if count_tokens(bpe, &text[seg_start_byte..one_char_end_byte]) > max_tokens {
                out.push((seg_start_byte, one_char_end_byte));
                char_idx += 1;
                continue;
            }

            let mut lo = 1;
            let mut hi = n_chars - char_idx;
            let mut best_count = 1;

            while lo <= hi {
                let mid = lo + (hi - lo) / 2;
                let cand_end_byte = if char_idx + mid < n_chars {
                    start + char_indices[char_idx + mid].0
                } else {
                    end
                };
                if count_tokens(bpe, &text[seg_start_byte..cand_end_byte]) <= max_tokens {
                    best_count = mid;
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }

            let seg_end_byte = if char_idx + best_count < n_chars {
                start + char_indices[char_idx + best_count].0
            } else {
                end
            };
            out.push((seg_start_byte, seg_end_byte));
            char_idx += best_count;
        }
    }

    pub(super) fn pack_segments(
        text: &str,
        segs: &[(usize, usize)],
        max_tokens: usize,
        overlap_tokens: usize,
        bpe: &CoreBPE,
    ) -> Vec<(usize, usize, usize)> {
        let n = segs.len();
        if n == 0 {
            return Vec::new();
        }

        let mut out: Vec<(usize, usize, usize)> = Vec::new();
        let mut i = 0usize;

        while i < n {
            let first_tc = count_tokens(bpe, &text[segs[i].0..segs[i].1]);
            let mut last_good = i;

            if first_tc <= max_tokens {
                let mut j = i + 1;
                while j < n {
                    let cand_tc = count_tokens(bpe, &text[segs[i].0..segs[j].1]);
                    if cand_tc <= max_tokens {
                        last_good = j;
                        j += 1;
                    } else {
                        break;
                    }
                }
            }

            let chunk_start_byte = segs[i].0;
            let chunk_end_byte = segs[last_good].1;
            let chunk_tc = count_tokens(bpe, &text[chunk_start_byte..chunk_end_byte]);
            out.push((chunk_start_byte, chunk_end_byte, chunk_tc));

            if last_good + 1 >= n {
                break;
            }

            let next_unseen = last_good + 1;
            let mut next_start = next_unseen;

            if overlap_tokens > 0 {
                for cand_k in (i + 1..=last_good).rev() {
                    let ol_text = &text[segs[cand_k].0..segs[last_good].1];
                    let ol_tc = count_tokens(bpe, ol_text);
                    if ol_tc > overlap_tokens {
                        break;
                    }

                    let with_next_text = &text[segs[cand_k].0..segs[next_unseen].1];
                    let with_next_tc = count_tokens(bpe, with_next_text);
                    if with_next_tc <= max_tokens {
                        next_start = cand_k;
                    } else {
                        break;
                    }
                }
            }

            i = next_start.max(i + 1);
        }

        out
    }

    pub(super) fn split_fixed(
        text: &str,
        max_tokens: usize,
        overlap: usize,
        bpe: &CoreBPE,
    ) -> Vec<(usize, usize, usize)> {
        let all_tokens = bpe.encode_ordinary(text);
        if all_tokens.is_empty() {
            return Vec::new();
        }
        let n_tokens = all_tokens.len();
        let step = max_tokens.saturating_sub(overlap).max(1);

        let mut byte_offsets: Vec<usize> = Vec::with_capacity(n_tokens + 1);
        let mut pos = 0usize;
        for &tok in &all_tokens {
            byte_offsets.push(pos);
            let len = bpe.decode_bytes(&[tok]).map(|b| b.len()).unwrap_or(0);
            pos += len;
        }
        byte_offsets.push(pos.max(text.len()));

        let mut out: Vec<(usize, usize, usize)> = Vec::new();
        let mut tok_start = 0usize;

        while tok_start < n_tokens {
            let tok_end = (tok_start + max_tokens).min(n_tokens);

            let raw_byte_start = byte_offsets[tok_start].min(text.len());
            let raw_byte_end = byte_offsets[tok_end].min(text.len());

            let safe_start = snap_to_char_start(text, raw_byte_start);
            let mut safe_end = snap_to_char_end(text, raw_byte_end);

            if safe_start < safe_end {
                while safe_end > safe_start
                    && count_tokens(bpe, &text[safe_start..safe_end]) > max_tokens
                {
                    let mut prev_char = safe_end - 1;
                    while prev_char > safe_start && !text.is_char_boundary(prev_char) {
                        prev_char -= 1;
                    }
                    safe_end = prev_char;
                }

                if safe_start < safe_end {
                    let tc = count_tokens(bpe, &text[safe_start..safe_end]);
                    out.push((safe_start, safe_end, tc));
                }
            }

            if tok_end >= n_tokens {
                break;
            }
            tok_start += step;
        }
        out
    }

    pub(super) fn snap_to_char_start(text: &str, pos: usize) -> usize {
        let pos = pos.min(text.len());
        let mut p = pos;
        while p > 0 && !text.is_char_boundary(p) {
            p -= 1;
        }
        p
    }

    pub(super) fn snap_to_char_end(text: &str, pos: usize) -> usize {
        let pos = pos.min(text.len());
        let mut p = pos;
        while p < text.len() && !text.is_char_boundary(p) {
            p += 1;
        }
        p
    }
}

// ─── Char/byte boundary helpers ──────────────────────────────────────────────

#[cfg(feature = "tiktoken-rs")]
struct CharTable {
    table: Option<Vec<usize>>,
}

#[cfg(feature = "tiktoken-rs")]
fn build_char_table(text: &str) -> CharTable {
    if text.is_ascii() {
        return CharTable { table: None };
    }
    let mut t: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
    t.push(text.len());
    CharTable { table: Some(t) }
}

#[cfg(feature = "tiktoken-rs")]
fn byte_to_char(ct: &CharTable, b: usize) -> usize {
    match &ct.table {
        None => b,
        Some(t) => t.partition_point(|&x| x < b),
    }
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Split a single text into token-sized chunks.
pub fn chunk_tokens(text: &str, cfg: &TokenConfig) -> Result<Vec<TokenChunk>, TokenChunkError> {
    cfg.validate()?;

    #[cfg(not(feature = "tiktoken-rs"))]
    {
        let _ = text;
        return Err(TokenChunkError::FeatureNotEnabled);
    }

    #[cfg(feature = "tiktoken-rs")]
    {
        use impl_tokens::*;

        if text.is_empty() {
            return Ok(Vec::new());
        }

        let bpe = get_bpe(cfg.encoding);
        let bounds = build_char_table(text);

        let spans: Vec<(usize, usize, usize)> = match cfg.strategy {
            TokenStrategy::Fixed => split_fixed(text, cfg.chunk_size, cfg.chunk_overlap, bpe),
            TokenStrategy::Recursive => {
                let segs = split_into_segments(text, cfg.chunk_size, cfg.keep_separator, bpe);
                if segs.is_empty() {
                    vec![]
                } else {
                    pack_segments(text, &segs, cfg.chunk_size, cfg.chunk_overlap, bpe)
                }
            }
        };

        let result = spans
            .into_iter()
            .filter(|(s, e, _tc)| {
                s < e && {
                    if cfg.drop_whitespace {
                        !text[*s..*e].trim().is_empty()
                    } else {
                        true
                    }
                }
            })
            .enumerate()
            .map(|(index, (s, e, tc))| {
                let char_start = byte_to_char(&bounds, s);
                let char_end = byte_to_char(&bounds, e);
                TokenChunk {
                    text: text[s..e].to_string(),
                    start: char_start,
                    end: char_end,
                    index,
                    token_count: tc,
                }
            })
            .collect();
        Ok(result)
    }
}

/// Split many texts in parallel using rayon. Results preserve input order.
pub fn chunk_texts_tokens<S: AsRef<str> + Sync>(
    texts: &[S],
    cfg: &TokenConfig,
) -> Result<Vec<Vec<TokenChunk>>, TokenChunkError> {
    cfg.validate()?;

    #[cfg(not(feature = "tiktoken-rs"))]
    {
        let _ = texts;
        return Err(TokenChunkError::FeatureNotEnabled);
    }

    #[cfg(feature = "tiktoken-rs")]
    {
        use rayon::prelude::*;
        texts
            .par_iter()
            .map(|t| chunk_tokens(t.as_ref(), cfg))
            .collect()
    }
}

// ─── Tests (require tiktoken-rs feature) ─────────────────────────────────────

#[cfg(all(test, feature = "tiktoken-rs"))]
mod tests {
    use super::*;

    fn default_cfg() -> TokenConfig {
        TokenConfig {
            chunk_size: 20,
            chunk_overlap: 4,
            encoding: Encoding::Cl100kBase,
            strategy: TokenStrategy::Recursive,
            drop_whitespace: true,
            keep_separator: KeepSeparator::End,
        }
    }

    #[test]
    fn rejects_zero_chunk_size() {
        let mut cfg = default_cfg();
        cfg.chunk_size = 0;
        assert!(matches!(cfg.validate(), Err(TokenChunkError::InvalidSize)));
    }

    #[test]
    fn rejects_overlap_equal_to_size() {
        let mut cfg = default_cfg();
        cfg.chunk_overlap = cfg.chunk_size;
        assert!(matches!(
            cfg.validate(),
            Err(TokenChunkError::InvalidOverlap { .. })
        ));
    }

    #[test]
    fn encoding_roundtrip() {
        assert_eq!(
            "cl100k_base".parse::<Encoding>().unwrap(),
            Encoding::Cl100kBase
        );
        assert_eq!(
            "o200k_base".parse::<Encoding>().unwrap(),
            Encoding::O200kBase
        );
        assert!("unknown".parse::<Encoding>().is_err());
    }

    #[test]
    fn all_chunks_within_token_limit() {
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(50);
        for strategy in [TokenStrategy::Recursive, TokenStrategy::Fixed] {
            let cfg = TokenConfig {
                chunk_size: 30,
                chunk_overlap: 5,
                strategy,
                ..Default::default()
            };
            let chunks = chunk_tokens(&text, &cfg).unwrap();
            assert!(
                !chunks.is_empty(),
                "{strategy}: should produce at least one chunk"
            );
            for (i, c) in chunks.iter().enumerate() {
                assert_eq!(c.index, i, "{strategy}: index must be sequential");
                assert!(
                    c.token_count <= 30,
                    "{strategy}: chunk {i} has {} tokens > 30",
                    c.token_count
                );
            }
        }
    }

    #[test]
    fn offsets_map_back_to_source() {
        let text = "Héllo wörld. 日本語のテキスト。😀 Another sentence here.\n\nNew paragraph. "
            .repeat(10);
        let cfg = TokenConfig {
            chunk_size: 40,
            chunk_overlap: 8,
            ..Default::default()
        };
        let chars: Vec<char> = text.chars().collect();
        let chunks = chunk_tokens(&text, &cfg).unwrap();
        assert!(!chunks.is_empty(), "should produce chunks");
        for c in &chunks {
            assert!(c.start <= c.end, "start must be <= end");
            assert!(c.end <= chars.len(), "end must not exceed source length");
            let slice: String = chars[c.start..c.end].iter().collect();
            assert_eq!(
                slice, c.text,
                "token chunk offset invariant: source[{}..{}] must equal chunk.text",
                c.start, c.end
            );
        }
    }

    #[test]
    fn empty_input_produces_no_chunks() {
        let cfg = default_cfg();
        assert!(chunk_tokens("", &cfg).unwrap().is_empty());
    }

    #[test]
    fn drop_whitespace_true_drops_whitespace_only() {
        let cfg = TokenConfig {
            chunk_size: 5,
            chunk_overlap: 0,
            drop_whitespace: true,
            ..Default::default()
        };
        let text = "   \n\n   ";
        let chunks = chunk_tokens(text, &cfg).unwrap();
        for c in &chunks {
            assert!(
                !c.text.trim().is_empty(),
                "drop_whitespace=true: got whitespace-only chunk: {:?}",
                c.text
            );
        }
    }

    #[test]
    fn reported_token_count_matches_actual_encoding() {
        use tiktoken_rs::cl100k_base_singleton;
        let text = "Hello world, this is a test of the token counter. Some more words here.";
        let bpe = cl100k_base_singleton();
        let cfg = TokenConfig {
            chunk_size: 10,
            chunk_overlap: 2,
            strategy: TokenStrategy::Recursive,
            ..Default::default()
        };
        let chunks = chunk_tokens(text, &cfg).unwrap();
        assert!(!chunks.is_empty(), "should produce chunks");
        for c in &chunks {
            let actual = bpe.encode_ordinary(&c.text).len();
            assert_eq!(
                actual, c.token_count,
                "reported token_count {} != actual encoding {} for {:?}",
                c.token_count, actual, c.text
            );
            assert!(
                actual <= 10,
                "chunk token count {} > chunk_size 10 for {:?}",
                actual,
                c.text
            );
        }
    }

    #[test]
    fn unicode_and_emoji_offsets() {
        let text = "日本語 😀 café résumé. More text.";
        let cfg = TokenConfig {
            chunk_size: 8,
            chunk_overlap: 2,
            ..Default::default()
        };
        let chars: Vec<char> = text.chars().collect();
        let chunks = chunk_tokens(text, &cfg).unwrap();
        assert!(!chunks.is_empty(), "should produce chunks");
        for c in &chunks {
            assert!(
                c.end <= chars.len(),
                "end={} exceeds chars.len()={}",
                c.end,
                chars.len()
            );
            let slice: String = chars[c.start..c.end].iter().collect();
            assert_eq!(slice, c.text, "Unicode offset invariant");
        }
    }

    #[test]
    fn o200k_base_encoding_works() {
        let text = "Hello world. This is a test with the o200k_base encoding.";
        let cfg = TokenConfig {
            chunk_size: 10,
            chunk_overlap: 2,
            encoding: Encoding::O200kBase,
            ..Default::default()
        };
        let chunks = chunk_tokens(text, &cfg).unwrap();
        assert!(!chunks.is_empty(), "should produce chunks with o200k_base");
        for c in &chunks {
            assert!(
                c.token_count <= 10,
                "o200k_base: chunk has {} tokens > 10",
                c.token_count
            );
        }
    }

    #[test]
    fn fixed_strategy_all_within_limit() {
        let text = "word ".repeat(100);
        let cfg = TokenConfig {
            chunk_size: 20,
            chunk_overlap: 4,
            strategy: TokenStrategy::Fixed,
            ..Default::default()
        };
        let chunks = chunk_tokens(&text, &cfg).unwrap();
        assert!(!chunks.is_empty(), "Fixed: should produce chunks");
        for c in &chunks {
            assert!(
                c.token_count <= 20,
                "Fixed: chunk has {} tokens > 20",
                c.token_count
            );
        }
    }

    #[test]
    fn parallel_matches_sequential() {
        let texts: Vec<String> = (0..16)
            .map(|i| format!("Document {i}: The quick brown fox. ").repeat(20))
            .collect();
        let cfg = TokenConfig {
            chunk_size: 30,
            chunk_overlap: 5,
            ..Default::default()
        };
        let par = chunk_texts_tokens(&texts, &cfg).unwrap();
        for (t, p) in texts.iter().zip(par) {
            let seq = chunk_tokens(t, &cfg).unwrap();
            assert_eq!(seq, p, "parallel must match sequential for same input");
        }
    }

    #[test]
    fn long_text_all_within_token_limit() {
        let text = "The fox jumped over the fence and ran into the forest quickly. ".repeat(80);
        let cfg = TokenConfig {
            chunk_size: 128,
            chunk_overlap: 16,
            ..Default::default()
        };
        let chars: Vec<char> = text.chars().collect();
        let chunks = chunk_tokens(&text, &cfg).unwrap();
        assert!(!chunks.is_empty());
        for c in &chunks {
            assert!(
                c.token_count <= 128,
                "long: chunk has {} tokens > 128",
                c.token_count
            );
            let slice: String = chars[c.start..c.end].iter().collect();
            assert_eq!(slice, c.text, "long: offset invariant");
        }
    }

    #[test]
    fn recursive_keep_separator_modes_produce_valid_offsets() {
        let text = "First paragraph.\n\nSecond paragraph.\n\nThird one here.";
        let chars: Vec<char> = text.chars().collect();
        for ks in [
            KeepSeparator::End,
            KeepSeparator::Start,
            KeepSeparator::Discard,
        ] {
            let cfg = TokenConfig {
                chunk_size: 10,
                chunk_overlap: 2,
                keep_separator: ks,
                drop_whitespace: false,
                ..Default::default()
            };
            let chunks = chunk_tokens(text, &cfg).unwrap();
            for c in &chunks {
                assert!(c.end <= chars.len(), "{ks:?}: end out of bounds");
                let slice: String = chars[c.start..c.end].iter().collect();
                assert_eq!(slice, c.text, "{ks:?}: offset invariant");
            }
        }
    }
}

// ── Compile-time stub tests (no tiktoken-rs feature) ─────────────────────────

#[cfg(all(test, not(feature = "tiktoken-rs")))]
mod stub_tests {
    use super::*;

    #[test]
    fn returns_feature_not_enabled_error() {
        let cfg = TokenConfig::default();
        assert!(matches!(
            chunk_tokens("hello", &cfg),
            Err(TokenChunkError::FeatureNotEnabled)
        ));
    }
}
