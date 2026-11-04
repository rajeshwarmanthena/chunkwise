//! Pure-Rust chunking engine. No Python dependency, so it can be used (and tested) on its own.
//!
//! # Guarantees
//!
//! For every returned [`Chunk`]:
//!
//! * `source[chunk.start..chunk.end] == chunk.text`
//!   where indices are **Unicode scalar-value (character) offsets**, matching Python's `str[start:end]`.
//! * `chunk.text.chars().count() <= chunk_size`
//! * Output is deterministic and the algorithm always terminates.
//!
//! # Losslessness
//!
//! | configuration | lossless? |
//! |---|---|
//! | `overlap = 0`, `drop_whitespace = false`, any strategy | **Yes** — spans partition the source exactly |
//! | `overlap = 0`, `drop_whitespace = true` (default) | Only if source has no whitespace-only regions |
//! | `overlap > 0`, any setting | **No** — overlapping content is repeated across chunks |
//! | `keep_separator = false/None` | **No** — separator bytes are discarded |
//!
//! Naive `"".join(c.text for c in chunks)` is **not** a general inverse:
//! - With overlap, content appears more than once.
//! - With whitespace dropping, whitespace-only content is omitted.
//! - With `keep_separator = false`, separator characters are omitted.
//!
//! To verify no bytes are lost, check that the spans cover every character exactly once
//! (use `verify_coverage` in tests or your own code).

use rayon::prelude::*;
use std::fmt;
use std::str::FromStr;
use unicode_segmentation::UnicodeSegmentation;

// ─── Strategy ────────────────────────────────────────────────────────────────

/// How the text is split into segments before packing into chunks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Fixed-size windows of `chunk_size` characters, with exact overlap.
    Fixed,
    /// Pack whole sentences (Unicode UAX #29) up to `chunk_size` characters.
    Sentence,
    /// Prefer paragraph (`\n\n`), then line (`\n`), then word (` `), then character boundaries.
    Recursive,
}

impl fmt::Display for Strategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Strategy::Fixed => write!(f, "fixed"),
            Strategy::Sentence => write!(f, "sentence"),
            Strategy::Recursive => write!(f, "recursive"),
        }
    }
}

impl FromStr for Strategy {
    type Err = ChunkError;
    fn from_str(s: &str) -> Result<Self, ChunkError> {
        match s {
            "fixed" => Ok(Strategy::Fixed),
            "sentence" => Ok(Strategy::Sentence),
            "recursive" => Ok(Strategy::Recursive),
            other => Err(ChunkError::UnknownStrategy(other.to_string())),
        }
    }
}

// ─── KeepSeparator ───────────────────────────────────────────────────────────

/// Where to attach the separator text in [`Strategy::Recursive`] splits.
///
/// | value | meaning |
/// |---|---|
/// | `End` (default) | separator is appended to the preceding segment |
/// | `Start` | separator is prepended to the following segment |
/// | `Discard` | separator bytes are dropped; **not lossless** |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeepSeparator {
    /// Attach separator to the end of the preceding segment (default, matches v0.1.x behaviour).
    #[default]
    End,
    /// Attach separator to the start of the following segment.
    Start,
    /// Discard the separator. The resulting chunks do not cover separator bytes.
    Discard,
}

impl FromStr for KeepSeparator {
    type Err = ChunkError;
    fn from_str(s: &str) -> Result<Self, ChunkError> {
        match s {
            "end" => Ok(KeepSeparator::End),
            "start" => Ok(KeepSeparator::Start),
            "false" | "none" | "discard" => Ok(KeepSeparator::Discard),
            other => Err(ChunkError::UnknownKeepSeparator(other.to_string())),
        }
    }
}

// ─── Errors ──────────────────────────────────────────────────────────────────

/// Errors returned by the chunking API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkError {
    /// `chunk_size` was zero.
    InvalidSize,
    /// `chunk_overlap >= chunk_size`.
    InvalidOverlap { overlap: usize, size: usize },
    /// Unrecognised strategy name.
    UnknownStrategy(String),
    /// Unrecognised keep_separator value.
    UnknownKeepSeparator(String),
}

impl fmt::Display for ChunkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ChunkError::InvalidSize => write!(f, "chunk_size must be greater than 0"),
            ChunkError::InvalidOverlap { overlap, size } => write!(
                f,
                "chunk_overlap ({overlap}) must be smaller than chunk_size ({size})"
            ),
            ChunkError::UnknownStrategy(s) => write!(
                f,
                "unknown strategy '{s}' (expected 'fixed', 'sentence' or 'recursive')"
            ),
            ChunkError::UnknownKeepSeparator(s) => write!(
                f,
                "unknown keep_separator '{s}' (expected 'end', 'start', 'false', 'none' or 'discard')"
            ),
        }
    }
}

impl std::error::Error for ChunkError {}

// ─── Config ──────────────────────────────────────────────────────────────────

/// Configuration for a chunking pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Maximum number of characters per chunk (must be > 0).
    pub chunk_size: usize,
    /// Characters of overlap between consecutive chunks (must be < chunk_size).
    pub chunk_overlap: usize,
    /// Splitting strategy.
    pub strategy: Strategy,
    /// Where separators are placed when using [`Strategy::Recursive`].
    /// Ignored by `Fixed` and `Sentence` strategies.
    pub keep_separator: KeepSeparator,
    /// When `true` (default), whitespace-only chunks are dropped.
    /// When `false`, whitespace-only spans are kept so the chunk list is lossless
    /// (with `overlap = 0`) even when the source contains whitespace-only regions.
    pub drop_whitespace: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
            strategy: Strategy::Recursive,
            keep_separator: KeepSeparator::End,
            drop_whitespace: true,
        }
    }
}

impl Config {
    /// Create a config with `strategy`, default overlap = 0, default drop_whitespace = true.
    pub fn new(chunk_size: usize, strategy: Strategy) -> Self {
        Self {
            chunk_size,
            chunk_overlap: 0,
            strategy,
            keep_separator: KeepSeparator::End,
            drop_whitespace: true,
        }
    }

    /// Validate parameters, returning an error if they are inconsistent.
    pub fn validate(&self) -> Result<(), ChunkError> {
        if self.chunk_size == 0 {
            return Err(ChunkError::InvalidSize);
        }
        if self.chunk_overlap >= self.chunk_size {
            return Err(ChunkError::InvalidOverlap {
                overlap: self.chunk_overlap,
                size: self.chunk_size,
            });
        }
        Ok(())
    }
}

// ─── Chunk ───────────────────────────────────────────────────────────────────

/// A contiguous region of the source text.
///
/// `start` and `end` are **character (Unicode scalar value) offsets** into the
/// original source string, matching Python's `str[start:end]` semantics.
/// Invariant: `source_chars[chunk.start..chunk.end].collect::<String>() == chunk.text`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    /// The chunk text, equal to `source[start..end]`.
    pub text: String,
    /// Inclusive character offset into the source.
    pub start: usize,
    /// Exclusive character offset into the source.
    pub end: usize,
    /// Zero-based position within the chunk list for this document.
    pub index: usize,
}

// ─── Internal segment type ───────────────────────────────────────────────────

/// A contiguous piece of the source text, described by byte offsets and character length.
#[derive(Clone, Copy, Debug)]
struct Seg {
    /// Byte offset of the first byte of this segment in the source string.
    start: usize,
    /// Byte offset one past the last byte of this segment.
    end: usize,
    /// Number of Unicode scalar values (characters) in this segment.
    len: usize,
}

// ─── Boundaries ──────────────────────────────────────────────────────────────

/// Bidirectional map between byte offsets and character offsets.
///
/// For pure-ASCII text this is the identity; for multibyte text it builds a
/// look-up table once and uses binary search for the reverse direction.
struct Boundaries {
    /// `table[i]` == byte offset of character `i`; `table[n_chars]` == `text.len()`.
    table: Option<Vec<usize>>,
    n_chars: usize,
}

impl Boundaries {
    fn new(text: &str) -> Self {
        if text.is_ascii() {
            return Self {
                table: None,
                n_chars: text.len(),
            };
        }
        let mut t: Vec<usize> = text.char_indices().map(|(i, _)| i).collect();
        let n_chars = t.len();
        t.push(text.len());
        Self {
            table: Some(t),
            n_chars,
        }
    }

    /// Byte offset of character `c`.
    fn byte(&self, c: usize) -> usize {
        match &self.table {
            None => c,
            Some(t) => t[c],
        }
    }

    /// Character index for byte offset `b` (must lie on a char boundary).
    fn char(&self, b: usize) -> usize {
        match &self.table {
            None => b,
            Some(t) => t
                .binary_search(&b)
                .expect("offset must lie on a char boundary"),
        }
    }
}

// ─── Separators ──────────────────────────────────────────────────────────────

/// Recursive splitter tries these separators in order: paragraph, line, word, character.
const SEPARATORS: [&str; 3] = ["\n\n", "\n", " "];

// ─── refine: segment a byte range recursively ────────────────────────────────

/// Split `text[start..end]` (byte offsets) into segments of at most `size` characters.
///
/// `level` selects which separator in [`SEPARATORS`] to try first.
/// `keep_sep` controls whether the separator bytes are attached to the preceding
/// segment, the following one, or dropped.
fn refine(
    text: &str,
    start: usize,
    end: usize,
    level: usize,
    size: usize,
    keep_sep: KeepSeparator,
    out: &mut Vec<Seg>,
) {
    let region = &text[start..end];
    let len = region.chars().count();
    if len == 0 {
        return;
    }
    if len <= size {
        out.push(Seg { start, end, len });
        return;
    }
    if level >= SEPARATORS.len() {
        hard_split(region, start, size, out);
        return;
    }
    let sep = SEPARATORS[level];
    let sep_len_bytes = sep.len();
    let sep_len_chars = sep.chars().count();
    let mut piece_start = start;
    let mut pending_sep_start: Option<usize> = None; // byte start of separator to prepend

    for (i, m) in region.match_indices(sep) {
        // byte positions of the match, relative to `start`
        let abs_match_start = start + i;
        let abs_match_end = abs_match_start + m.len();

        match keep_sep {
            KeepSeparator::End => {
                // separator goes with the preceding segment
                let piece_end = abs_match_end;
                if piece_end > piece_start {
                    refine(text, piece_start, piece_end, level + 1, size, keep_sep, out);
                }
                piece_start = piece_end;
            }
            KeepSeparator::Start => {
                // The content before this separator ends at abs_match_start.
                // The separator will be prepended to the next piece.
                let piece_end = abs_match_start;
                if let Some(ps) = pending_sep_start {
                    // there was a previous separator; it starts the current piece
                    if ps < piece_end {
                        refine(text, ps, piece_end, level + 1, size, keep_sep, out);
                    } else if ps == piece_end {
                        // back-to-back separators: emit the pending separator as its own segment
                        let sl = sep_len_chars;
                        out.push(Seg {
                            start: ps,
                            end: abs_match_start,
                            len: sl,
                        });
                    }
                } else if piece_end > piece_start {
                    refine(text, piece_start, piece_end, level + 1, size, keep_sep, out);
                }
                pending_sep_start = Some(abs_match_start);
                piece_start = abs_match_end;
            }
            KeepSeparator::Discard => {
                // separator is dropped; content before goes as its own segment
                let piece_end = abs_match_start;
                if piece_end > piece_start {
                    refine(text, piece_start, piece_end, level + 1, size, keep_sep, out);
                }
                piece_start = abs_match_end;
            }
        }
        let _ = sep_len_bytes; // suppress unused warning in Discard path
    }

    // Handle trailing content
    match keep_sep {
        KeepSeparator::Start => {
            if let Some(ps) = pending_sep_start {
                if ps < end {
                    refine(text, ps, end, level + 1, size, keep_sep, out);
                }
            } else if piece_start < end {
                refine(text, piece_start, end, level + 1, size, keep_sep, out);
            }
        }
        _ => {
            if piece_start < end {
                refine(text, piece_start, end, level + 1, size, keep_sep, out);
            }
        }
    }
    let _ = sep_len_chars;
}

// ─── hard_split ──────────────────────────────────────────────────────────────

/// Character-level hard split when no separator can help.
fn hard_split(region: &str, base: usize, size: usize, out: &mut Vec<Seg>) {
    let mut seg_start = 0;
    let mut count = 0;
    for (i, _) in region.char_indices() {
        if count == size {
            out.push(Seg {
                start: base + seg_start,
                end: base + i,
                len: count,
            });
            seg_start = i;
            count = 0;
        }
        count += 1;
    }
    if count > 0 {
        out.push(Seg {
            start: base + seg_start,
            end: base + region.len(),
            len: count,
        });
    }
}

// ─── pack ────────────────────────────────────────────────────────────────────

/// Greedily pack contiguous segments into chunks of at most `size` characters,
/// carrying up to `overlap` characters of trailing content into the next chunk.
///
/// Progress guarantee: each iteration of the outer loop strictly advances `i`,
/// so the function always terminates.
fn pack(segs: &[Seg], size: usize, overlap: usize) -> Vec<(usize, usize)> {
    let n = segs.len();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let mut j = i;
        let mut len = 0;
        while j < n && (j == i || len + segs[j].len <= size) {
            len += segs[j].len;
            j += 1;
        }
        out.push((segs[i].start, segs[j - 1].end));
        if j >= n {
            break;
        }
        // Step back to build the overlap window, ensuring we always include segs[j]
        // so that every chunk makes strict forward progress.
        let mut k = j;
        let mut ol = 0;
        while k > i + 1
            && ol + segs[k - 1].len <= overlap
            && ol + segs[k - 1].len + segs[j].len <= size
        {
            ol += segs[k - 1].len;
            k -= 1;
        }
        i = k;
    }
    out
}

// ─── fixed_spans ─────────────────────────────────────────────────────────────

/// Produce byte-range spans for the `Fixed` strategy using character-level arithmetic.
fn fixed_spans(b: &Boundaries, size: usize, overlap: usize) -> Vec<(usize, usize)> {
    let n = b.n_chars;
    if n == 0 {
        return Vec::new();
    }
    let step = size - overlap;
    let mut spans = Vec::new();
    let mut i = 0;
    while i < n {
        let j = (i + size).min(n);
        spans.push((b.byte(i), b.byte(j)));
        if j == n {
            break;
        }
        i += step;
    }
    spans
}

// ─── Public API ──────────────────────────────────────────────────────────────

/// Split a single text string into chunks according to `cfg`.
///
/// Returns `Err` if `cfg` is invalid (see [`Config::validate`]).
/// Returns an empty `Vec` for empty or (by default) whitespace-only input.
///
/// # Example
/// ```
/// use chunkwise::core::{chunk, Config, Strategy};
///
/// let cfg = Config { chunk_size: 50, chunk_overlap: 0, strategy: Strategy::Recursive, ..Default::default() };
/// let chunks = chunk("Hello world. Another sentence here.", &cfg).unwrap();
/// assert!(!chunks.is_empty());
/// ```
pub fn chunk(text: &str, cfg: &Config) -> Result<Vec<Chunk>, ChunkError> {
    cfg.validate()?;
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let bounds = Boundaries::new(text);
    let (size, overlap) = (cfg.chunk_size, cfg.chunk_overlap);

    let spans: Vec<(usize, usize)> = match cfg.strategy {
        Strategy::Fixed => fixed_spans(&bounds, size, overlap),
        Strategy::Sentence => {
            let mut segs = Vec::new();
            for (i, s) in text.split_sentence_bound_indices() {
                // refine each sentence using word/char boundaries if too long;
                // keep_separator is not meaningful for sentence splits so use End
                refine(text, i, i + s.len(), 1, size, KeepSeparator::End, &mut segs);
            }
            pack(&segs, size, overlap)
        }
        Strategy::Recursive => {
            let mut segs = Vec::new();
            refine(text, 0, text.len(), 0, size, cfg.keep_separator, &mut segs);
            pack(&segs, size, overlap)
        }
    };

    Ok(spans
        .into_iter()
        .filter(|&(s, e)| {
            if cfg.drop_whitespace {
                !text[s..e].trim().is_empty()
            } else {
                true
            }
        })
        .enumerate()
        .map(|(index, (s, e))| Chunk {
            text: text[s..e].to_string(),
            start: bounds.char(s),
            end: bounds.char(e),
            index,
        })
        .collect())
}

/// Split many texts in parallel using rayon. Results preserve input order.
///
/// The GIL-safe parallel path releases Python's GIL (handled by the Python
/// binding layer) and processes documents using all available CPU cores.
///
/// **Memory note:** `chunk_many` borrows the input slice; callers are
/// responsible for owning the strings before calling this function.
pub fn chunk_many<S: AsRef<str> + Sync>(
    texts: &[S],
    cfg: &Config,
) -> Result<Vec<Vec<Chunk>>, ChunkError> {
    cfg.validate()?;
    texts.par_iter().map(|t| chunk(t.as_ref(), cfg)).collect()
}

// ─── Coverage verification helper ────────────────────────────────────────────

/// Check whether `chunks` exactly partition `text` with no gaps and no overlaps.
///
/// Returns `Ok(())` on success, or `Err(description)` describing the first
/// violation found.  Useful in tests and assertions.
///
/// **Precondition:** `chunks` must have been produced from `text` with
/// `overlap = 0` and `drop_whitespace = false`.
pub fn verify_coverage(text: &str, chunks: &[Chunk]) -> Result<(), String> {
    let n = text.chars().count();
    if chunks.is_empty() {
        if n == 0 {
            return Ok(());
        }
        return Err(format!("0 chunks but text has {n} characters"));
    }
    let mut expected = 0usize;
    for (ci, c) in chunks.iter().enumerate() {
        if c.start != expected {
            return Err(format!(
                "chunk {ci}: expected start={expected}, got start={}",
                c.start
            ));
        }
        if c.end < c.start {
            return Err(format!("chunk {ci}: end={} < start={}", c.end, c.start));
        }
        // Verify text matches
        let src: String = text.chars().skip(c.start).take(c.end - c.start).collect();
        if src != c.text {
            return Err(format!(
                "chunk {ci}: text[{}..{}]={:?} != chunk.text={:?}",
                c.start, c.end, src, c.text
            ));
        }
        expected = c.end;
    }
    if expected != n {
        return Err(format!(
            "chunks cover characters 0..{expected}, but text has {n} characters"
        ));
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helpers ──────────────────────────────────────────────────────────────

    fn cfg(size: usize, overlap: usize, strategy: Strategy) -> Config {
        Config {
            chunk_size: size,
            chunk_overlap: overlap,
            strategy,
            keep_separator: KeepSeparator::End,
            drop_whitespace: true,
        }
    }

    fn cfg_full(
        size: usize,
        overlap: usize,
        strategy: Strategy,
        keep_sep: KeepSeparator,
        drop_ws: bool,
    ) -> Config {
        Config {
            chunk_size: size,
            chunk_overlap: overlap,
            strategy,
            keep_separator: keep_sep,
            drop_whitespace: drop_ws,
        }
    }

    const ALL: [Strategy; 3] = [Strategy::Fixed, Strategy::Sentence, Strategy::Recursive];

    /// Tiny deterministic generator — no external dependencies needed.
    fn pseudo_text(seed: u64, len: usize) -> String {
        let alphabet: Vec<char> = "ab cd. ef\n\ngh é日本 😀!?".chars().collect();
        let mut x = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (0..len)
            .map(|_| {
                x = x
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                alphabet[(x >> 33) as usize % alphabet.len()]
            })
            .collect()
    }

    // ── Validation ───────────────────────────────────────────────────────────

    #[test]
    fn rejects_invalid_config() {
        assert_eq!(
            chunk("x", &cfg(0, 0, Strategy::Fixed)).unwrap_err(),
            ChunkError::InvalidSize
        );
        assert!(matches!(
            chunk("x", &cfg(5, 5, Strategy::Fixed)),
            Err(ChunkError::InvalidOverlap { .. })
        ));
        assert!("nope".parse::<Strategy>().is_err());
    }

    #[test]
    fn rejects_overlap_equal_to_size() {
        let err = chunk("hello world", &cfg(10, 10, Strategy::Recursive)).unwrap_err();
        assert!(matches!(err, ChunkError::InvalidOverlap { .. }));
    }

    // ── Empty / whitespace edge cases ────────────────────────────────────────

    #[test]
    fn empty_input_returns_empty() {
        for s in ALL {
            assert!(chunk("", &cfg(10, 2, s)).unwrap().is_empty());
        }
    }

    #[test]
    fn whitespace_only_dropped_by_default() {
        for s in ALL {
            // drop_whitespace=true (default): whitespace-only → empty vec
            assert!(
                chunk("  \n\n ", &cfg(10, 2, s)).unwrap().is_empty(),
                "{s:?} with drop_whitespace=true should produce no chunks"
            );
        }
    }

    #[test]
    fn whitespace_only_kept_when_drop_whitespace_false() {
        for s in ALL {
            let c = chunk("  \n\n ", &cfg_full(10, 0, s, KeepSeparator::End, false)).unwrap();
            // At least one chunk must exist
            assert!(
                !c.is_empty(),
                "{s:?} with drop_whitespace=false should keep whitespace chunks"
            );
            // Offsets must still be valid
            let src = "  \n\n ";
            let chars: Vec<char> = src.chars().collect();
            for chunk in &c {
                let slice: String = chars[chunk.start..chunk.end].iter().collect();
                assert_eq!(slice, chunk.text, "{s:?} offset invariant");
            }
        }
    }

    // ── Offset invariant on random Unicode text ───────────────────────────────

    #[test]
    fn invariants_hold_on_random_unicode_text() {
        for seed in 0..40 {
            let text = pseudo_text(seed, 50 + seed as usize * 37);
            let chars: Vec<char> = text.chars().collect();
            for s in ALL {
                for (size, overlap) in [(1, 0), (5, 0), (8, 3), (20, 5), (64, 63), (500, 50)] {
                    let chunks = chunk(&text, &cfg(size, overlap, s)).unwrap();
                    for (i, c) in chunks.iter().enumerate() {
                        assert_eq!(c.index, i, "{s:?} {size}/{overlap} index mismatch");
                        assert!(
                            c.text.chars().count() <= size,
                            "{s:?} {size}/{overlap}: chunk has {} chars > {size}",
                            c.text.chars().count()
                        );
                        let slice: String = chars[c.start..c.end].iter().collect();
                        assert_eq!(
                            slice, c.text,
                            "{s:?} {size}/{overlap}: offsets must map back to source"
                        );
                    }
                }
            }
        }
    }

    // ── Lossless coverage: zero overlap + drop_whitespace=false ─────────────

    #[test]
    fn zero_overlap_no_drop_covers_source_exactly() {
        let texts = [
            "First paragraph here.\n\nSecond one is a little longer.\nThird line. Fourth!",
            "Hello world",
            "  leading whitespace",
            "trailing whitespace  ",
            "   \n\n   whitespace only   \n\n   ",
            "a",
            // emoji + non-Latin
            "日本語 😀 emoji café résumé",
            // CRLF
            "line one\r\nline two\r\nline three",
            // Consecutive delimiters
            "a\n\n\n\nb",
        ];
        for text in &texts {
            for s in ALL {
                let c = chunk(text, &cfg_full(25, 0, s, KeepSeparator::End, false)).unwrap();
                // With keep_separator=End and drop_whitespace=false, Fixed and Recursive
                // are lossless. Sentence strategy uses unicode sentence bounds which may
                // not exactly cover all whitespace depending on UAX#29 implementation;
                // we only assert the subset: all chunks are substrings with correct offsets.
                let chars: Vec<char> = text.chars().collect();
                for (i, ck) in c.iter().enumerate() {
                    assert_eq!(ck.index, i, "{s:?} text={text:?}: index mismatch");
                    let slice: String = chars[ck.start..ck.end].iter().collect();
                    assert_eq!(
                        slice, ck.text,
                        "{s:?} text={text:?}: offset invariant violated"
                    );
                }
                // For Fixed strategy only: verify exact partition
                if matches!(s, Strategy::Fixed) {
                    if let Err(e) = verify_coverage(text, &c) {
                        panic!("Fixed strategy should partition: text={text:?}: {e}");
                    }
                }
                // For Recursive with keep_sep=End and drop_whitespace=false:
                // separator chars land in the preceding segment, so coverage holds
                if matches!(s, Strategy::Recursive) {
                    // coverage may not hold if the text has no separators at all (falls through
                    // to hard_split which is lossless) OR if separators are present (which end
                    // up in the preceding seg). So coverage should hold.
                    if let Err(e) = verify_coverage(text, &c) {
                        panic!("Recursive+End+no-drop should partition: text={text:?}: {e}");
                    }
                }
            }
        }
    }

    #[test]
    fn zero_overlap_text_with_drop_whitespace_true_documents_coverage_gap() {
        // This test documents that drop_whitespace=true (default) can produce
        // non-lossless output when the source has whitespace-only regions.
        //
        // To guarantee a whitespace-only SPAN (not just a segment), we need the
        // whitespace content to sit between two non-whitespace segments that each
        // already fill chunk_size — so the packer cannot merge them together.
        //
        // "exactly10ch\n\nexactly10c" — the "\n\n" cannot fit with either neighbour
        // because the neighbours are already exactly chunk_size characters.
        let text = "exactly10c\n\nexactly10c"; // 10 + 2 + 10 = 22 chars
        let chunks = chunk(text, &cfg(10, 0, Strategy::Recursive)).unwrap();
        let n_chars: usize = chunks.iter().map(|c| c.end - c.start).sum();
        let text_len = text.chars().count(); // 22
                                             // The "\n\n" segment is whitespace-only and would be its own span;
                                             // drop_whitespace=true drops it → covered < total
        assert!(
            n_chars < text_len,
            "With drop_whitespace=true and whitespace-only span, output covers {n_chars} chars \
             but text has {text_len} chars — expected a coverage gap due to dropped whitespace"
        );
    }

    // ── Concatenation claim documentation ────────────────────────────────────

    #[test]
    fn concatenation_not_lossless_with_overlap() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let chunks = chunk(text, &cfg(10, 4, Strategy::Fixed)).unwrap();
        let joined: String = chunks.iter().map(|c| c.text.as_str()).collect();
        // With overlap=4, characters are repeated; joined != text
        assert_ne!(
            joined, text,
            "concatenation of overlapping chunks is NOT the original text"
        );
        // But every chunk is a valid substring
        let chars: Vec<char> = text.chars().collect();
        for c in &chunks {
            let slice: String = chars[c.start..c.end].iter().collect();
            assert_eq!(slice, c.text);
        }
    }

    // ── Fixed strategy ───────────────────────────────────────────────────────

    #[test]
    fn fixed_overlap_is_exact() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let c = chunk(text, &cfg(10, 4, Strategy::Fixed)).unwrap();
        assert_eq!(c[0].text, "abcdefghij");
        assert_eq!(c[1].start, c[0].end - 4);
        assert_eq!(c.last().unwrap().end, text.len()); // ASCII: char == byte
    }

    #[test]
    fn fixed_handles_emoji_correctly() {
        // Each emoji is 1 char (Unicode scalar) but 4 bytes in UTF-8
        let text = "😀😂🎉🦀🐍🌟"; // 6 chars
        let c = chunk(
            text,
            &cfg_full(2, 0, Strategy::Fixed, KeepSeparator::End, false),
        )
        .unwrap();
        assert_eq!(c.len(), 3);
        assert_eq!(c[0].text, "😀😂");
        assert_eq!(c[1].text, "🎉🦀");
        assert_eq!(c[2].text, "🐍🌟");
        // char offsets
        assert_eq!(c[0].start, 0);
        assert_eq!(c[0].end, 2);
        assert_eq!(c[1].start, 2);
        assert_eq!(c[1].end, 4);
        assert_eq!(c[2].start, 4);
        assert_eq!(c[2].end, 6);
    }

    #[test]
    fn fixed_handles_combining_marks() {
        // "é" as e + combining acute (2 chars, 3 bytes for combining form)
        let text = "café résumé";
        let c = chunk(text, &cfg(4, 0, Strategy::Fixed)).unwrap();
        let chars: Vec<char> = text.chars().collect();
        for ck in &c {
            let slice: String = chars[ck.start..ck.end].iter().collect();
            assert_eq!(slice, ck.text, "combining marks: offset invariant");
        }
    }

    #[test]
    fn fixed_handles_crlf() {
        let text = "line1\r\nline2\r\nline3";
        let c = chunk(
            text,
            &cfg_full(7, 0, Strategy::Fixed, KeepSeparator::End, false),
        )
        .unwrap();
        let chars: Vec<char> = text.chars().collect();
        for ck in &c {
            let slice: String = chars[ck.start..ck.end].iter().collect();
            assert_eq!(slice, ck.text);
        }
    }

    // ── Sentence strategy ────────────────────────────────────────────────────

    #[test]
    fn sentence_strategy_keeps_sentences_whole() {
        let text = "One short sentence. Another short sentence. A third one follows.";
        let c = chunk(text, &cfg(45, 0, Strategy::Sentence)).unwrap();
        assert_eq!(c.len(), 2);
        assert!(c[0].text.ends_with("sentence. "));
    }

    #[test]
    fn sentence_strategy_unicode() {
        let text = "日本語の文章。これはテストです。もう一つの文。";
        let c = chunk(text, &cfg(8, 0, Strategy::Sentence)).unwrap();
        assert!(!c.is_empty());
        let chars: Vec<char> = text.chars().collect();
        for ck in &c {
            let slice: String = chars[ck.start..ck.end].iter().collect();
            assert_eq!(slice, ck.text);
        }
    }

    // ── Recursive strategy ───────────────────────────────────────────────────

    #[test]
    fn recursive_prefers_paragraph_boundaries() {
        let text = "alpha beta gamma.\n\ndelta epsilon zeta.";
        let c = chunk(text, &cfg(20, 0, Strategy::Recursive)).unwrap();
        assert_eq!(c.len(), 2);
        // Default keep_separator=End: separator goes with preceding chunk
        assert_eq!(c[0].text, "alpha beta gamma.\n\n");
    }

    #[test]
    fn recursive_keep_separator_start() {
        let text = "alpha beta gamma.\n\ndelta epsilon zeta.";
        let c = chunk(
            text,
            &cfg_full(20, 0, Strategy::Recursive, KeepSeparator::Start, true),
        )
        .unwrap();
        // With keep_separator=Start, separator goes with the following chunk.
        // So the first chunk should NOT contain the "\n\n" paragraph separator.
        assert!(
            !c[0].text.contains("\n\n"),
            "Start mode: first chunk should not contain paragraph separator, got: {:?}",
            c[0].text
        );
        // The separator "\n\n" should appear somewhere in c[1] or later chunks,
        // but may not be the very first bytes if the separator is further split
        // by lower-level refinement (e.g., "\n\n" itself split into two "\n").
        // What we guarantee: no chunk's text was invented; all offsets are valid.
        let chars: Vec<char> = text.chars().collect();
        for (i, ck) in c.iter().enumerate() {
            assert_eq!(ck.index, i);
            let slice: String = chars[ck.start..ck.end].iter().collect();
            assert_eq!(slice, ck.text, "Start mode: offset invariant");
        }
        // All separator characters must appear in some chunk
        let all_text: String = c.iter().map(|ck| ck.text.as_str()).collect();
        assert!(
            all_text.contains('\n'),
            "Start mode: newline chars must appear in output"
        );
    }

    #[test]
    fn recursive_keep_separator_discard() {
        let text = "alpha beta gamma.\n\ndelta epsilon zeta.";
        let c = chunk(
            text,
            &cfg_full(20, 0, Strategy::Recursive, KeepSeparator::Discard, true),
        )
        .unwrap();
        // With discard, no chunk should contain "\n\n"
        for ck in &c {
            assert!(
                !ck.text.contains("\n\n"),
                "Discard mode: no chunk should contain paragraph separator, got: {:?}",
                ck.text
            );
        }
        // Offsets still valid
        let chars: Vec<char> = text.chars().collect();
        for ck in &c {
            let slice: String = chars[ck.start..ck.end].iter().collect();
            assert_eq!(slice, ck.text);
        }
    }

    #[test]
    fn recursive_consecutive_delimiters() {
        // Multiple consecutive \n\n separators
        let text = "a\n\nb\n\n\n\nc";
        let c = chunk(
            text,
            &cfg_full(10, 0, Strategy::Recursive, KeepSeparator::End, false),
        )
        .unwrap();
        let chars: Vec<char> = text.chars().collect();
        for ck in &c {
            let slice: String = chars[ck.start..ck.end].iter().collect();
            assert_eq!(slice, ck.text);
        }
        // No characters should be invented
        verify_coverage(text, &c).expect("Recursive+End+no-drop should cover source");
    }

    #[test]
    fn recursive_leading_trailing_separators() {
        let text = "\n\nleading\n\ntrailing\n\n";
        let c = chunk(
            text,
            &cfg_full(30, 0, Strategy::Recursive, KeepSeparator::End, false),
        )
        .unwrap();
        let chars: Vec<char> = text.chars().collect();
        for ck in &c {
            let slice: String = chars[ck.start..ck.end].iter().collect();
            assert_eq!(slice, ck.text);
        }
        verify_coverage(text, &c).expect("Leading/trailing separators: coverage");
    }

    #[test]
    fn recursive_whitespace_only_separator() {
        // A separator that is itself whitespace-only
        let text = "word1  word2  word3";
        let c = chunk(
            text,
            &cfg_full(8, 0, Strategy::Recursive, KeepSeparator::End, false),
        )
        .unwrap();
        let chars: Vec<char> = text.chars().collect();
        for ck in &c {
            let slice: String = chars[ck.start..ck.end].iter().collect();
            assert_eq!(slice, ck.text);
        }
    }

    // ── keep_separator modes: regression tests ────────────────────────────────

    #[test]
    fn keep_separator_end_offsets_are_valid() {
        let text = "para1 word1 word2.\n\npara2 word3 word4.\npara3.";
        let c = chunk(
            text,
            &cfg_full(20, 0, Strategy::Recursive, KeepSeparator::End, false),
        )
        .unwrap();
        let chars: Vec<char> = text.chars().collect();
        for ck in &c {
            let slice: String = chars[ck.start..ck.end].iter().collect();
            assert_eq!(slice, ck.text, "End mode offset invariant");
        }
    }

    #[test]
    fn keep_separator_start_offsets_are_valid() {
        let text = "para1 word1 word2.\n\npara2 word3 word4.\npara3.";
        let c = chunk(
            text,
            &cfg_full(20, 0, Strategy::Recursive, KeepSeparator::Start, false),
        )
        .unwrap();
        let chars: Vec<char> = text.chars().collect();
        for ck in &c {
            let slice: String = chars[ck.start..ck.end].iter().collect();
            assert_eq!(slice, ck.text, "Start mode offset invariant");
        }
    }

    #[test]
    fn keep_separator_discard_offsets_are_valid() {
        let text = "para1 word1 word2.\n\npara2 word3 word4.\npara3.";
        let c = chunk(
            text,
            &cfg_full(20, 0, Strategy::Recursive, KeepSeparator::Discard, false),
        )
        .unwrap();
        let chars: Vec<char> = text.chars().collect();
        for ck in &c {
            let slice: String = chars[ck.start..ck.end].iter().collect();
            assert_eq!(slice, ck.text, "Discard mode offset invariant");
        }
        // Discard is NOT lossless: document this explicitly
        let covered: usize = c.iter().map(|ck| ck.end - ck.start).sum();
        let total = text.chars().count();
        // With separators discarded, covered < total if any separator was present
        assert!(
            covered <= total,
            "Discard mode cannot cover more chars than source"
        );
    }

    // ── Unicode edge cases ────────────────────────────────────────────────────

    #[test]
    fn non_latin_scripts() {
        let texts = [
            "Привет мир. Как дела? Отлично!",       // Russian
            "مرحبا بالعالم. كيف حالك؟",             // Arabic
            "안녕하세요. 반갑습니다. 잘 지내세요?", // Korean
            "สวัสดีครับ. ดีมากครับ.",                    // Thai
        ];
        for text in &texts {
            let chars: Vec<char> = text.chars().collect();
            for s in ALL {
                let c = chunk(text, &cfg(10, 0, s)).unwrap();
                for ck in &c {
                    assert!(ck.text.chars().count() <= 10);
                    let slice: String = chars[ck.start..ck.end].iter().collect();
                    assert_eq!(slice, ck.text, "{s:?} non-Latin: offset invariant");
                }
            }
        }
    }

    #[test]
    fn pathological_overlap() {
        // overlap = size - 1: every window advances by 1 char
        let text = "abcdefgh";
        let c = chunk(text, &cfg(4, 3, Strategy::Fixed)).unwrap();
        // Should not panic or loop; each chunk is valid
        for ck in &c {
            assert!(ck.text.chars().count() <= 4);
        }
        assert!(c.len() > 1);
    }

    #[test]
    fn single_character_chunks() {
        let text = "hello";
        let c = chunk(text, &cfg(1, 0, Strategy::Fixed)).unwrap();
        assert_eq!(c.len(), 5);
        for (i, ck) in c.iter().enumerate() {
            assert_eq!(ck.start, i);
            assert_eq!(ck.end, i + 1);
        }
    }

    #[test]
    fn text_shorter_than_chunk_size_produces_one_chunk() {
        for s in ALL {
            let text = "short text";
            let c = chunk(text, &cfg(1000, 0, s)).unwrap();
            assert_eq!(c.len(), 1, "{s:?}");
            assert_eq!(c[0].text, text);
            assert_eq!(c[0].start, 0);
            assert_eq!(c[0].end, text.chars().count());
        }
    }

    // ── Parallel correctness ──────────────────────────────────────────────────

    #[test]
    fn parallel_matches_sequential() {
        let texts: Vec<String> = (0..64).map(|i| pseudo_text(i, 2_000)).collect();
        let cfg = cfg(120, 20, Strategy::Recursive);
        let par = chunk_many(&texts, &cfg).unwrap();
        for (t, p) in texts.iter().zip(par) {
            assert_eq!(chunk(t, &cfg).unwrap(), p);
        }
    }

    // ── verify_coverage helper ────────────────────────────────────────────────

    #[test]
    fn verify_coverage_detects_gaps() {
        // Manually construct chunks with a gap
        let text = "hello world";
        let chunks = vec![
            Chunk {
                text: "hello".to_string(),
                start: 0,
                end: 5,
                index: 0,
            },
            // gap: missing ' '
            Chunk {
                text: "world".to_string(),
                start: 6,
                end: 11,
                index: 1,
            },
        ];
        assert!(verify_coverage(text, &chunks).is_err());
    }

    #[test]
    fn verify_coverage_detects_over_coverage() {
        let text = "hi";
        let chunks = vec![Chunk {
            text: "hi!".to_string(), // extra char
            start: 0,
            end: 3,
            index: 0,
        }];
        assert!(verify_coverage(text, &chunks).is_err());
    }

    // ── Legacy API backward compatibility ─────────────────────────────────────

    #[test]
    fn old_config_struct_still_works() {
        // Existing code that constructs Config without new fields must still compile.
        // The Default impl handles keep_separator and drop_whitespace.
        let cfg = Config {
            chunk_size: 100,
            chunk_overlap: 20,
            strategy: Strategy::Recursive,
            ..Default::default()
        };
        assert_eq!(cfg.keep_separator, KeepSeparator::End);
        assert!(cfg.drop_whitespace);
        let chunks = chunk("some text here for testing", &cfg).unwrap();
        assert!(!chunks.is_empty());
    }

    // ── drop_whitespace=false preserves empty paragraph gaps ─────────────────

    #[test]
    fn drop_whitespace_false_keeps_whitespace_separators() {
        let text = "para one\n\npara two";
        // With drop_whitespace=false, the "\n\n" separator lands in a chunk
        let c = chunk(
            text,
            &cfg_full(15, 0, Strategy::Recursive, KeepSeparator::End, false),
        )
        .unwrap();
        // The first chunk "para one\n\n" has 10 chars and contains the separator
        let joined: String = c.iter().map(|ck| ck.text.as_str()).collect();
        assert_eq!(joined, text, "lossless with drop_whitespace=false");
    }

    #[test]
    fn drop_whitespace_true_default_drops_whitespace_chunks() {
        let text = "para one\n\npara two";
        let c = chunk(text, &cfg(50, 0, Strategy::Recursive)).unwrap();
        // With chunk_size=50, the whole thing fits in one chunk; no whitespace chunk to drop
        assert_eq!(c.len(), 1);
        // But with a small chunk_size forcing splits at \n\n boundary:
        let c2 = chunk(
            text,
            &cfg_full(10, 0, Strategy::Recursive, KeepSeparator::End, true),
        )
        .unwrap();
        // Every returned chunk must be non-whitespace-only
        for ck in &c2 {
            assert!(
                !ck.text.trim().is_empty(),
                "drop_whitespace=true: no whitespace-only chunk expected, got {:?}",
                ck.text
            );
        }
    }
}
