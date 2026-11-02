//! Pure-Rust chunking engine. No Python dependency, so it can be used (and tested) on its own.
//!
//! Guarantees, enforced by tests:
//! * `text[chunk.start..chunk.end] == chunk.text` (offsets are in Unicode scalar values),
//!   so every chunk can be traced back to its source for citations and audits;
//! * no chunk is longer than `chunk_size` characters;
//! * output is deterministic and the algorithm always terminates.

use rayon::prelude::*;
use std::fmt;
use std::str::FromStr;
use unicode_segmentation::UnicodeSegmentation;

/// How the text is split.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Strategy {
    /// Fixed-size windows of `chunk_size` characters.
    Fixed,
    /// Pack whole sentences (Unicode UAX #29) up to `chunk_size`.
    Sentence,
    /// Prefer paragraph, then line, then word boundaries, then characters.
    Recursive,
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkError {
    InvalidSize,
    InvalidOverlap { overlap: usize, size: usize },
    UnknownStrategy(String),
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
        }
    }
}

impl std::error::Error for ChunkError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub strategy: Strategy,
}

impl Config {
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

/// A chunk of the source text. `start`/`end` are character (not byte) offsets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub index: usize,
}

/// A contiguous piece of the text: byte range plus length in characters.
#[derive(Clone, Copy)]
struct Seg {
    start: usize,
    end: usize,
    len: usize,
}

/// Maps between byte offsets and character offsets. Identity for ASCII text.
struct Boundaries {
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
    fn byte(&self, c: usize) -> usize {
        match &self.table {
            None => c,
            Some(t) => t[c],
        }
    }
    fn char(&self, b: usize) -> usize {
        match &self.table {
            None => b,
            Some(t) => t.binary_search(&b).expect("offset lies on a char boundary"),
        }
    }
}

const SEPARATORS: [&str; 3] = ["\n\n", "\n", " "];

/// Split `text[start..end]` into segments no longer than `size` characters,
/// trying separators from `level` onwards and falling back to a hard split.
fn refine(text: &str, start: usize, end: usize, level: usize, size: usize, out: &mut Vec<Seg>) {
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
    let mut piece_start = start;
    for (i, m) in region.match_indices(sep) {
        let piece_end = start + i + m.len();
        if piece_end > piece_start {
            refine(text, piece_start, piece_end, level + 1, size, out);
            piece_start = piece_end;
        }
    }
    if piece_start < end {
        refine(text, piece_start, end, level + 1, size, out);
    }
}

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

/// Greedily pack contiguous segments into chunks of at most `size` characters,
/// carrying up to `overlap` characters of trailing segments into the next chunk.
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
        // Step back to build the overlap, but always leave room for segs[j]
        // so that every chunk makes forward progress.
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

fn fixed_spans(b: &Boundaries, size: usize, overlap: usize) -> Vec<(usize, usize)> {
    let n = b.n_chars;
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

/// Chunk a single text.
pub fn chunk(text: &str, cfg: &Config) -> Result<Vec<Chunk>, ChunkError> {
    cfg.validate()?;
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let bounds = Boundaries::new(text);
    let (size, overlap) = (cfg.chunk_size, cfg.chunk_overlap);
    let spans = match cfg.strategy {
        Strategy::Fixed => fixed_spans(&bounds, size, overlap),
        Strategy::Sentence => {
            let mut segs = Vec::new();
            for (i, s) in text.split_sentence_bound_indices() {
                refine(text, i, i + s.len(), 1, size, &mut segs);
            }
            pack(&segs, size, overlap)
        }
        Strategy::Recursive => {
            let mut segs = Vec::new();
            refine(text, 0, text.len(), 0, size, &mut segs);
            pack(&segs, size, overlap)
        }
    };
    Ok(spans
        .into_iter()
        .filter(|&(s, e)| !text[s..e].trim().is_empty())
        .enumerate()
        .map(|(index, (s, e))| Chunk {
            text: text[s..e].to_string(),
            start: bounds.char(s),
            end: bounds.char(e),
            index,
        })
        .collect())
}

/// Chunk many texts in parallel (rayon). Results keep the input order.
pub fn chunk_many<S: AsRef<str> + Sync>(
    texts: &[S],
    cfg: &Config,
) -> Result<Vec<Vec<Chunk>>, ChunkError> {
    cfg.validate()?;
    texts.par_iter().map(|t| chunk(t.as_ref(), cfg)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(size: usize, overlap: usize, strategy: Strategy) -> Config {
        Config {
            chunk_size: size,
            chunk_overlap: overlap,
            strategy,
        }
    }

    const ALL: [Strategy; 3] = [Strategy::Fixed, Strategy::Sentence, Strategy::Recursive];

    /// Tiny deterministic generator so the suite needs no extra dependencies.
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
    fn empty_and_whitespace_only() {
        for s in ALL {
            assert!(chunk("", &cfg(10, 2, s)).unwrap().is_empty());
            assert!(chunk("  \n\n ", &cfg(10, 2, s)).unwrap().is_empty());
        }
    }

    #[test]
    fn invariants_hold_on_random_unicode_text() {
        for seed in 0..40 {
            let text = pseudo_text(seed, 50 + seed as usize * 37);
            let chars: Vec<char> = text.chars().collect();
            for s in ALL {
                for (size, overlap) in [(1, 0), (5, 0), (8, 3), (20, 5), (64, 63), (500, 50)] {
                    let chunks = chunk(&text, &cfg(size, overlap, s)).unwrap();
                    for (i, c) in chunks.iter().enumerate() {
                        assert_eq!(c.index, i);
                        assert!(c.text.chars().count() <= size, "{s:?} {size}/{overlap}");
                        let slice: String = chars[c.start..c.end].iter().collect();
                        assert_eq!(slice, c.text, "offsets must map back to the source");
                    }
                }
            }
        }
    }

    #[test]
    fn zero_overlap_reconstructs_the_text() {
        let text = "First paragraph here.\n\nSecond one is a little longer, with words.\nThird line. Fourth sentence!";
        for s in ALL {
            let joined: String = chunk(text, &cfg(25, 0, s))
                .unwrap()
                .into_iter()
                .map(|c| c.text)
                .collect();
            assert_eq!(joined, text, "{s:?}");
        }
    }

    #[test]
    fn fixed_overlap_is_exact() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let c = chunk(text, &cfg(10, 4, Strategy::Fixed)).unwrap();
        assert_eq!(c[0].text, "abcdefghij");
        assert_eq!(c[1].start, c[0].end - 4);
        assert_eq!(c.last().unwrap().end, text.len());
    }

    #[test]
    fn sentence_strategy_keeps_sentences_whole() {
        let text = "One short sentence. Another short sentence. A third one follows.";
        let c = chunk(text, &cfg(45, 0, Strategy::Sentence)).unwrap();
        assert_eq!(c.len(), 2);
        assert!(c[0].text.ends_with("sentence. "));
    }

    #[test]
    fn recursive_prefers_paragraph_boundaries() {
        let text = "alpha beta gamma.\n\ndelta epsilon zeta.";
        let c = chunk(text, &cfg(20, 0, Strategy::Recursive)).unwrap();
        assert_eq!(c.len(), 2);
        assert_eq!(c[0].text, "alpha beta gamma.\n\n");
    }

    #[test]
    fn parallel_matches_sequential() {
        let texts: Vec<String> = (0..64).map(|i| pseudo_text(i, 2_000)).collect();
        let cfg = cfg(120, 20, Strategy::Recursive);
        let par = chunk_many(&texts, &cfg).unwrap();
        for (t, p) in texts.iter().zip(par) {
            assert_eq!(chunk(t, &cfg).unwrap(), p);
        }
    }
}
