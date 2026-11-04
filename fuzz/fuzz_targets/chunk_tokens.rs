#![no_main]
//! Feeds arbitrary text and parameters to token chunking and verifies invariants:
//! no panic, chunk token_count <= chunk_size, offsets strictly map back to source.

use chunkwise::core::KeepSeparator;
use chunkwise::token_core::{chunk_tokens, Encoding, TokenConfig, TokenStrategy};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: (&str, u8, u8, u8, u8, u8, bool)| {
    let (text, size, overlap, enc, strat, keep_sep, drop_ws) = input;
    let size = (size as usize % 64) + 1;
    let overlap = overlap as usize % size;
    let encoding = if enc % 2 == 0 {
        Encoding::Cl100kBase
    } else {
        Encoding::O200kBase
    };
    let strategy = if strat % 2 == 0 {
        TokenStrategy::Recursive
    } else {
        TokenStrategy::Fixed
    };
    let keep_separator = match keep_sep % 3 {
        0 => KeepSeparator::End,
        1 => KeepSeparator::Start,
        _ => KeepSeparator::Discard,
    };
    let cfg = TokenConfig {
        chunk_size: size,
        chunk_overlap: overlap,
        encoding,
        strategy,
        keep_separator,
        drop_whitespace: drop_ws,
    };
    let chars: Vec<char> = text.chars().collect();
    if let Ok(chunks) = chunk_tokens(text, &cfg) {
        for (i, c) in chunks.iter().enumerate() {
            assert_eq!(c.index, i);
            assert!(c.token_count <= size);
            assert!(c.start <= c.end);
            assert!(c.end <= chars.len());
            let slice: String = chars[c.start..c.end].iter().collect();
            assert_eq!(slice, c.text);
        }
    }
});

