#![no_main]
//! Feeds arbitrary text and parameters to every strategy and checks the invariants
//! the library promises: no panic, chunks within `chunk_size`, offsets map back to the source.

use chunkwise::core::{chunk, Config, Strategy};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|input: (&str, u8, u8, u8)| {
    let (text, size, overlap, strat) = input;
    let size = size as usize % 200 + 1;
    let overlap = overlap as usize % size;
    let strategy = match strat % 3 {
        0 => Strategy::Fixed,
        1 => Strategy::Sentence,
        _ => Strategy::Recursive,
    };
    let cfg = Config { chunk_size: size, chunk_overlap: overlap, strategy };
    let chars: Vec<char> = text.chars().collect();
    let chunks = chunk(text, &cfg).expect("valid config must not fail");
    for (i, c) in chunks.iter().enumerate() {
        assert_eq!(c.index, i);
        assert!(c.text.chars().count() <= size);
        let slice: String = chars[c.start..c.end].iter().collect();
        assert_eq!(slice, c.text);
    }
});
