use std::array;

pub struct State<const IDIM: usize> {
    pub bits: Vec<u64>,
}

impl<const IDIM: usize> Default for State<IDIM> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const IDIM: usize> State<IDIM> {
    pub fn new() -> Self {
        let num_words = (IDIM + 63) / 64; // ceiling division
        Self {
            bits: vec![0; num_words],
        }
    }

    /// Creates State from a slice where each u8 represents a single bit (0 or 1)
    pub fn from_bool_slice(a: &[u8]) -> Self {
        assert!(a.len() == IDIM);
        let mut state = State::<IDIM>::default();
        for (i, &e) in a.iter().enumerate() {
            if e == 1 {
                state.set(i, 1);
            }
        }
        state
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        // This ensures bit 0 is always the LSB of the first byte,
        // regardless of the host system's endianness
        let bytes_needed = (IDIM + 7) / 8;
        assert!(bytes.len() >= bytes_needed, "Not enough bits in byte slice");
        
        let num_words = (IDIM + 63) / 64;
        let mut bits = vec![0u64; num_words];
        
        // Always interpret as little-endian for consistent bit ordering
        for i in 0..num_words {
            let start = i * 8;
            let end = ((i + 1) * 8).min(bytes_needed);
            let mut word_bytes = [0u8; 8];
            word_bytes[..(end - start)].copy_from_slice(&bytes[start..end]);
            bits[i] = u64::from_le_bytes(word_bytes);
        }
        
        // Clear extra bits
        if IDIM % 64 != 0 {
            let last_word_bits = IDIM % 64;
            let mask = (1u64 << last_word_bits) - 1;
            bits[num_words - 1] &= mask;
        }
        
        State { bits }
    }

    /// Creates a State from any number of byte slices.
    pub fn from_slices<'a>(slices: &[&'a [u8]]) -> Self {
        let bytes_needed = (IDIM + 7) / 8;
        let total_len: usize = slices.iter().map(|s| s.len()).sum();
        assert!(total_len >= bytes_needed, "Not enough bits in byte slices");

        let num_words = (IDIM + 63) / 64;
        let mut bits = vec![0u64; num_words];

        let mut byte_iter = slices.iter().flat_map(|slice| slice.iter());

        for i in 0..num_words {
            let mut word_bytes = [0u8; 8];
            
            // Pull bytes from the iterator to fill the 8-byte buffer.
            for byte_in_word in word_bytes.iter_mut() {
                let Some(&byte) = byte_iter.next() else {
                    break;
                };
                *byte_in_word = byte;
            }
            
            bits[i] = u64::from_le_bytes(word_bytes);
        }

        if IDIM % 64 != 0 {
            let last_word_bits = IDIM % 64;
            let mask = (1u64 << last_word_bits) - 1;
            bits[num_words - 1] &= mask;
        }

        State { bits }
    }

    /// Gets the value (0 or 1) of the i-th bit in the state vector.
    pub fn get(&self, i: usize) -> u8 {
        debug_assert!(i < IDIM);
        let word_index = i / 64;
        let bit_offset = i % 64;
        ((self.bits[word_index] >> bit_offset) & 1) as u8
    }

    /// Sets the value of the i-th bit in the state vector.
    pub fn set(&mut self, i: usize, value: u8) {
        debug_assert!(i < IDIM);
        let word_index = i / 64;
        let bit_offset = i % 64;

        if value == 1 {
            // Set the bit: OR with a 1 at the bit_offset position
            self.bits[word_index] |= 1u64 << bit_offset;
        } else {
            // Clear the bit: AND with a mask that has a 0 at the bit_offset position
            self.bits[word_index] &= !(1u64 << bit_offset);
        }
    }

    pub fn to_u8_array(&self) -> [u8; IDIM] {
        array::from_fn(|i| self.get(i))
    }
}

#[cfg(test)]
mod tests {
    use super::State;

    // We only need the bounds on the function if we use a size-dependent type inside.
    fn create_state<const N: usize>() -> State<N> {
        State::<N>::default()
    }

    #[test]
    fn state_get_set_test() {
        let mut state = create_state::<100>();

        // Initial state is all 0s
        assert_eq!(state.get(5), 0);
        assert_eq!(state.get(64), 0); // Crosses word boundary

        state.set(5, 1);
        state.set(64, 1);

        assert_eq!(state.get(5), 1);
        assert_eq!(state.get(64), 1);
        assert_eq!(state.get(6), 0);

        state.set(5, 0);
        assert_eq!(state.get(5), 0);
        assert_eq!(state.get(64), 1);

        // Check underlying storage
        // Bit 5 is (1 << 5) = 32. Bit 64 is (1 << 0) in the second word.
        // The check was faulty; u64 << 64 is 0.
        // Word 0 should be !(1u64 << 5) in a system where IDIM=100.
        // Correct check: bits[0] should be 0 (since 5 was set and cleared)
        assert_eq!(
            state.bits[0], 0,
            "Word 0 should be 0 after set(5,1) and set(5,0)"
        );
        assert_eq!(
            state.bits[1], 1,
            "Word 1 should have only bit 0 (index 64) set"
        );
    }

    #[test]
    fn from_u8_slice_test() {
        // State: [1, 0, 1, 0, 0, 0, 1, 1]
        // Indices: 0, 2, 6, 7 are set.
        // u64 value: 2^0 + 2^2 + 2^6 + 2^7 = 1 + 4 + 64 + 128 = 197
        let state = State::<8>::from_bool_slice(&[1, 0, 1, 0, 0, 0, 1, 1]);
        assert_eq!(state.get(0), 1);
        assert_eq!(state.get(1), 0);
        assert_eq!(state.get(2), 1);
        assert_eq!(state.get(6), 1);
        assert_eq!(state.get(7), 1);
        assert_eq!(state.bits[0], 197);
    }
}
