// Copyright 2025 RISC Zero, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use risc0_binfmt::WordAddr;

const ENTRY_COUNT: usize = 1 << 11;
const PAGED_MAP_MASK: u32 = (1 << 10) - 1;

struct PagedMapEntry {
    words: Vec<Option<u32>>,
}

impl Default for PagedMapEntry {
    fn default() -> Self {
        Self {
            words: vec![None; ENTRY_COUNT],
        }
    }
}

struct PagedMapTable {
    entries: Vec<u32>,
}

impl Default for PagedMapTable {
    fn default() -> Self {
        Self {
            entries: vec![0; ENTRY_COUNT],
        }
    }
}

pub(crate) struct PagedMap {
    high: PagedMapTable,
    mid: Vec<PagedMapTable>,
    low: Vec<PagedMapEntry>,
}

impl Default for PagedMap {
    fn default() -> Self {
        let mut mid = Vec::with_capacity(16);
        let mut low = Vec::with_capacity(64);

        mid.push(PagedMapTable { entries: Vec::new() });
        low.push(PagedMapEntry { words: Vec::new() });

        Self {
            high: PagedMapTable::default(),
            mid,
            low,
        }
    }
}

impl PagedMap {
    /// Returns the value corresponding to the key.
    #[inline(always)]
    pub fn get(&mut self, addr: &WordAddr) -> Option<u32> {
        let idx = addr.0 >> 20;

        if idx as usize >= self.high.entries.len() {
            return None;
        }

        let mid = unsafe { self.high.entries.get_unchecked(idx as usize) };
        if *mid != 0 {
            let mid_table = unsafe { self.mid.get_unchecked(*mid as usize) };
            let idx = (addr.0 >> 10) & PAGED_MAP_MASK;

            if (idx as usize) < mid_table.entries.len() {
                let low = unsafe { mid_table.entries.get_unchecked(idx as usize) };
                if *low != 0 {
                    let low_table = unsafe { self.low.get_unchecked(*low as usize) };
                    let idx = addr.0 & PAGED_MAP_MASK;
                    if (idx as usize) < low_table.words.len() {
                        return unsafe { *low_table.words.get_unchecked(idx as usize) };
                    }
                }
            }
        }
        None
    }

    /// Returns a mutable reference to the value corresponding to the key.
    #[inline(always)]
    pub fn get_mut(&mut self, addr: &WordAddr) -> &mut Option<u32> {
        let high_idx = addr.0 >> 20;
        let mid_idx = (addr.0 >> 10) & PAGED_MAP_MASK;
        let low_idx = addr.0 & PAGED_MAP_MASK;

        if (high_idx as usize) >= self.high.entries.len() {
            self.high.entries.resize((high_idx as usize) + 1, 0);
        }

        let mid = unsafe { self.high.entries.get_unchecked_mut(high_idx as usize) };
        if *mid == 0 {
            *mid = self.mid.len() as u32;
            self.mid.push(PagedMapTable::default());
        }

        let mid_table = unsafe { self.mid.get_unchecked_mut(*mid as usize) };

        if (mid_idx as usize) >= mid_table.entries.len() {
            mid_table.entries.resize((mid_idx as usize) + 1, 0);
        }

        let low = unsafe { mid_table.entries.get_unchecked_mut(mid_idx as usize) };
        if *low == 0 {
            *low = self.low.len() as u32;
            self.low.push(PagedMapEntry::default());
        }

        let low_table = unsafe { self.low.get_unchecked_mut(*low as usize) };

        if (low_idx as usize) >= low_table.words.len() {
            low_table.words.resize((low_idx as usize) + 1, None);
        }

        unsafe { low_table.words.get_unchecked_mut(low_idx as usize) }
    }

    /// Inserts a key-value pair into the map.
    ///
    /// If the map did not have this key present, None is returned.
    ///
    /// If the map did have this key present, the value is updated, and the old value is returned.
    #[inline(always)]
    pub fn insert(&mut self, addr: &WordAddr, word: u32) -> Option<u32> {
        std::mem::replace(self.get_mut(addr), Some(word))
    }

    #[inline(always)]
    pub fn insert_default(&mut self, addr: &WordAddr, word: u32, default: u32) -> u32 {
        self.insert(addr, word).unwrap_or(default)
    }
}
