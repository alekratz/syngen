use std::cmp::Ordering;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Pos {
    pub source: usize,
    pub line: usize,
}

impl Pos {
    pub fn adv(&mut self) {
        self.adv_by(1)
    }

    pub fn adv_by(&mut self, count: usize) {
        self.source += count;
    }

    pub fn line(&mut self) {
        self.line += 1;
    }
}


impl PartialOrd for Pos {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Pos {
    fn cmp(&self, other: &Self) -> Ordering {
        self.source.cmp(&other.source)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Span {
    pub start: Pos,
    pub end: Pos,
}

impl Span {
    pub fn new(start: Pos, end: Pos) -> Self {
        Span { start, end, }
    }

    pub fn len(&self) -> usize {
        self.end.source - self.start.source
    }
}
