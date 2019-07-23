pub trait AstItem: Lookaheads<char> {
    fn span(&self) -> Span;
    fn text(&self) -> &str;
    fn lookaheads() -> &'static [char];
}
