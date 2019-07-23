use std::fmt::Debug;
use crate::{
    common::span::*,
};

pub trait TokenKind: Debug + Clone + Sized { }

#[derive(Debug, Clone)]
pub struct Token<Kind: TokenKind> {
    kind: Kind,
    text: String,
    span: Span,
}

impl<Kind: TokenKind> Token<Kind> {
    pub fn new(kind: Kind, text: String, span: Span) -> Self {
        Token { kind, text, span, }
    }

    pub fn kind(&self) -> &Kind { &self.kind }

    pub fn text(&self) -> &str { &self.text }

    pub fn span(&self) -> Span { self.span }
}
