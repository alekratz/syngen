pub mod regex;

use crate::{
    common::span::*,
    syn::{error::*, token::*},
};
use std::fmt::Debug;

pub type Result<T> = std::result::Result<T, LexerError>;

pub trait Lexer {
    type TokenKind: TokenKind;
    fn span(&self) -> Span;
    fn next(&mut self) -> Result<Token<Self::TokenKind>>;
    fn text(&self, span: Span) -> String;
    fn is_eof(&self) -> bool;
}
