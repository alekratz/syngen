use crate::syn::lexer::Lexer;

pub trait Parser {
    type Ast;
    type Lexer: Lexer;


}
