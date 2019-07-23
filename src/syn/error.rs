use crate::common::span::Span;
use snafu::Snafu;

#[derive(Debug, Snafu)]
pub enum LexerError {
    #[snafu(display("expected {}, but got {} instead", expected, got))]
    ExpectedGot {
        span: Span,
        expected: String,
        got: String,
    },

    #[snafu(display("unexpected {}", what))]
    Unexpected {
        span: Span,
        what: String,
    },

    #[snafu(display("unknown {}", what))]
    Unknown {
        span: Span,
        what: String,
    },
}
