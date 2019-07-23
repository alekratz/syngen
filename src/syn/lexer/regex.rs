//! Regex-based lexer.
use crate::{
    common::span::*,
    syn::{
        error::*,
        lexer::{Lexer as BaseLexer, Result},
        token::{Token, TokenKind as BaseTokenKind},
    },
};
use regex::Regex;
use std::{
    collections::BTreeMap,
    fmt::{self, Debug, Display, Formatter},
};

#[derive(Debug)]
pub enum Rule {
    Regex(Regex),
    Name(String),
    And(Vec<Rule>),
    Or(Vec<Rule>),
    Star(Box<Rule>),
    Plus(Box<Rule>),
    Optional(Box<Rule>),
}

impl Rule {
    /// Gets the display text for what this rule expects.
    pub fn expects(&self) -> String {
        match self {
            Rule::Regex(r) => format!("regular expression {}", r.as_str().escape_debug()),
            Rule::Name(n) => format!("{} token", n),
            Rule::And(a) => a[0].expects(),
            Rule::Or(o) => {
                match o.len() {
                    0 => unreachable!(),
                    1 => o[0].expects(),
                    2 => format!("{} or {}", o[0].expects(), o[1].expects()),
                    n => {
                        const MAX_DISPLAY: usize = 6;
                        let take = if n < MAX_DISPLAY {
                            // take all but the last item
                            o.len()
                        } else {
                            // take the first five
                            MAX_DISPLAY
                        } - 1;

                        let commas = o
                            .iter()
                            .take(take)
                            .map(Rule::expects)
                            .collect::<Vec<_>>()
                            .join(", ");
                        format!("{} or {}", commas, o[take].expects())
                    }
                }
            }
            Rule::Star(r) => format!("zero or more of {}", r.expects()),
            Rule::Plus(r) => format!("one or more of {}", r.expects()),
            Rule::Optional(r) => format!("zero or one of {}", r.expects()),
        }
    }

    /// Creates a new `Rule` regular expression rule.
    pub fn regex(r: impl AsRef<str>) -> Self {
        Rule::Regex(Regex::new(r.as_ref()).expect("invalid regex"))
    }

    /// Creates a new `Rule` named rule.
    pub fn name(name: impl ToString) -> Self {
        Rule::Name(name.to_string())
    }

    /// Creates or merges a `Rule` "and" (or "concat") rule.
    ///
    /// If either this rule, or the provided rule, is an "and" rule, the two rules are merged to a
    /// single "and".
    pub fn and(self, other: Rule) -> Self {
        match (self, other) {
            (Rule::And(mut r1), Rule::And(r2)) => {
                r1.extend(r2);
                Rule::And(r1)
            }
            (Rule::And(mut r1), r2) => {
                r1.push(r2);
                Rule::And(r1)
            }
            (r1, Rule::And(mut r2)) => {
                r2.insert(0, r1);
                Rule::And(r2)
            }
            (r1, r2) => Rule::And(vec![r1, r2]),
        }
    }

    /// Creates or merges a `Rule` "or" rule.
    ///
    /// If either this rule, or the provided rule, is an "or" rule, the two rules are merged to a
    /// single "or".
    pub fn or(self, other: Rule) -> Self {
        match (self, other) {
            (Rule::Or(mut r1), Rule::Or(r2)) => {
                r1.extend(r2);
                Rule::Or(r1)
            }
            (Rule::Or(mut r1), r2) => {
                r1.push(r2);
                Rule::Or(r1)
            }
            (r1, Rule::Or(mut r2)) => {
                r2.insert(0, r1);
                Rule::Or(r2)
            }
            (r1, r2) => Rule::Or(vec![r1, r2]),
        }
    }

    /// Converts this rule into a "zero or more" (Kleene Star) rule.
    pub fn star(self) -> Self {
        Rule::Star(self.into())
    }

    /// Converts this rule into a "one more" rule.
    pub fn plus(self) -> Self {
        Rule::Plus(self.into())
    }

    /// Converts this rule into an "optional" rule.
    pub fn optional(self) -> Self {
        Rule::Optional(self.into())
    }

    /// Hack method to ensure that regular expression matches work.
    ///
    /// All regular expression rules need to be anchored to the start of the text, because
    /// otherwise the regular expression will look *anywhere* in the string for the rule. This is
    /// not good eats. This method recursively checks every rule and patches all regular
    /// expressions to start with a "^" character, if they don't already.
    fn patch_regex(&mut self) {
        match self {
            Rule::Regex(r) => {
                let needs_patch = {
                    let r_text = r.as_str();
                    !r_text.starts_with('^')
                };
                if needs_patch {
                    let new_regex = format!("^{}", r.as_str());
                    *r = Regex::new(&new_regex).unwrap();
                }
            }
            Rule::And(rules) | Rule::Or(rules) => for rule in rules.iter_mut() {
                rule.patch_regex();
            },
            | Rule::Star(rule)
            | Rule::Plus(rule)
            | Rule::Optional(rule) => { rule.patch_regex() }
            Rule::Name(_) => {}
        }
    }
}

pub struct GrammarBuilder<T: BaseTokenKind> {
    rules: BTreeMap<String, (Rule, Box<dyn Fn() -> T>)>,
    whitespace_rule: Rule,
    eof_kind: Option<T>,
}

impl<T: BaseTokenKind> Default for GrammarBuilder<T> {
    fn default() -> Self {
        GrammarBuilder {
            rules: Default::default(),
            whitespace_rule: Rule::regex(r"\s+"),
            eof_kind: Default::default(),
        }
    }
}

impl<T: BaseTokenKind> GrammarBuilder<T> {
    /// Adds a new rule with the specified name and token factory.
    pub fn rule<F>(mut self, name: impl ToString, lexer_rule: Rule, token_kind_factory: F) -> Self
    where
        F: Fn() -> T + 'static,
    {
        self.rules
            .insert(name.to_string(), (lexer_rule, Box::new(token_kind_factory)));
        self
    }

    /// Sets the whitespace rule for this grammar.
    pub fn whitespace_rule(mut self, rule: Rule) -> Self {
        self.whitespace_rule = rule;
        self
    }

    /// Sets the EOF token kind.
    pub fn eof_kind(mut self, token_kind: T) -> Self {
        self.eof_kind = Some(token_kind);
        self
    }

    /// Consumes this builder and creates a grammar from it.
    pub fn finish(self) -> Grammar<T> {
        let GrammarBuilder {
            rules,
            whitespace_rule,
            eof_kind,
        } = self;
        Grammar::new(
            rules,
            whitespace_rule,
            eof_kind.expect("eof token kind not specified"),
        )
    }
}

pub struct Grammar<T: BaseTokenKind> {
    rules: BTreeMap<String, (Rule, Box<dyn Fn() -> T>)>,
    whitespace_rule: Rule,
    eof_kind: T,
}

impl<T: BaseTokenKind> Debug for Grammar<T> {
    fn fmt(&self, fmt: &mut Formatter) -> fmt::Result {
        fmt.debug_struct("RegexGrammar")
            .field(
                "rules",
                &self
                    .rules
                    .iter()
                    // no ToString defined for this one
                    .map(|(k, (rule, _))| (k, rule))
                    .collect::<BTreeMap<_, _>>(),
            )
            .field("whitespace_rule", &self.whitespace_rule)
            .field("eof_kind", &self.eof_kind)
            .finish()
    }
}

impl<T: BaseTokenKind> Grammar<T> {
    pub fn new(
        rules: BTreeMap<String, (Rule, Box<dyn Fn() -> T>)>,
        mut whitespace_rule: Rule,
        eof_kind: T,
    ) -> Self {
        whitespace_rule.patch_regex();
        Grammar {
            rules: rules
                .into_iter()
                .map(|(k, (mut rule, factory))| {
                    rule.patch_regex();
                    (k, (rule, factory))
                })
                .collect(),
            whitespace_rule,
            eof_kind,
        }
    }

    pub fn make_lexer(&self, text: impl ToString) -> Lexer<T> {
        Lexer::new(self, text.to_string())
    }

    fn whitespace_rule(&self) -> &Rule {
        &self.whitespace_rule
    }

    fn rule(&self, name: &str) -> &Rule {
        &self
            .rules()
            .get(name)
            .expect(&format!("tried to use undefined lexer rule {:?}", name))
            .0
    }

    fn rules(&self) -> &BTreeMap<String, (Rule, Box<dyn Fn() -> T>)> {
        &self.rules
    }

    fn make_token(&self, token_name: &str, text: String, span: Span) -> Token<T> {
        let (_, ctor) = &self.rules()[token_name];
        Token::new((ctor)(), text, span)
    }
}

#[derive(Debug)]
pub struct Lexer<'grammar, T: BaseTokenKind> {
    grammar: &'grammar Grammar<T>,
    span: Span,
    checkpoints: Vec<(Span, usize)>,
    bytes: usize,
    text: String,
}

impl<'grammar, T: BaseTokenKind> Lexer<'grammar, T> {
    pub fn new(grammar: &'grammar Grammar<T>, text: String) -> Self {
        Lexer {
            grammar,
            span: Default::default(),
            checkpoints: Default::default(),
            bytes: 0,
            text,
        }
    }

    pub fn grammar(&self) -> &'grammar Grammar<T> {
        self.grammar
    }

    /// Creates a checkpoint for backtracking.
    fn checkpoint(&mut self) {
        self.checkpoints.push((self.span, self.bytes));
    }

    /// Removes the last checkpoint from the list.
    fn uncheck(&mut self) {
        self.checkpoints.pop().unwrap();
    }

    /// Return to the previous checkpoint that was set.
    fn backtrack(&mut self) {
        let (span, bytes) = self.checkpoints.pop().unwrap();
        self.span = span;
        self.bytes = bytes;
    }

    /// "Catches up" this span's start point to be equal to the end point.
    fn catchup(&mut self) -> Span {
        let span = self.span;
        self.span.start = self.span.end;
        span
    }

    pub fn match_token(&mut self, name: &str) -> Result<Token<T>> {
        self.checkpoint();
        let rule = self.grammar().rule(name);
        let span = match self.match_rule(rule) {
            Ok(span) => span,
            Err(e) => {
                // Backtrack so we're in a valid state to try to match more tokens
                self.backtrack();
                return Err(e);
            }
        };
        self.uncheck();
        let text = self.text(span);
        Ok(self.grammar().make_token(name, text, span))
    }

    fn match_regex(&mut self, r: &Regex) -> Option<Span> {
        if let Some(m) = r.find_at(&self.text[self.bytes..], 0) {
            assert_eq!(m.start(), 0, "match was not at the start of the string");
            self.bytes += m.as_str().as_bytes().len();
            self.span.end.adv_by(m.as_str().len());
            Some(self.catchup())
        } else {
            None
        }
    }

    fn match_rule(&mut self, rule: &Rule) -> Result<Span> {
        match rule {
            Rule::Regex(r) => self
                .match_regex(r)
                .ok_or_else(|| LexerError::ExpectedGot {
                    span: self.span(),
                    expected: format!("token matching pattern `{:?}`", rule),
                    got: self.c_or_eof(),
                })
                .map(|t| t),
            Rule::Name(name) => {
                let rule = self.grammar().rule(name);
                self.match_rule(rule).map_err(|_| LexerError::ExpectedGot {
                    span: self.span(),
                    expected: format!("{} token", name.to_lowercase()),
                    got: self.c_or_eof(),
                })
            }
            Rule::And(r) => {
                let start = self.span.start;
                let mut end = self.span.end;
                for rule in r.iter() {
                    end = self.match_rule(rule)?.end;
                }
                Ok(Span::new(start, end))
            }
            Rule::Or(r) => {
                let longest_match = r.iter()
                    .filter_map(|r| {
                        self.checkpoint();
                        let span = self.match_rule(rule).map(Some).unwrap_or(None);
                        self.backtrack();
                        span.map(|s| (s, rule))
                    })
                    .max_by_key(|(span, _)| span.len());
                if let Some((_, rule)) = longest_match {
                    let span = self.match_rule(rule).unwrap();
                    Ok(span)
                } else {
                    Err(LexerError::ExpectedGot {
                        span: self.span(),
                        expected: format!("one of {}", rule.expects()),
                        got: self.c_or_eof(),
                    })
                }
            }
            Rule::Star(r) => {
                let start = self.span().start;
                let mut end = self.span().end;
                loop {
                    self.checkpoint();
                    match self.match_rule(r) {
                        Ok(span) => {
                            self.uncheck();
                            end = span.end;
                        }
                        Err(_) => {
                            self.backtrack();
                            break;
                        }
                    }
                }
                Ok(Span::new(start, end))
            }
            Rule::Plus(r) => {
                let mut span = self.match_rule(r)?;
                loop {
                    self.checkpoint();
                    match self.match_rule(r) {
                        Ok(s) => {
                            self.uncheck();
                            span.end = s.end;
                        }
                        // TODO: backtrack
                        Err(_) => {
                            self.backtrack();
                            break;
                        }
                    }
                }
                Ok(span)
            }
            Rule::Optional(r) => {
                self.checkpoint();
                match self.match_rule(r) {
                    Ok(span) => {
                        self.uncheck();
                        Ok(span)
                    }
                    Err(_) => {
                        self.backtrack();
                        Ok(self.catchup())
                    }
                }
            }
        }
    }

    fn c_or_eof(&self) -> String {
        if let Some(c) = self.c() {
            format!("{:?}", c)
        } else {
            "EOF".to_string()
        }
    }

    fn c(&self) -> Option<char> {
        self.text.chars().skip(self.span.end.source).next()
    }

    fn skip_whitespace(&mut self) {
        self.checkpoint();
        while let Ok(_) = self.match_rule(self.grammar().whitespace_rule()) {
            self.uncheck();
            self.checkpoint();
        }
        self.backtrack();
    }
}

impl<T: BaseTokenKind> BaseLexer for Lexer<'_, T> {
    type TokenKind = T;

    fn span(&self) -> Span {
        self.span
    }

    fn next(&mut self) -> Result<Token<Self::TokenKind>> {
        self.skip_whitespace();

        if self.is_eof() {
            self.catchup();
            return Ok(Token::new(
                self.grammar().eof_kind.clone(),
                String::new(),
                self.span(),
            ));
        }

        // TODO: traverse the regex AST with regex-syntax crate to determine lookaheads (?)
        // XXX: for now, go through all rules we know of and try to match those instead. obviously
        //      this is a little risky if you have crazy long potential matches.
        //
        // Also, this is really inefficient because it:
        // 1. checks for *all* matches, backtracking each time
        // 2. singles out the longest match
        // 3. discards the token that was made
        // 4. uses the rule name of the discovered "longest match" to produce the token (with
        //    the intended side effects of the lexer)
        //
        // This could be faster if you feel like keeping up with all lexer side effects, but this
        // is easier and (knock on wood) hopefully temporary for now.
        let longest_match = self
            .grammar()
            .rules()
            .keys()
            .filter_map(|rule_name| {
                self.checkpoint();
                let token = self.match_token(rule_name).map(Some).unwrap_or(None);
                self.backtrack();
                // no dangling checkpoints
                assert_eq!(self.checkpoints.len(), 0);
                token.map(|t| (rule_name, t))
            })
            .max_by_key(|(_, t)| t.text().len());
        if let Some((rule_name, _)) = longest_match {
            let token = self.match_token(rule_name).unwrap();
            Ok(token)
        } else {
            Err(LexerError::Unexpected {
                span: self.span(),
                what: format!(
                    "character {} did not match any rules",
                    self.c_or_eof().escape_debug()
                ),
            })
        }
    }

    fn text(&self, span: Span) -> String {
        self.text
            .chars()
            .skip(span.start.source)
            .take(span.len())
            .collect()
    }

    fn is_eof(&self) -> bool {
        self.c().is_none()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_match_rule() {
        #[derive(Debug, Clone, PartialEq)]
        enum TokenKind {
            A,
            B,
            AB,
            Eof,
        }
        impl BaseTokenKind for TokenKind {}

        let grammar = GrammarBuilder::default()
            .rule("A", Rule::regex("a"), || TokenKind::A)
            .rule("B", Rule::regex("b"), || TokenKind::B)
            .rule("AB", Rule::name("A").and(Rule::name("B")), || TokenKind::AB)
            .eof_kind(TokenKind::Eof)
            .finish();
        let mut lexer = grammar.make_lexer("aabb");

        let token = lexer.next().unwrap();
        assert_eq!(token.kind(), &TokenKind::A);
        let token = lexer.next().unwrap();
        assert_eq!(token.kind(), &TokenKind::AB);
        let token = lexer.next().unwrap();
        assert_eq!(token.kind(), &TokenKind::B);
        assert!(lexer.is_eof());
        let token = lexer.next().unwrap();
        assert_eq!(token.kind(), &TokenKind::Eof);
    }

    #[test]
    fn test_match_numbers() {
        #[derive(Debug, Clone, PartialEq)]
        enum TokenKind {
            Num,
            Eof,
        }
        impl BaseTokenKind for TokenKind {}

        let grammar = GrammarBuilder::default()
            .rule("Num", Rule::regex(r"[0-9]+"), || TokenKind::Num)
            .eof_kind(TokenKind::Eof)
            .finish();

        let mut lexer = grammar.make_lexer(r#"
        1234
        56789
        oopz
        "#);
        let token = lexer.next().unwrap();
        assert_eq!(token.text(), "1234");
        let token = lexer.next().unwrap();
        assert_eq!(token.text(), "56789");
        let token = lexer.next();
        assert!(token.is_err());
    }
}
