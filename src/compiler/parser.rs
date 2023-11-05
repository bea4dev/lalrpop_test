use std::ops::Range;

use ariadne::{Fmt, Color, Report, Label, Source};
use either::Either;
use lalrpop_util::{lalrpop_mod, ErrorRecovery};


#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind<'input> {
    Function,
    Static,
    Private,
    Suspend,
    Native,
    Uncycle,
    Open,
    New,
    Drop,
    Mutex,
    ParenthesisLeft,
    ParenthesisRight,
    BraceLeft,
    BraceRight,
    BracketLeft,
    BracketRight,
    Class,
    Struct,
    Interface,
    Extends,
    Implements,
    Import,
    DoubleColon,
    Comma,
    Loop,
    While,
    Var,
    Let,
    Equal,
    Exchange,
    Or,
    And,
    EqEqual,
    NotEqual,
    GreaterThan,
    GreaterOrEq,
    LessThan,
    LessOrEq,
    Plus,
    Minus,
    Star,
    Slash,
    VerticalBar,
    Dot,
    Null,
    InterrogationMark,
    ExclamationMark,
    InterrogationElvis,
    ExclamationElvis,
    Colon,
    If,
    Else,
    Return,
    FatArrow,
    ThinArrow,
    Literal(&'input str),
    Semicolon,
    LineFeed,
    Whitespace,
    UnexpectedCharactor(&'input str)
}

enum TokenizerKind<'input> {
    Keyword(TokenKind<'input>, &'static str),
    Functional(fn(current_input: &'input str) -> (usize, TokenKind<'input>))
}

impl<'input> TokenizerKind<'input> {
    fn tokenize(&self, current_input: &'input str) -> (usize, TokenKind<'input>) {
        return match self {
            TokenizerKind::Keyword(kind, keyword) => {
                let mut input_chars = current_input.chars();
                let mut keyword_chars = keyword.chars();
                let mut current_byte_length = 0;
                loop {
                    let keyword_char = match keyword_chars.next(){
                        Some(c) => c,
                        _ => break
                    };
                    let current_char = match input_chars.next(){
                        Some(c) => c,
                        _ => return (0, kind.clone()) // reject
                    };
                    if current_char != keyword_char {
                        return (0, kind.clone()) // reject
                    }

                    current_byte_length += current_char.len_utf8();
                }
                (current_byte_length, kind.clone()) // accept
            },
            TokenizerKind::Functional(tokenizer) => {
                tokenizer(current_input)
            },
        }
    }
}

#[inline(always)]
fn tokenizers<'input>() -> [TokenizerKind<'input>; 60] {
    [
        TokenizerKind::Keyword(TokenKind::Function, "function"),
        TokenizerKind::Keyword(TokenKind::Static, "static"),
        TokenizerKind::Keyword(TokenKind::Private, "private"),
        TokenizerKind::Keyword(TokenKind::Suspend, "suspend"),
        TokenizerKind::Keyword(TokenKind::Native, "native"),
        TokenizerKind::Keyword(TokenKind::Uncycle, "uncycle"),
        TokenizerKind::Keyword(TokenKind::Open, "open"),
        TokenizerKind::Keyword(TokenKind::New, "new"),
        TokenizerKind::Keyword(TokenKind::Drop, "drop"),
        TokenizerKind::Keyword(TokenKind::Mutex, "mutex"),
        TokenizerKind::Keyword(TokenKind::ParenthesisLeft, "("),
        TokenizerKind::Keyword(TokenKind::ParenthesisRight, ")"),
        TokenizerKind::Keyword(TokenKind::BraceLeft, "{"),
        TokenizerKind::Keyword(TokenKind::BraceRight, "}"),
        TokenizerKind::Keyword(TokenKind::BracketLeft, "["),
        TokenizerKind::Keyword(TokenKind::BracketRight, "]"),
        TokenizerKind::Keyword(TokenKind::Class, "class"),
        TokenizerKind::Keyword(TokenKind::Struct, "struct"),
        TokenizerKind::Keyword(TokenKind::Interface, "interface"),
        TokenizerKind::Keyword(TokenKind::Extends, "extends"),
        TokenizerKind::Keyword(TokenKind::Implements, "implements"),
        TokenizerKind::Keyword(TokenKind::Import, "import"),
        TokenizerKind::Keyword(TokenKind::DoubleColon, "::"),
        TokenizerKind::Keyword(TokenKind::Comma, ","),
        TokenizerKind::Keyword(TokenKind::Loop, "loop"),
        TokenizerKind::Keyword(TokenKind::Var, "var"),
        TokenizerKind::Keyword(TokenKind::Let, "let"),
        TokenizerKind::Keyword(TokenKind::Equal, "="),
        TokenizerKind::Keyword(TokenKind::Exchange, "<=>"),
        TokenizerKind::Keyword(TokenKind::Or, "or"),
        TokenizerKind::Keyword(TokenKind::And, "and"),
        TokenizerKind::Keyword(TokenKind::EqEqual, "=="),
        TokenizerKind::Keyword(TokenKind::NotEqual, "=/"),
        TokenizerKind::Keyword(TokenKind::GreaterThan, ">"),
        TokenizerKind::Keyword(TokenKind::GreaterOrEq, ">="),
        TokenizerKind::Keyword(TokenKind::LessThan, "<"),
        TokenizerKind::Keyword(TokenKind::LessOrEq, "<="),
        TokenizerKind::Keyword(TokenKind::Plus, "+"),
        TokenizerKind::Keyword(TokenKind::Minus, "-"),
        TokenizerKind::Keyword(TokenKind::Star, "*"),
        TokenizerKind::Keyword(TokenKind::Slash, "/"),
        TokenizerKind::Keyword(TokenKind::VerticalBar, "|"),
        TokenizerKind::Keyword(TokenKind::Dot, "."),
        TokenizerKind::Keyword(TokenKind::Null, "null"),
        TokenizerKind::Keyword(TokenKind::InterrogationMark, "?"),
        TokenizerKind::Keyword(TokenKind::ExclamationMark, "!"),
        TokenizerKind::Keyword(TokenKind::InterrogationElvis, "?:"),
        TokenizerKind::Keyword(TokenKind::ExclamationElvis, "!:"),
        TokenizerKind::Keyword(TokenKind::Colon, ":"),
        TokenizerKind::Keyword(TokenKind::If, "if"),
        TokenizerKind::Keyword(TokenKind::Else, "else"),
        TokenizerKind::Keyword(TokenKind::Return, "return"),
        TokenizerKind::Keyword(TokenKind::FatArrow, "=>"),
        TokenizerKind::Keyword(TokenKind::ThinArrow, "->"),
        TokenizerKind::Functional(literal_tokenizer),
        TokenizerKind::Keyword(TokenKind::Semicolon, ";"),
        TokenizerKind::Keyword(TokenKind::LineFeed, "\r"),
        TokenizerKind::Keyword(TokenKind::LineFeed, "\n"),
        TokenizerKind::Keyword(TokenKind::LineFeed, "\r\n"),
        TokenizerKind::Functional(whitespace_tokenizer)
    ]
}

fn literal_tokenizer<'input>(current_input: &'input str) -> (usize, TokenKind<'input>) {
    let mut input_chars = current_input.chars();
    let mut current_byte_length = 0;
    loop {
        let current_char = match input_chars.next() {
            Some(c) => c,
            _ => break
        };
        if !(current_char == '_' || current_char.is_alphanumeric()) {
            break;
        }
        current_byte_length += current_char.len_utf8();
    }

    return (current_byte_length, TokenKind::Literal(&current_input[0..current_byte_length]));
}

fn whitespace_tokenizer<'input>(current_input: &'input str) -> (usize, TokenKind<'input>) {
    let mut input_chars = current_input.chars();
    let mut current_byte_length = 0;
    loop {
        let current_char = match input_chars.next() {
            Some(c) => c,
            _ => break
        };
        if !(current_char != '\n' && current_char != '\r' && current_char.is_whitespace()) {
            break;
        }
        current_byte_length += current_char.len_utf8();
    }

    return (current_byte_length, TokenKind::Whitespace);
}

pub struct Lexer<'input> {
    source: &'input str,
    current_byte_position: usize,
    tokenizers: [TokenizerKind<'input>; 60]
}

impl<'input> Lexer<'input> {
    pub fn new(source: &'input str) -> Lexer<'input> {
        return Self {
            source,
            current_byte_position: 0,
            tokenizers: tokenizers(),
        }
    }
}

impl<'input> Iterator for Lexer<'input> {
    type Item = Result<(usize, TokenKind<'input>, usize), ()>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_byte_position == self.source.len() {
                return None;
            }

            let current_input = &self.source[self.current_byte_position..self.source.len()];

            let mut current_max_length = 0;
            let mut current_token_kind = TokenKind::Whitespace;
            for tokenizer in self.tokenizers.iter() {
                let result = tokenizer.tokenize(current_input);
                let byte_length = result.0;
                let token_kind = result.1;
                
                if byte_length > current_max_length {
                    current_max_length = byte_length;
                    current_token_kind = token_kind;
                }
            }

            let start_position = self.current_byte_position;

            let result = if current_max_length == 0 {
                self.current_byte_position += 1;
                let end_position = start_position + 1;
                Ok((start_position, TokenKind::UnexpectedCharactor(&self.source[start_position..end_position]), end_position))
            } else {
                self.current_byte_position += current_max_length;
                
                if let TokenKind::Whitespace = current_token_kind {
                    continue;
                }

                let end_posiiotn = self.current_byte_position;
                Ok((start_position, current_token_kind, end_posiiotn))
            };
            
            return Some(result);
        }
    }
}




#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned<T> {
    pub value: T,
    pub span: Range<usize>
}

impl<T> Spanned<T> {
    pub fn new(value: T, span: Range<usize>) -> Self {
        return Self {
            value,
            span
        }
    }
}

pub struct StringToken(pub usize, pub String);

pub type ASTParseError<'input> = ErrorRecovery<usize, TokenKind<'input>, ()>;

pub type ParseResult<'input, T> = Result<T, ASTParseError<'input>>;

pub type Program<'input> = Box<ProgramAST<'input>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramAST<'input> {
    pub statements: Vec<Statement<'input>>,
    pub span: Range<usize>
}

pub type Statement<'input> = ParseResult<'input, StatementAST<'input>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatementAST<'input> {
    Assignment(Assignment<'input>),
    Exchange(Exchange<'input>),
    Import(Import),
    VariableDefine(VariableDefine<'input>),
    Loop(Loop<'input>),
    FunctionDefine(FunctionDefine<'input>),
    DataStructDefine(DataStructDefine<'input>),
    DropStatement(DropStatement<'input>),
    Expression(Expression<'input>)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionDefine<'input> {
    pub attributes: Vec<StatementAttribute>,
    pub generics: Option<Generics>,
    pub name: Either<Identifier, MemoryManageAttribute>,
    pub args: Vec<FunctionArgument>,
    pub type_tag: Option<TypeTag>,
    pub block: Block<'input>,
    pub span: Range<usize>
}

pub type StatementAttribute = Spanned<StatementAttributeEnum>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StatementAttributeEnum {
    Static,
    Private,
    Suspend,
    Native,
    Uncycle,
    Open
}

pub type MemoryManageAttribute = Spanned<MemoryManageAttributeEnum>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoryManageAttributeEnum {
    New,
    Drop,
    Mutex
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionArgument {
    pub name: Identifier,
    pub type_tag: TypeTag,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DataStructDefine<'input> {
    pub attributes: Vec<StatementAttribute>,
    pub kind: DataStructKind,
    pub name: Identifier,
    pub generics: Option<Generics>,
    pub extends: Option<Extends>,
    pub implements: Option<Implements>,
    pub block: Block<'input>,
    pub span: Range<usize>
}

pub type DataStructKind = Spanned<DataStructKindEnum>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataStructKindEnum {
    Class,
    Struct,
    Interface
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Extends {
    pub type_info: TypeInfo,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Implements {
    pub type_infos: Vec<TypeInfo>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Import {
    pub import_path: Vec<Identifier>,
    pub elements: Vec<Identifier>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropStatement<'input> {
    pub uncycle_keyword_span: Option<Range<usize>>,
    pub expression: Expression<'input>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Loop<'input> {
    pub block: Block<'input>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Block<'input> {
    pub program: Program<'input>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariableDefine<'input> {
    pub attributes: VariableAttributes,
    pub name: Identifier,
    pub type_tag: Option<TypeTag>,
    pub expression: Option<Expression<'input>>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariableAttributes {
    pub statement_attributes: Vec<StatementAttribute>,
    pub is_var: bool,
    pub var_let_span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment<'input> {
    pub left_expr: Expression<'input>,
    pub right_expr: Expression<'input>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exchange<'input> {
    pub left_expr: Expression<'input>,
    pub right_expr: Expression<'input>,
    pub span: Range<usize>
}

pub type Expression<'input> = Box<ExpressionEnum<'input>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExpressionEnum<'input> {
    OrExpression(OrExpression<'input>),
    IfExpression(IfExpression<'input>),
    ReturnExpression(ReturnExpression<'input>),
    Closure(Closure<'input>)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrExpression<'input> {
    pub left_expr: AndExpression<'input>,
    pub right_exprs: Vec<(OrOperatorSpan, AndExpression<'input>)>,
    pub span: Range<usize>
}

type OrOperatorSpan = Range<usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AndExpression<'input> {
    pub left_expr: EQNEExpression<'input>,
    pub right_exprs: Vec<(AndOperatorSpan, EQNEExpression<'input>)>,
    pub span: Range<usize>
}

type AndOperatorSpan = Range<usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EQNEExpression<'input> {
    pub left_expr: CompareExpression<'input>,
    pub right_exprs: Vec<(EQNECompareKind, CompareExpression<'input>)>,
    pub span: Range<usize>
}

pub type EQNECompareKind = Spanned<EQNECompareKindEnum>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EQNECompareKindEnum {
    Equal,
    NotEqual
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompareExpression<'input> {
    pub left_expr: AddOrSubExpression<'input>,
    pub right_exprs: Vec<(CompareKind, AddOrSubExpression<'input>)>,
    pub span: Range<usize>
}

pub type CompareKind = Spanned<CompareKindEnum>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompareKindEnum {
    GreaterThan,
    GreaterOrEqual,
    LessThan,
    LessOrEqual
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AddOrSubExpression<'input> {
    pub left_expr: MulOrDivExpression<'input>,
    pub right_exprs: Vec<(AddOrSubOp, MulOrDivExpression<'input>)>,
    pub span: Range<usize>
}

pub type AddOrSubOp = Spanned<AddOrSubOpEnum>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AddOrSubOpEnum {
    Add,
    Sub
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MulOrDivExpression<'input> {
    pub left_expr: Factor<'input>,
    pub right_exprs: Vec<(MulOrDivOp, Factor<'input>)>,
    pub span: Range<usize>
}

pub type MulOrDivOp = Spanned<MulOrDivOpEnum>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MulOrDivOpEnum {
    Mul,
    Div
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Factor<'input> {
    pub negative_keyword_span: Option<Range<usize>>,
    pub primary: Primary<'input>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Primary<'input> {
    pub first: PrimaryFirst<'input>,
    pub chain: Vec<PrimarySecond<'input>>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimaryFirst<'input> {
    pub first_expr: Either<(SimplePrimary<'input>, Option<FunctionCall<'input>>), NewExpression<'input>>,
    pub mapping_operator: Option<MappingOperator<'input>>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimarySecond<'input> {
    pub separator_kind: PrimarySeparatorKind,
    pub second_expr: Option<(Identifier, Option<FunctionCall<'input>>)>,
    pub mapping_operator: Option<MappingOperator<'input>>,
    pub span: Range<usize>
}

pub type PrimarySeparatorKind = Spanned<PrimarySeparatorKindEnum>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrimarySeparatorKindEnum {
    Dot,
    DoubleColon
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SimplePrimary<'input> {
    Expression(Expression<'input>),
    Identifier(Identifier),
    NullKeyword(Range<usize>)
}

pub type MappingOperator<'input> = Spanned<MappingOperatorEnum<'input>>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MappingOperatorEnum<'input> {
    NullPropagation,
    NullUnwrap,
    NullElvisBlock(Block<'input>),
    ResultPropagation,
    ResultUnwrap,
    ResultElvisBlock(Block<'input>)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunctionCall<'input> {
    pub generics: Option<Generics>,
    pub arguments: Vec<Expression<'input>>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NewExpression<'input> {
    pub new_keyword_span: Range<usize>,
    pub uncycle_keyword_span: Option<Range<usize>>,
    pub path: Vec<Identifier>,
    pub function_call: FunctionCall<'input>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IfExpression<'input> {
    pub if_statement: IfStatement<'input>,
    pub chain: Vec<Either<IfStatement<'input>, Block<'input>>>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IfStatement<'input> {
    pub condition: Expression<'input>,
    pub block: Block<'input>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReturnExpression<'input> {
    pub return_keyword_span: Range<usize>,
    pub expression: Option<Expression<'input>>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Closure<'input> {
    pub arguments: ClosureArguments,
    pub block: Either<Expression<'input>, Block<'input>>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClosureArguments {
    pub arguments: Vec<Either<Identifier, FunctionArgument>>,
    pub span: Range<usize>
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeTag {
    pub tag_kind: TypeTagKind,
    pub type_info: TypeInfo,
    pub span: Range<usize>
}

pub type TypeTagKind = Spanned<TypeTagKindEnum>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeTagKindEnum {
    Normal,
    ReturnType
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeInfo {
    pub name: Identifier,
    pub generics: Option<Generics>,
    pub type_attributes: Vec<TypeAttribute>,
    pub span: Range<usize>
}

pub type TypeAttribute = Spanned<TypeAttributeEnum>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeAttributeEnum {
    Optional,
    Result(Option<Generics>)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Generics {
    pub elements: Vec<TypeInfo>,
    pub span: Range<usize>
}

pub type Identifier = Spanned<String>;

pub type EndOfStatement = Spanned<EndOfStatementEnum>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EndOfStatementEnum {
    LineFeed,
    Semicolon
}

/*
    source              ::= program

    program             ::= [ statement ] { end_of_line [ statement ] }
    statement           ::= assignment | exchange_statement | import_statement | variable_define |
                            loop_statement | define_with_attr | drop_statement | expression

    define_with_attr    ::= statement_attribute ( function_define | data_struct_define )

    function_define     ::= "function" [ generics_info ] ( identifier | memory_manage_attr ) function_arguments [ function_type_tag ] block
    function_arguments  ::= "(" [ function_argument | "const" "this" ] { "," [ function_argument ] } ")"
    function_argument   ::= identifier type_tag

    memory_manage_attr  ::= "new" | "drop" | "mutex"

    statement_attribute ::= { "export" | "private" | "suspend" | "native" | "uncycle" | "open" }

    data_struct_define  ::= ( "class" | "struct" | "interface" ) identifier [ generics_info ] [ extends_info ] [ implements_info ] block
    extends_info        ::= "extends" type_info
    implements_info     ::= "implements" type_info { "," [ type_info ] }

    import_statement    ::= "import" identifier { "::" ( identifier | import_elements ) }
    import_elements     ::= "{" [ identifier ] { "," [ identifier ] } "}"

    drop_statement      ::= "drop" [ "uncycle" ] expression

    loop_statement      ::= "loop" block

    block               ::= "{" program "}"

    variable_define     ::= variable_attributes identifier [ type_tag ] [ "=" expression ]
    variable_attributes ::= [ "export" ] ( "const" | "var" )

    assignment          ::= expression "=" expression

    exchange_statement  ::= expression "<=>" expression

    if_expression       ::= if_statement { "else" ( if_statement | block ) }
    if_statement        ::= "if" expression block

    expression          ::= or_expr | if_expression | return_expression | function_expr
    or_expr             ::= and_expr { "||" and_expr }
    and_expr            ::= equ_or_ine_expr { "&&" equ_or_ine_expr }
    equ_or_ine_expr     ::= les_or_gre_expr { ( "==" | "!=" ) les_or_gre_expr }
    les_or_gre_expr     ::= add_or_sub_expr { ( "<" | ">" | "<=" | ">=" ) add_or_sub_expr }
    add_or_sub_expr     ::= mul_or_div_expr { ( "+" | "-" ) mul_or_div_expr }
    mul_or_div_expr     ::= factor { ( "*" | "/" ) factor }
    factor              ::= "-" primary | primary
    primary             ::= ( simple_primary [ function_call ] | new_expression ) [ mapping_operator ]
                            { ( "." | "::" ) [ identifier [ function_call ] ] [ mapping_operator ] }
    simple_primary      ::= "(" expression ")" | identifier | "null"
    mapping_operator    ::= "?" | "?" "!" | "!" | "!" "!" | ( "?" | "!" ) ":" block

    function_expr       ::= ( function_expr_args | identifier ) "=>" ( expression | block )
    function_expr_args  ::= "|" [ function_argument ] { "," [ function_argument ] } "|"

    function_call       ::= [ ":" generics_info ] "(" [ expression ] { "," [ expression ] } ")"

    new_expression      ::= "new" [ "uncycle" ] identifier { "::" identifier } function_call

    return_expression   ::= "return" [ expression ]

    type_tag            ::= ":" type_info
    function_type_tag   ::= "->" type_info
    type_info           ::= identifier [ generics_info ] { type_attribute }
    type_attribute      ::= "?" | result_type_attr
    result_type_attr    ::= "!" [ generics_info ]
    generics_info       ::= "<" [ type_info ] { "," [ type_info ] } ">"

    identifier          ::= r"[#\w]+" | "this"
    end_of_line         ::= r"\n+" | ";"
*/


pub fn get_unicode_span(source: &str, byte_span: Range<usize>) -> Range<usize> {
    let mut current_position_unicode = 0;
    let mut current_position_bytes = 0;

    let mut unicode_span = 0..0;

    for c in source.chars() {
        if byte_span.start >= current_position_bytes {
            unicode_span.start = current_position_unicode;
        }

        current_position_unicode += 1;
        current_position_bytes += c.len_utf8();

        if byte_span.end >= current_position_bytes {
            unicode_span.end = current_position_unicode;
        }
    }

    return unicode_span;
}

pub fn build_expected_string(expected: &Vec<String>) -> String {
    let mut message = String::new();
    
    if !expected.is_empty() {
        message += " Expected: ";

        let mut i = 0;
        for expected in expected {
            if i != 0 {
                if i + 1 == expected.len() {
                    message += " or "
                } else {
                    message += " , "
                }
            }
            message += format!("{}", expected.fg(Color::Cyan)).as_str();
            i += 1;
        }
    }

    return message;
}

lalrpop_mod!(pub calculator1);

#[test]
pub fn parse() {
    let source = "
let a: int = 0
static private let b = 200@

class TestClass {
    var field0: TestClass? = null
    var field1: TestClass!?!<T>? = null

    function new() {
        this.field0 = null
        this.field1 = null
    }

    static private function test() -> TestClass {
        return new TestClass()
    }
}
";
    let lexer = Lexer::new(source);
    //dbg!(lexer.collect::<Vec<_>>());
    //return;

    let mut errors = Vec::<_>::new();
    let result: Result<Program<'_>, lalrpop_util::ParseError<usize, TokenKind<'_>, ()>> = calculator1::ProgramParser::new().parse(&mut errors, lexer);
    
    dbg!(&result);
    println!("{:?}", &errors);

    let src_id = "test";

    for error in errors {
        let report_builder = match error.error {
            lalrpop_util::ParseError::InvalidToken { location } => {
                let unicode_span = get_unicode_span(source, location..location+1);

                Report::build(ariadne::ReportKind::Error, src_id, unicode_span.start)
                    .with_code(0)
                    .with_message("Invalid token.")
                    .with_label(
                        Label::new((src_id, unicode_span))
                            .with_message("This is an invalid token in this context.")
                            .with_color(Color::Red)
                    )
                    .finish()
            },
            lalrpop_util::ParseError::UnrecognizedEof { location, expected } => {
                let label_message = "The grammar at the end of the source code is incomplete.".to_string();
                let unicode_span = get_unicode_span(source, location..location+1);

                Report::build(ariadne::ReportKind::Error, src_id, location)
                    .with_code(1)
                    .with_message("Unrecognized EOF.")
                    .with_label(
                        Label::new((src_id, unicode_span))
                            .with_message(label_message + build_expected_string(&expected).as_str())
                            .with_color(Color::Red)
                    )
                    .finish()
            },
            lalrpop_util::ParseError::UnrecognizedToken { token, expected } => {
                let label_message = "This token is unexpected.".to_string();
                let unicode_span = get_unicode_span(source, token.0..token.2);

                Report::build(ariadne::ReportKind::Error, src_id, unicode_span.start)
                    .with_code(2)
                    .with_message("Unexpected token.")
                    .with_label(
                        Label::new((src_id, unicode_span))
                            .with_message(label_message + build_expected_string(&expected).as_str())
                            .with_color(Color::Red)
                    )
                    .finish()
            },
            lalrpop_util::ParseError::ExtraToken { token } => {
                let label_message = "This token is unneeded.";
                let unicode_span = get_unicode_span(source, token.0..token.2);

                Report::build(ariadne::ReportKind::Error, src_id, unicode_span.start)
                    .with_code(3)
                    .with_message("Extra token.")
                    .with_label(
                        Label::new((src_id, unicode_span))
                            .with_message(label_message)
                            .with_color(Color::Red)
                    )
                    .finish()
            },
            lalrpop_util::ParseError::User { error: _ } => {
                continue;
            },
        };

        report_builder.print((src_id, Source::from(source))).unwrap();
    }

}