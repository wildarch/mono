grammar Brim;

// TODO: uses semicolons, eventually we should move to python-style indentation based parsing.

// Operators
// Giving them a name allows referring to their token in generated code
MUL : '*';
DIV : '/';
ADD : '+';
SUB : '-';
EQ : '==';
NEQ : '~=';
GT : '>';
GTE : '>=';
LT : '<';
LTE : '<=';

AND : '&';
OR : '|';

// Keywords
// Defined before VAR to prevent recognizing keywords as identifiers
LET : 'let';
LETREC : 'letrec';
IN : 'in';
CASE : 'case';
OF : 'of';

WS: [ \t\r\n]+ -> skip;
// Variables start with lowercase
VAR: [a-z] [a-zA-Z0-9_']*;
// Names start with uppercase
NAME: [A-Z] [a-zA-Z0-9_']*;
NUM: [0-9]+;

module: supercomb (';' supercomb)* EOF;
supercomb:
    var '::' type ';'
    var+ '=' expr ';';
expr:
    // Defined in order of precedence
    // Each variant has a #Tag so that a visitor or listener knows the kind of expression encountered
	var # ExprVar
	| num # ExprNum
	| '(' expr ')' # ExprParen
	// Prec 6
	| expr expr # ExprApp // Application
	// Prec 5
	// op=(..) binds the operator token to a nane, so in the parser we can distinguish the operator
	| <assoc = left> expr op=(MUL | DIV) expr # ExprMulDiv
	// Prec 4
	| <assoc = left> expr op=(ADD | SUB) expr # ExprAddSub
	// Prec 3
	| <assoc = left> expr op=(
		EQ
		| NEQ
		| GT
		| GTE
		| LT
		| LTE
	) expr # ExprCompare
	| <assoc = left> expr AND expr # ExprAnd
	| <assoc = left> expr OR expr # ExprOr
	| rec=(LET | LETREC) defns IN expr # ExprLet
	| CASE expr OF alts # ExprCase
	| '\\' var+ '.' expr # ExprLam;

defns: defn (';' defn)*;
defn: var '=' expr;

alts: alt (';' alt)*;
// TODO: nested pattern matching
alt: name var* '->' expr;

type:
    var # TypeVar
    | name type* # TypeName
    | type '->' type #TypeFun
    | '(' type ')' # TypeParen
    ;

num: NUM;
var: VAR;
name: NAME;
