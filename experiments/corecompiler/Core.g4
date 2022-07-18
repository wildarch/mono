grammar Core;

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
PACK : 'Pack';
LET : 'let';
LETREC : 'letrec';
IN : 'in';
CASE : 'case';
OF : 'of';

WS: [ \t\r\n]+ -> skip;
VAR: [a-zA-Z] [a-zA-Z0-9_']*;
NUM: [0-9]+;

program: sc (';' sc)* EOF;
sc: var+ '=' expr;
expr:
    // Defined in order of precedence
    // Each variant has a #Tag so that a visitor or listener knows the kind of expression encountered
	var # ExprVar
	| num # ExprNum
	| PACK '{' num ',' num '}' # ExprConstr
	| '(' expr ')' # ExprParen
	// Prec 6
	| expr expr # ExprApp // Application
	// Prec 5
	// op=(..) binds the operator token to a nane, so in the parser we can distinguish the operator
	| <assoc = right> expr op=(MUL | DIV) expr # ExprMulDiv
	// Prec 4
	| <assoc = right> expr op=(ADD | SUB) expr # ExprAddSub
	// Prec 3
	| <assoc = right> expr op=(
		EQ
		| NEQ
		| GT
		| GTE
		| LT
		| LTE
	) expr # ExprCompare
	| <assoc = right> expr AND expr # ExprAnd
	| <assoc = right> expr OR expr # ExprOr
	| rec=(LET | LETREC) defns IN expr # ExprLet
	| CASE expr OF alts # ExprCase
	| '\\' var+ '.' expr # ExprLam;

defns: defn (';' defn)*;
defn: var '=' expr;

alts: alt (';' alt)*;
alt: '<' num '>' var* '->' expr;

num: NUM;
var: VAR;