grammar Core;

WS: [ \t\r\n]+ -> skip;
VAR: [a-zA-Z] [a-zA-Z0-9_]*;
NUM: [0-9]+;

program: sc+ EOF;
sc: var+ '=' expr;
expr:
	var # ExprVar
	| num # ExprNum
	| 'Pack' '{' num ',' num '}' # ExprPack
	| '(' expr ')' # ExprParen
	// Prec 6
	| expr expr # ExprApp // Application
	// Prec 5
	| <assoc = right> expr ('*' | '/') expr # ExprMulDiv
	// Prec 4
	| <assoc = right> expr ('+' | '-') expr # ExprAddSub
	// Prec 3
	| <assoc = right> expr (
		'=='
		| '~='
		| '>'
		| '>='
		| '<'
		| '<='
	) expr # ExprCompare
	| <assoc = right> expr '&' expr # ExprAnd
	| <assoc = right> expr '|' expr # ExprOr
	| 'let' defns 'in' expr # ExprLet
	| 'letrec' defns 'in' expr # ExprLetRec
	| 'case' expr 'of' alts # ExprCase
	| '\\' var+ '.' expr # ExprVarDef;
defns: defn (',' defn)*;
defn: var '=' expr;

alts: alt (';' alt)*;
alt: '<' num '>' var* '->' expr;

num: NUM;
var: VAR;