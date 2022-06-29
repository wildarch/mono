grammar Core;

WS: [ \t\r\n]+ -> skip;
VAR: [a-zA-Z] [a-zA-Z0-9_]*;
NUM: [0-9]+;

program: sc+ EOF;
sc: var+ '=' expr;
expr:
	var # ExprVar
	| num # ExprNum
	| 'Pack' '{' num ',' num '}' # ExprConstr
	| '(' expr ')' # ExprParen
	// Prec 6
	| expr expr # ExprApp // Application
	// Prec 5
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
	| <assoc = right> expr '&' expr # ExprAnd
	| <assoc = right> expr '|' expr # ExprOr
	| rec=(LET | LETREC) defns 'in' expr # ExprLet
	| 'case' expr 'of' alts # ExprCase
	| '\\' var+ '.' expr # ExprLam;

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

LET : 'let';
LETREC : 'letrec';

defns: defn (',' defn)*;
defn: var '=' expr;

alts: alt (';' alt)*;
alt: '<' num '>' var* '->' expr;

num: NUM;
var: VAR;