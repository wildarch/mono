grammar Recipe;

WS: [ \t\r\n]+ -> skip;
NUM: [0-9]+ ('.' [0-9]+)?;

// UNITS
TSP: 'tsp';
TBSP: 'tbsp';
G: 'g';
ML: 'ml';

WORD: [a-zA-Z] [a-zA-Z0-9]*;

ingredient: quantity? name EOF;

quantity: NUM unit=(TSP | TBSP | G | ML)?;

name: WORD+;