import sys

class Loc(object):
    def __init__(self, file, line, col):
        self.file = file
        self.line = line
        self.col = col

    def __str__(self):
        return f"{self.file}:{self.line}:{self.col}"
    
class LexError(Exception):
    def __init__(self, message, loc):
        super().__init__(f"{loc}: {message}")
        self.loc = loc
    
class Token(object):
    def __init__(self, type, loc, body):
        self.type = type
        self.loc = loc
        self.body = body

    def __str__(self):
        return f"{self.type} '{self.body}' @ {self.loc}"

KEYWORDS = {
    'func': 'FUNC',
    'return': 'RETURN',
    'for': 'FOR',
    'in': 'IN',
    'until': 'UNTIL',
    'true': 'TRUE',
    'false': 'FALSE',
    # Builtin functions
    'diag': 'DIAG',
    'apply': 'APPLY',
    'select': 'SELECT',
    'tril': 'TRIL',
    'triu': 'TRIU',
    'reduceRows': 'REDUCE_ROWS',
    'reduceCols': 'REDUCE_COLS',
    'reduce': 'REDUCE',
    'cast': 'CAST',
    'zero': 'ZERO',
    'one': 'ONE',
    'pickAny': 'PICK_ANY',
    # Semirings
    'bool': 'BOOL',
    'int': 'INT',
    'real': 'REAL',
    'trop_int': 'TROP_INT',
    'trop_real': 'TROP_REAL',
    'trop_max_int': 'TROP_MAX_INT',
    # Matrices
    'Matrix': 'MATRIX',
    'Vector': 'VECTOR',
    # Properties
    'T': 'T',
    'nrows': 'NROWS',
    'ncols': 'NCOLS',
    'nvals': 'NVALS',
}
    
class Lexer(object):
    def __init__(self, filename, buffer):
        self.filename = filename
        self.buffer = buffer
        self.offset = 0
        self.line = 1
        self.col = 1
    
    def cur(self):
        if self.offset < len(self.buffer):
            return self.buffer[self.offset]
        else:
            # End of buffer
            return None
    
    def peek(self):
        if self.offset + 1 < len(self.buffer):
            return self.buffer[self.offset + 1]
        else:
            # End of buffer
            return None
    
    def loc(self):
        return Loc(self.filename, self.line, self.col)
    
    def eat(self, c = None):
        if c is not None:
            if self.cur() != c:
                raise LexError(f"Expected '{c}', got {self.cur()}", self.loc())
        
        if self.cur() == '\n':
            self.col = 1
            self.line += 1
        else:
            self.col += 1

        self.offset += 1
    
    def eat_whitespace(self):
        while True:
            if self.cur() and self.cur().isspace():
                # Simple whitespace
                self.eat()
                continue
            
            if self.cur() == '/' and self.peek() == '/':
                # Line comment
                while self.cur() != '\n':
                    self.eat()
                self.eat('\n')
                continue
            
            break
    
    def nextToken(self):
        if self.cur() is None:
            return None

        self.eat_whitespace()
    
        if self.cur() and self.cur().isalpha():
            # Identifier [a-zA-Z][a-zA-Z0-9_]*
            loc = self.loc()
            start = self.offset
            self.eat()
            while (self.cur() and self.cur().isalnum()) or self.cur() == '_':
                self.eat()
            end = self.offset

            ident = self.buffer[start:end]
            if ident in KEYWORDS:
                return Token(KEYWORDS[ident], loc, ident)
            return Token('IDENT', loc, ident)

        if self.cur() and self.cur().isdigit():
            # Number 
            loc = self.loc()
            start = self.offset
            self.eat()
            while self.cur() and self.cur().isdigit():
                self.eat()

            if self.cur() == '.':
                # Float, parse decimals
                self.eat('.')

                while self.cur() and self.cur().isdigit():
                    self.eat()

                end = self.offset
                return Token('FLOAT', loc, self.buffer[start:end])
            else:
                end = self.offset
                return Token('INT', loc, self.buffer[start:end])

        DOUBLE = {
            '->': 'ARROW',
            '+=': 'ACCUM',
            '==': 'EQUAL',
            '!=': 'NOT_EQUAL',
            '<=': 'LEQ',
            '>=': 'GEQ',
        }

        if self.cur() and self.peek():
            two = self.cur() + self.peek()
            if two in DOUBLE:
                tok = Token(DOUBLE[two], self.loc(), two)
                self.eat(two[0])
                self.eat(two[1])
                return tok
        
        SINGLE = {
            # Brackets
            '(': 'LPAREN',
            ')': 'RPAREN',
            '{': 'LBRACKET',
            '}': 'RBRACKET',
            '[': 'LSBRACKET',
            ']': 'RSBRACKET',
            '<': 'LANGLE',
            '>': 'RANGLE',
            # Delimiters
            ':': 'COLON',
            ',': 'COMMA',
            '.': 'DOT',
            ';': 'SEMI',
            # Operators
            '+': 'PLUS',
            '-': 'MINUS',
            '*': 'STAR',
            '/': 'DIVIDE',
            '=': 'ASSIGN',
            '!': 'NOT',
        }

        if self.cur() in SINGLE:
            tok = Token(SINGLE[self.cur()], self.loc(), self.cur())
            self.eat()
            return tok

        if self.cur() is None:
            return None

        raise LexError(f"Unrecognized char '{self.cur()}'", self.loc())

if __name__ == "__main__":
    path = sys.argv[1]
    with open(path) as f:
        lex = Lexer(path, f.read())

        while True:
            tok = lex.nextToken()
            if tok is None:
                break
            print(tok)