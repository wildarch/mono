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
    
    def next_token(self):
        if self.cur() is None:
            return Token('EOF', self.loc(), '')

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
            return Token('EOF', self.loc(), '')

        raise LexError(f"Unrecognized char '{self.cur()}'", self.loc())

class ParseError(Exception):
    def __init__(self, message, token):
        super().__init__(f"{token.loc}: {message} (token {token})")
        self.token = token

SEMIRINGS = [
    'BOOL',
    'INT',
    'REAL',
    'TROP_INT',
    'TROP_REAL',
    'TROP_MAX_INT',
]

class Parser(object):
    def __init__(self, lexer):
        self.lexer = lexer
    
    def parse_expect(self, type):
        tok = self.lexer.next_token()
        if tok.type != type:
            raise ParseError(f"Expected {type}", tok)
        return tok
    
    def parse_params(self):
        self.parse_expect('LPAREN')
        params = []
        while True:
            tok = self.lexer.next_token()
            if tok.type == 'RPAREN':
                # End of params
                break

            if len(params) > 0:
                # Additional parameters
                if tok.type != 'COMMA':
                    raise ParseError("Expected comma before next parameter", tok)
                name = self.parse_expect('IDENT')
            else:
                # First parameter
                if tok.type != 'IDENT':
                    raise ParseError("Expected parameter name", tok)
                name = tok
            
            self.parse_expect('COLON')

            type = self.parse_type()
            params.append((name, type))

        return params
    
    def parse_dim(self):
        tok = self.lexer.next_token()
        if tok.type == 'IDENT':
            return tok.body
        elif tok.type == 'INT' and tok.body == '1':
            return '1'
        else:
            raise ParseError("Expected dimension symbol", tok)

    def parse_semiring(self):
        tok = self.lexer.next_token()
        if tok.type in SEMIRINGS:
            return tok.type
        else:
            raise ParseError("Expected semiring", tok)

    def parse_type(self):
        tok = self.lexer.next_token()
        if tok.type in SEMIRINGS:
            return tok.type
        elif tok.type == 'MATRIX':
            self.parse_expect('LANGLE')

            rows = self.parse_dim()
            self.parse_expect('COMMA')

            cols = self.parse_dim()
            self.parse_expect('COMMA')

            ring = self.parse_semiring()

            self.parse_expect('RANGLE')
            return {
                'type': 'MATRIX',
                'rows': rows,
                'cols': cols,
                'ring': ring,
            }
        elif tok.type == 'VECTOR':
            self.parse_expect('LANGLE')

            rows = self.parse_dim()
            self.parse_expect('COMMA')

            ring = self.parse_semiring()

            self.parse_expect('RANGLE')
            return {
                'type': 'VECTOR',
                'rows': rows,
                'ring': ring,
            }
        else:
            raise ParseError("Expected type", tok)
    
    def parse_expr(self):
        while True:
            tok = self.lexer.next_token()
            # TODO: Does not always end with SEMI
            if tok.type == 'SEMI':
                return None

    def parse_stmt(self):
        tok = self.lexer.next_token()
        if tok.type == 'RBRACKET':
            return None

        if tok.type == 'IDENT':
            base = tok
            tok = self.lexer.next_token()
            if tok.type == '<':
                # Mask
                # TODO
                pass

            fill = False
            if tok.type == 'LSBRACKET':
                # Fill
                fill = True
                self.parse_expect('COLON')

                # Optional ,:
                tok = self.lexer.next_token()
                if tok.type == 'COMMA':
                    self.parse_expect('COLON')
                    tok = self.lexer.next_token()
                
                if tok.type != 'RSBRACKET':
                    raise ParseError('Invalid fill spec', tok)

                tok = self.lexer.next_token()

            accum = False
            if tok.type == '+':
                # Accumulate
                # TODO: Don't allow mask or fill
                accum = True
                tok = self.lexer.next_token()
            
            if tok.type != 'ASSIGN':
                raise ParseError("Expected '='", tok)
            
            expr = self.parse_expr()
            return {
                'type': 'ASSIGN',
                'base': base.body,
                'fill': fill,
                'expr': expr,
            }
        elif tok.type == 'FOR':
            iter_var = self.parse_expect('IDENT')
            self.parse_expect('IN')

            # Range
            begin = self.parse_expr()
            tok = self.lexer.next_token()
            if tok.type == 'COLON':
                end = self.parse_expr()
                self.parse_expect('LBRACKET')
                range = {
                    'begin': begin,
                    'end': end,
                }
            elif tok.type == 'LBRACKET':
                range = {
                    'dim': begin,
                }
            else:
                raise ParseError("Invalid range", tok)

            # Body
            stmts = []
            while True:
                stmt = self.parse_stmt()
                if stmt:
                    stmts.append(stmt)
                else:
                    break
            return {
                'type': 'FOR',
                'iter_var': iter_var.body,
                'range': range,
                'body': stmts,
            }

            # TODO until
        elif tok.type == 'RETURN':
            expr = self.parse_expr()
            return {
                'type': 'RETURN',
                'expr': expr,
            }
    
    def parse_block(self):
        self.parse_expect('LBRACKET')
        stmts = []

        while True:
            stmt = self.parse_stmt()
            if stmt:
                stmts.append(stmt)
            else:
                break
    
    def parse_func(self):
        func = self.lexer.next_token()
        if func.type == 'EOF':
            return None

        if func.type != 'FUNC':
            raise ParseError("Expected 'func'", func)
        
        ident = self.parse_expect('IDENT')
        params = self.parse_params()

        arrow = self.parse_expect('ARROW')
        retty = self.parse_type()

        block = self.parse_block()

        return {
            'name': ident.body,
            'params': params,
            'return_type': retty,
            'block': block,
        }


    def parse_program(self):
        funcs = []
        while True:
            f = self.parse_func()
            if f is None:
                break
            funcs.append(f)
        return funcs

if __name__ == "__main__":
    path = sys.argv[1]
    with open(path) as f:
        lex = Lexer(path, f.read())
        parser = Parser(lex)
        program = parser.parse_program()
        print(program)