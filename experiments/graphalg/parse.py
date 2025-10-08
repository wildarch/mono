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
    
    def all_tokens(self):
        tokens = []
        while True:
            tok = self.next_token()
            tokens.append(tok)
            if tok.type == 'EOF':
                return tokens

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
    def __init__(self, tokens):
        self.tokens = tokens
        self.offset = 0
    
    def cur(self):
        if self.offset < len(self.tokens):
            return self.tokens[self.offset]
    
    def peek(self):
        if self.offset + 1 < len(self.tokens):
            return self.tokens[self.offset + 1]
        
    def eat(self, type=None):
        cur = self.cur()
        if type and cur.type != type:
            raise ParseError(f"Expected {type}", self.cur())
        self.offset += 1
        return cur
    
    def try_eat(self, type):
        cur = self.cur()
        if cur.type == type:
            self.offset += 1
            return cur
        return None
    
    def parse_params(self):
        self.eat('LPAREN')
        params = []
        while True:
            if self.try_eat('RPAREN'):
                # End of params
                break

            if len(params) > 0:
                # Additional parameters
                self.eat('COMMA')
                name = self.eat('IDENT')
            else:
                # First parameter
                name = self.eat('IDENT')

            self.eat('COLON')

            type = self.parse_type()
            params.append((name.body, type))

        return params
    
    def parse_dim(self):
        tok = self.cur()
        if tok.type == 'IDENT':
            self.eat()
            return tok.body
        elif tok.type == 'INT' and tok.body == '1':
            self.eat()
            return '1'
        else:
            raise ParseError("Expected dimension symbol", tok)

    def try_parse_semiring(self):
        tok = self.cur()
        if tok.type in SEMIRINGS:
            self.eat()
            return tok.type

    def parse_type(self):
        ring = self.try_parse_semiring()
        if ring:
            return ring
        elif self.try_eat('MATRIX'):
            self.eat('LANGLE')

            rows = self.parse_dim()
            self.eat('COMMA')

            cols = self.parse_dim()
            self.eat('COMMA')

            ring = self.try_parse_semiring()
            if not ring:
                raise ParseError('Expected semiring', self.cur())

            self.eat('RANGLE')
            return {
                'type': 'MATRIX',
                'rows': rows,
                'cols': cols,
                'ring': ring,
            }
        elif self.try_eat('VECTOR'):
            self.eat('LANGLE')

            rows = self.parse_dim()
            self.eat('COMMA')

            ring = self.try_parse_semiring()
            if not ring:
                raise ParseError('Expected semiring', self.cur())

            self.eat('RANGLE')
            return {
                'type': 'VECTOR',
                'rows': rows,
                'ring': ring,
            }
        else:
            raise ParseError("Expected type", tok)
        
    def parse_range(self):
        begin = self.parse_expr()
        if self.try_eat('COLON'):
            end = self.parse_expr()
            return {
                'begin': begin,
                'end': end,
            }
        else:
            return {
                'dim': begin,
            }
    
    def try_parse_mask(self):
        if not self.try_eat('LANGLE'):
            return None
        
        complement = self.try_eat('NOT')
        name = self.eat('IDENT')

        self.eat('RANGLE')
        return {
            'complement': complement,
            'name': name.body,
        }
    
    def try_parse_fill(self):
        if not self.try_eat('LSBRACKET'):
            return False
        
        self.eat('COLON')
        if self.try_eat('COMMA'):
            self.eat('COLON')

        self.eat('RSBRACKET')
        return True
    
    def parse_expr(self):
        while self.cur().type not in ['LBRACKET', 'SEMI']:
            self.eat()
        return None
    
    def parse_stmt(self):
        if self.try_eat('FOR'):
            iter_var = self.eat('IDENT')
            self.eat('IN')
            iter_range = self.parse_range()
            body = self.parse_block()
            return {
                'type': 'FOR',
                'iter_var': iter_var.body,
                'range': iter_range,
                'body': body,
            }
        elif self.try_eat('RETURN'):
            expr = self.parse_expr()
            self.eat('SEMI')
            return {
                'type': 'RETURN',
                'expr': expr,
            }

        base = self.eat('IDENT')

        accum = self.try_eat('ACCUM')
        if not accum:
            mask = self.try_parse_mask()
            fill = self.try_parse_fill()
            self.eat('ASSIGN')
        else:
            mask = None
            fill = False

        expr = self.parse_expr()
        self.eat('SEMI')
        return {
            'base': base.body,
            'accum': accum is not None,
            'mask': mask,
            'fill': fill,
            'expr': expr,
        }
    
    def parse_block(self):
        self.eat('LBRACKET')
        stmts = []

        while self.cur().type != 'RBRACKET':
            stmts.append(self.parse_stmt())
            
        self.eat('RBRACKET')
        return stmts
    
    def parse_func(self):
        self.eat('FUNC')
        ident = self.eat('IDENT')
        params = self.parse_params()

        arrow = self.eat('ARROW')
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
        while self.cur().type == 'FUNC':
            funcs.append(self.parse_func())
        if self.cur().type != 'EOF':
            raise ParseError("Invalid top-level definition", self.cur())

        return funcs

if __name__ == "__main__":
    path = sys.argv[1]
    with open(path) as f:
        lex = Lexer(path, f.read())
        parser = Parser(lex.all_tokens())
        program = parser.parse_program()
        print(program)