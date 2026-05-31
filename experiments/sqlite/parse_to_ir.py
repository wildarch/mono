#!/usr/bin/env python3
"""
Parse ./sqlite3.c into a custom Intermediate Representation (IR).

The IR is a JSON-serializable nested dictionary/list structure that captures:
  - Operations (a + b, function calls, etc.)
  - Variable/function names
  - Control flow structure
  - Type information sufficient for round-tripping back to C

It deliberately ignores comments, whitespace, and formatting.

Usage:
    python3 parse_to_ir.py                  # parses sqlite3.c
    python3 parse_to_ir.py <input_file>     # parses a custom file
    python3 parse_to_ir.py --pretty         # pretty-print JSON output
    python3 parse_to_ir.py --stats          # print IR node statistics
"""

import sys
import os
import json
import argparse
from collections import defaultdict

from pycparser import c_ast, parse_file

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, 'sqlite3.c')

# pycparser's bundled fake libc headers
PYCPARSER_FAKE_LIBC = "/home/daan/workspace/pycparser/utils/fake_libc_include"

# ---------------------------------------------------------------------------
# IR node-type constants  (used as "kind" values in the IR)
# ---------------------------------------------------------------------------

# --- Top-level ---
K_FILE = "File"
K_FUNC_DEF = "FuncDef"
K_DECL = "Decl"          # global declaration / function declaration

# --- Statements ---
K_COMPOUND = "Compound"
K_IF = "If"
K_WHILE = "While"
K_DO_WHILE = "DoWhile"
K_FOR = "For"
K_SWITCH = "Switch"
K_CASE = "Case"
K_DEFAULT = "Default"
K_RETURN = "Return"
K_BREAK = "Break"
K_CONTINUE = "Continue"
K_GOTO = "Goto"
K_LABEL = "Label"
K_EMPTY = "EmptyStatement"

# --- Expressions ---
K_BINARY_OP = "BinaryOp"
K_UNARY_OP = "UnaryOp"
K_TERNARY = "TernaryOp"
K_ASSIGN = "Assignment"
K_FUNC_CALL = "FuncCall"
K_CAST = "Cast"
K_SIZEOF = "Sizeof"
K_ARRAY_REF = "ArrayRef"
K_STRUCT_REF = "StructRef"
K_MEMBER_DOT = "MemberDot"      # a.b
K_MEMBER_ARROW = "MemberArrow"  # a->b
K_ID = "ID"
K_CONSTANT = "Constant"

# --- Types ---
K_TYPEDEF = "TypeDef"
K_STRUCT = "Struct"
K_UNION = "Union"
K_ENUM = "Enum"
K_ENUMERATOR = "Enumerator"
K_TYPE_DECL = "TypeDecl"
K_PTR_DECL = "PtrDecl"
K_ARRAY_DECL = "ArrayDecl"
K_FUNC_DECL = "FuncDecl"
K_TYPENAME = "Typename"
K_IDENTIFIER_TYPE = "IdentifierType"

# --- Initialisation ---
K_INIT_LIST = "InitList"
K_NAMED_INIT = "NamedInitializer"


# ===========================================================================
# Helper utilities
# ===========================================================================

def _coord(node):
    """Return a dict with line/column info, or None."""
    c = node.coord
    if c is None:
        return None
    return {"line": c.line, "column": c.column}


def _unwrap(node_or_list):
    """If the argument is a list with a single element, unwrap it."""
    if isinstance(node_or_list, list) and len(node_or_list) == 1:
        return node_or_list[0]
    return node_or_list


# ===========================================================================
# Type converter  (pycparser type subtree → IR dict)
# ===========================================================================

class TypeConverter(c_ast.NodeVisitor):
    """Convert a pycparser type subtree into an IR type dict.

    The IR type representation is designed to be round-trippable back to C.
    """

    def convert(self, node):
        """Entry point: return an IR dict for the given type node."""
        if node is None:
            return None
        method = 'visit_' + type(node).__name__
        visitor = getattr(self, method, self._unknown_type)
        return visitor(node)

    def _unknown_type(self, node):
        return {"kind": "UnknownType", "name": type(node).__name__}

    # --- Compound types ---

    def visit_Struct(self, node):
        members = None
        if node.decls:
            members = [convert_decl(d) for d in node.decls]
        return {
            "kind": K_STRUCT,
            "name": node.name,
            "members": members,
        }

    def visit_Union(self, node):
        members = None
        if node.decls:
            members = [convert_decl(d) for d in node.decls]
        return {
            "kind": K_UNION,
            "name": node.name,
            "members": members,
        }

    def visit_Enum(self, node):
        values = None
        if node.values:
            values = [
                {"kind": "EnumValue", "name": e.name, "value": convert_expr(e.value)}
                for e in node.values.enumerators
            ]
        return {
            "kind": K_ENUM,
            "name": node.name,
            "values": values,
        }

    def visit_Enumerator(self, node):
        return {
            "kind": K_ENUMERATOR,
            "name": node.name,
            "value": convert_expr(node.value),
        }

    def visit_EnumeratorList(self, node):
        return [self.visit_Enumerator(e) for e in node.enumerators]

    # --- Derived types ---

    def visit_PtrDecl(self, node):
        return {
            "kind": K_PTR_DECL,
            "type": self.convert(node.type),
        }

    def visit_ArrayDecl(self, node):
        return {
            "kind": K_ARRAY_DECL,
            "type": self.convert(node.type),
            "dim": convert_expr(node.dim),
        }

    def visit_FuncDecl(self, node):
        params = None
        if node.args:
            params = convert_param_list(node.args)
        return {
            "kind": K_FUNC_DECL,
            "type": self.convert(node.type),
            "params": params,
        }

    # --- Leaf types ---

    def visit_TypeDecl(self, node):
        return {
            "kind": K_TYPE_DECL,
            "declname": node.declname,
            "type": self.visit_IdentifierType(node.type)
                    if isinstance(node.type, c_ast.IdentifierType)
                    else self.convert(node.type),
        }

    def visit_IdentifierType(self, node):
        return {
            "kind": K_IDENTIFIER_TYPE,
            "names": list(node.names),
        }

    def visit_Typename(self, node):
        return {
            "kind": K_TYPENAME,
            "name": node.name,
            "type": self.convert(node.type),
        }


# Singleton type converter
_type_converter = TypeConverter()


def convert_type(node):
    """Convert a pycparser type node to an IR dict."""
    return _type_converter.convert(node)


# ===========================================================================
# Expression converter  (pycparser expr → IR dict)
# ===========================================================================

def convert_expr(node):
    """Convert any pycparser expression node to an IR dict."""
    if node is None:
        return None

    cls = type(node).__name__

    # --- Constants ---
    if cls == 'Constant':
        return {
            "kind": K_CONSTANT,
            "type": node.type,
            "value": node.value,
        }

    # --- Identifiers ---
    if cls == 'ID':
        return {
            "kind": K_ID,
            "name": node.name,
        }

    # --- Binary operations ---
    if cls == 'BinaryOp':
        return {
            "kind": K_BINARY_OP,
            "op": node.op,
            "left": convert_expr(node.left),
            "right": convert_expr(node.right),
        }

    # --- Unary operations ---
    if cls == 'UnaryOp':
        op = node.op
        # sizeof is special — its "expr" is a type, not an expression
        if op == 'sizeof':
            arg = convert_type(node.expr) if isinstance(node.expr, c_ast.Typename) \
                  else convert_expr(node.expr)
            return {
                "kind": K_SIZEOF,
                "op": op,
                "arg": arg,
            }
        return {
            "kind": K_UNARY_OP,
            "op": op,
            "expr": convert_expr(node.expr),
        }

    # --- Ternary ---
    if cls == 'TernaryOp':
        return {
            "kind": K_TERNARY,
            "cond": convert_expr(node.cond),
            "iftrue": convert_expr(node.iftrue),
            "iffalse": convert_expr(node.iffalse),
        }

    # --- Assignment ---
    if cls == 'Assignment':
        return {
            "kind": K_ASSIGN,
            "op": node.op,
            "lvalue": convert_expr(node.lvalue),
            "rvalue": convert_expr(node.rvalue),
        }

    # --- Function call ---
    if cls == 'FuncCall':
        args = None
        if node.args:
            args = [convert_expr(a) for a in node.args.exprs]
        return {
            "kind": K_FUNC_CALL,
            "name": convert_expr(node.name),
            "args": args,
        }

    # --- Cast ---
    if cls == 'Cast':
        return {
            "kind": K_CAST,
            "type": convert_type(node.to_type),
            "expr": convert_expr(node.expr),
        }

    # --- Array subscript ---
    if cls == 'ArrayRef':
        return {
            "kind": K_ARRAY_REF,
            "name": convert_expr(node.name),
            "subscript": convert_expr(node.subscript),
        }

    # --- Struct/union member access ---
    if cls == 'StructRef':
        if node.type == '.':
            kind = K_MEMBER_DOT
        else:
            kind = K_MEMBER_ARROW
        return {
            "kind": kind,
            "name": convert_expr(node.name),
            "member": node.field.name,
        }

    # --- Compound literal (C99) ---
    if cls == 'CompoundLiteral':
        return {
            "kind": "CompoundLiteral",
            "type": convert_type(node.type),
            "init": convert_expr(node.init),
        }

    # --- Initializer list ---
    if cls == 'InitList':
        return {
            "kind": K_INIT_LIST,
            "exprs": [convert_expr(e) for e in node.exprs],
        }

    # --- Named initializer (designated initializer, C99) ---
    if cls == 'NamedInitializer':
        return {
            "kind": K_NAMED_INIT,
            "name": [d.name if isinstance(d, c_ast.ID) else convert_expr(d)
                     for d in node.name],
            "expr": convert_expr(node.expr),
        }

    # --- Expression list (used in function args, for init, etc.) ---
    if cls == 'ExprList':
        return [convert_expr(e) for e in node.exprs]

    # --- Fallback ---
    return {"kind": "UnknownExpr", "node_type": cls}


# ===========================================================================
# Statement converter  (pycparser stmt → IR dict)
# ===========================================================================

def convert_stmt(node):
    """Convert any pycparser statement node to an IR dict."""
    if node is None:
        return None

    cls = type(node).__name__

    # --- Compound block ---
    if cls == 'Compound':
        block_items = []
        if node.block_items:
            for item in node.block_items:
                block_items.append(convert_stmt(item))
        return {
            "kind": K_COMPOUND,
            "block_items": block_items,
        }

    # --- If / else ---
    if cls == 'If':
        return {
            "kind": K_IF,
            "cond": convert_expr(node.cond),
            "iftrue": convert_stmt(node.iftrue),
            "iffalse": convert_stmt(node.iffalse),
        }

    # --- While ---
    if cls == 'While':
        return {
            "kind": K_WHILE,
            "cond": convert_expr(node.cond),
            "stmt": convert_stmt(node.stmt),
        }

    # --- Do-while ---
    if cls == 'DoWhile':
        return {
            "kind": K_DO_WHILE,
            "cond": convert_expr(node.cond),
            "stmt": convert_stmt(node.stmt),
        }

    # --- For ---
    if cls == 'For':
        return {
            "kind": K_FOR,
            "init": _convert_for_init(node.init),
            "cond": convert_expr(node.cond),
            "next": convert_expr(node.next),
            "stmt": convert_stmt(node.stmt),
        }

    # --- Switch ---
    if cls == 'Switch':
        return {
            "kind": K_SWITCH,
            "cond": convert_expr(node.cond),
            "stmt": convert_stmt(node.stmt),
        }

    # --- Case ---
    if cls == 'Case':
        return {
            "kind": K_CASE,
            "expr": convert_expr(node.expr),
            "stmts": [convert_stmt(s) for s in node.stmts],
        }

    # --- Default ---
    if cls == 'Default':
        return {
            "kind": K_DEFAULT,
            "stmts": [convert_stmt(s) for s in node.stmts],
        }

    # --- Return ---
    if cls == 'Return':
        return {
            "kind": K_RETURN,
            "expr": convert_expr(node.expr),
        }

    # --- Break ---
    if cls == 'Break':
        return {"kind": K_BREAK}

    # --- Continue ---
    if cls == 'Continue':
        return {"kind": K_CONTINUE}

    # --- Goto ---
    if cls == 'Goto':
        return {
            "kind": K_GOTO,
            "name": node.name,
        }

    # --- Label ---
    if cls == 'Label':
        return {
            "kind": K_LABEL,
            "name": node.name,
            "stmt": convert_stmt(node.stmt),
        }

    # --- Empty statement ---
    if cls == 'EmptyStatement':
        return {"kind": K_EMPTY}

    # --- Expression statement (an expression used as a statement) ---
    if cls in ('BinaryOp', 'UnaryOp', 'TernaryOp', 'Assignment',
               'FuncCall', 'Cast', 'ArrayRef', 'StructRef', 'Constant',
               'ID', 'CompoundLiteral', 'InitList', 'NamedInitializer',
               'ExprList'):
        return convert_expr(node)

    # --- Declaration used as a statement ---
    if cls == 'Decl':
        return convert_decl(node)

    # --- Typedef used as a statement (C99 allows typedefs inside functions) ---
    if cls == 'Typedef':
        return convert_decl(node)

    # --- Declaration list (e.g. in for-loop init) ---
    if cls == 'DeclList':
        return {
            "kind": "DeclList",
            "decls": [convert_decl(d) for d in node.decls],
        }

    # --- Fallback ---
    return {"kind": "UnknownStmt", "node_type": cls}


def _convert_for_init(node):
    """Convert a for-loop init field (could be DeclList, ExprList, Assignment, etc.)."""
    if node is None:
        return None
    cls = type(node).__name__
    if cls == 'DeclList':
        return {
            "kind": "DeclList",
            "decls": [convert_decl(d) for d in node.decls],
        }
    if cls == 'ExprList':
        return [convert_expr(e) for e in node.exprs]
    if cls == 'Assignment':
        return convert_expr(node)
    return convert_expr(node)


# ===========================================================================
# Declaration converter  (pycparser decl → IR dict)
# ===========================================================================

def convert_decl(node):
    """Convert a pycparser Decl node to an IR dict."""
    if node is None:
        return None

    cls = type(node).__name__

    if cls == 'Decl':
        # Determine if this is a typedef
        is_typedef = False
        if isinstance(node.type, c_ast.Typedef):
            is_typedef = True
            inner_type = node.type.type
        else:
            inner_type = node.type

        # Determine if this is a function definition
        if isinstance(inner_type, c_ast.FuncDef):
            return convert_func_def(inner_type, node.name)

        # Determine if this is a function declaration (not definition)
        if isinstance(inner_type, c_ast.FuncDecl):
            return {
                "kind": K_DECL,
                "name": node.name,
                "quals": list(node.quals) if node.quals else [],
                "storage": list(node.storage) if node.storage else [],
                "type": convert_type(inner_type),
                "init": None,
                "bitsize": convert_expr(node.bitsize),
            }

        # Regular variable declaration
        return {
            "kind": K_DECL,
            "name": node.name,
            "quals": list(node.quals) if node.quals else [],
            "storage": list(node.storage) if node.storage else [],
            "type": convert_type(inner_type),
            "init": convert_expr(node.init),
            "bitsize": convert_expr(node.bitsize),
        }

    if cls == 'Typedef':
        return {
            "kind": K_TYPEDEF,
            "name": node.name,
            "type": convert_type(node.type),
        }

    if cls == 'FuncDef':
        return convert_func_def(node)

    # Fallback
    return {"kind": "UnknownDecl", "node_type": cls}


def convert_func_def(node, name=None):
    """Convert a FuncDef (or FuncDef wrapped in Decl) to IR."""
    if isinstance(node, c_ast.FuncDef):
        decl = node.decl
        body = node.body
        # Collect parameter names from param_decls (FuncDef-specific)
        param_decls = node.param_decls or []
    else:
        decl = node
        body = None
        param_decls = []

    func_name = name or decl.name
    func_type = decl.type  # FuncDecl or TypeDecl

    # Extract params
    params = None
    if isinstance(func_type, c_ast.FuncDecl) and func_type.args:
        params = convert_param_list(func_type.args)

    # Build the return type (unwrap FuncDecl to get the return type)
    if isinstance(func_type, c_ast.FuncDecl):
        ret_type = convert_type(func_type.type)
    else:
        ret_type = convert_type(func_type)

    return {
        "kind": K_FUNC_DEF,
        "name": func_name,
        "return_type": ret_type,
        "params": params,
        "body": convert_stmt(body) if body else None,
    }


def convert_param_list(node):
    """Convert a ParamList to a list of IR decl dicts."""
    if node is None or node.params is None:
        return []
    result = []
    for p in node.params:
        if isinstance(p, c_ast.EllipsisParam):
            result.append({"kind": "EllipsisParam"})
        elif isinstance(p, c_ast.Decl):
            result.append(convert_decl(p))
        elif isinstance(p, c_ast.Typename):
            result.append({
                "kind": "TypenameParam",
                "name": p.name,
                "quals": list(p.quals) if p.quals else [],
                "type": convert_type(p.type),
            })
        else:
            result.append({"kind": "UnknownParam", "node_type": type(p).__name__})
    return result


# ===========================================================================
# Top-level file converter
# ===========================================================================

def convert_file(ast):
    """Convert the top-level FileAST to an IR dict."""
    ext = []
    for node in ast.ext:
        cls = type(node).__name__

        if cls == 'FuncDef':
            ext.append(convert_func_def(node))
        elif cls == 'Decl':
            ext.append(convert_decl(node))
        elif cls == 'Typedef':
            ext.append({
                "kind": K_TYPEDEF,
                "name": node.name,
                "type": convert_type(node.type),
            })
        elif cls == 'Pragma':
            ext.append({"kind": "Pragma", "string": node.string})
        elif cls == 'EmptyStatement':
            ext.append({"kind": K_EMPTY})
        else:
            ext.append({"kind": "UnknownExt", "node_type": cls})

    return {
        "kind": K_FILE,
        "ext": ext,
    }


# ===========================================================================
# Statistics
# ===========================================================================

def collect_stats(ir_node, counter=None):
    """Walk the IR and count node kinds."""
    if counter is None:
        counter = defaultdict(int)

    if isinstance(ir_node, dict):
        counter[ir_node.get("kind", "unknown")] += 1
        for key, val in ir_node.items():
            if key == "kind":
                continue
            collect_stats(val, counter)
    elif isinstance(ir_node, list):
        for item in ir_node:
            collect_stats(item, counter)

    return counter


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Parse a preprocessed C file into a custom IR."
    )
    parser.add_argument(
        "input_file", nargs="?", default=DEFAULT_INPUT,
        help="Path to the preprocessed C file (default: sqlite3.c)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path (default: stdout)"
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print JSON output"
    )
    parser.add_argument(
        "--stats", action="store_true",
        help="Print IR node statistics"
    )
    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {input_file}", file=sys.stderr)

    try:
        ast = parse_file(
            input_file,
            use_cpp=True,
            cpp_path='cpp',
            cpp_args=['-I', PYCPARSER_FAKE_LIBC, '-D__attribute__(x)=']
        )
    except Exception as e:
        print(f"Parse error: {e}", file=sys.stderr)
        sys.exit(1)

    print("Converting AST to IR...", file=sys.stderr)
    ir = convert_file(ast)

    if args.stats:
        stats = collect_stats(ir)
        print("\nIR Node Statistics:", file=sys.stderr)
        print("-" * 40, file=sys.stderr)
        for kind, count in sorted(stats.items(), key=lambda x: -x[1]):
            print(f"  {kind:<30s} {count:>8d}", file=sys.stderr)
        print("-" * 40, file=sys.stderr)
        print(f"  {'TOTAL':<30s} {sum(stats.values()):>8d}", file=sys.stderr)

    indent = 2 if args.pretty else None
    json_str = json.dumps(ir, indent=indent, default=str)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(json_str)
        print(f"IR written to {args.output}", file=sys.stderr)
    else:
        print(json_str)


if __name__ == '__main__':
    main()
