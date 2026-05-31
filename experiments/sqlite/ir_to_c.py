#!/usr/bin/env python3
"""
Convert the custom IR (JSON) back to C code.

This is the inverse of parse_to_ir.py. It reads a JSON IR file produced by
parse_to_ir.py and emits equivalent C code.

Usage:
    python3 ir_to_c.py <input.json>              # output to stdout
    python3 ir_to_c.py <input.json> -o <output.c>
"""

import sys
import os
import json
import argparse

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(SCRIPT_DIR, 'sqlite3.c.json')


# ===========================================================================
# Type emitter  (IR type dict → C type string)
# ===========================================================================

def emit_type(type_node, name=None):
    """Convert an IR type dict to a C type declaration string.

    If `name` is provided, it is the declarator name (e.g. variable or
    function name) and the result is a full declaration like "int x".
    If `name` is None, the result is just the type part like "int *".
    """
    if type_node is None:
        return name or ""

    kind = type_node.get("kind")

    if kind == "IdentifierType":
        quals = type_node.get("quals", [])
        base = " ".join(type_node["names"])
        if quals:
            base = f"{base} {' '.join(quals)}"
        if name:
            return f"{base} {name}"
        return base

    if kind == "TypeDecl":
        quals = type_node.get("quals", [])
        declname = type_node.get("declname")
        # Emit the base type first
        base_str = emit_type(type_node["type"])
        if quals:
            base_str = f"{base_str} {' '.join(quals)}"
        if name:
            return f"{base_str} {name}"
        if declname:
            return f"{base_str} {declname}"
        return base_str

    if kind == "PtrDecl":
        inner_type = type_node["type"]
        # Check if the inner type is a FuncDecl or ArrayDecl — those need
        # parentheses around the whole "name" part.
        inner_kind = inner_type.get("kind") if inner_type else None
        if inner_kind in ("FuncDecl", "ArrayDecl"):
            if name:
                return emit_type(inner_type, f"(*{name})")
            return emit_type(inner_type, "(*)")
        else:
            if name:
                return emit_type(inner_type, f"*{name}")
            return emit_type(inner_type, "*")

    if kind == "ArrayDecl":
        dim_str = ""
        if type_node.get("dim") is not None:
            dim_str = f"[{emit_expr(type_node['dim'])}]"
        if name:
            return emit_type(type_node["type"], f"{name}{dim_str}")
        return emit_type(type_node["type"], dim_str)

    if kind == "FuncDecl":
        params = type_node.get("params")
        if params is None:
            param_str = ""
        elif len(params) == 0:
            param_str = "void"
        else:
            param_parts = []
            for p in params:
                param_parts.append(emit_param(p))
            param_str = ", ".join(param_parts)
        if name:
            return emit_type(type_node["type"], f"{name}({param_str})")
        return emit_type(type_node["type"], f"({param_str})")

    if kind == "Struct":
        name_part = type_node.get("name") or ""
        members = type_node.get("members")
        if members is None:
            # Forward declaration / opaque reference
            if name:
                return f"struct {name_part} {name}"
            return f"struct {name_part}"
        member_strs = []
        for m in members:
            member_strs.append(f"  {emit_decl(m, indent=2)};")
        body = "\n".join(member_strs)
        if name:
            return f"struct {name_part} {{\n{body}\n}} {name}"
        return f"struct {name_part} {{\n{body}\n}}"

    if kind == "Union":
        name_part = type_node.get("name") or ""
        members = type_node.get("members")
        if members is None:
            if name:
                return f"union {name_part} {name}"
            return f"union {name_part}"
        member_strs = []
        for m in members:
            member_strs.append(f"  {emit_decl(m, indent=2)};")
        body = "\n".join(member_strs)
        if name:
            return f"union {name_part} {{\n{body}\n}} {name}"
        return f"union {name_part} {{\n{body}\n}}"

    if kind == "Enum":
        name_part = type_node.get("name") or ""
        values = type_node.get("values")
        if values is None:
            if name:
                return f"enum {name_part} {name}"
            return f"enum {name_part}"
        value_strs = []
        for v in values:
            vname = v["name"]
            vval = emit_expr(v["value"]) if v.get("value") else ""
            if vval:
                value_strs.append(f"  {vname} = {vval}")
            else:
                value_strs.append(f"  {vname}")
        body = ",\n".join(value_strs)
        if name:
            return f"enum {name_part} {{\n{body}\n}} {name}"
        return f"enum {name_part} {{\n{body}\n}}"

    if kind == "Typename":
        # Used in casts — emit the type with the given name
        return emit_type(type_node["type"], name)

    # Fallback
    if name:
        return f"/* unknown type {kind} */ {name}"
    return f"/* unknown type {kind} */"


# ===========================================================================
# Expression emitter  (IR expr dict → C expression string)
# ===========================================================================

def emit_expr(node):
    """Convert an IR expression dict to a C expression string."""
    if node is None:
        return ""

    if isinstance(node, list):
        # ExprList
        return ", ".join(emit_expr(e) for e in node)

    kind = node.get("kind")

    if kind == "Constant":
        return node["value"]

    if kind == "ID":
        return node["name"]

    if kind == "BinaryOp":
        left = emit_expr(node["left"])
        right = emit_expr(node["right"])
        return f"{left} {node['op']} {right}"

    if kind == "UnaryOp":
        op = node["op"]
        expr = emit_expr(node["expr"])
        if op in ("++", "--"):
            # Could be prefix or postfix — pycparser stores prefix as "p++" etc.
            if op.startswith("p"):
                return f"{op[1:]}{expr}"
            return f"{op}{expr}"
        if op == "sizeof":
            return f"sizeof({expr})"
        return f"{op}{expr}"

    if kind == "TernaryOp":
        cond = emit_expr(node["cond"])
        iftrue = emit_expr(node["iftrue"])
        iffalse = emit_expr(node["iffalse"])
        return f"{cond} ? {iftrue} : {iffalse}"

    if kind == "Assignment":
        lvalue = emit_expr(node["lvalue"])
        rvalue = emit_expr(node["rvalue"])
        return f"{lvalue} {node['op']} {rvalue}"

    if kind == "FuncCall":
        name = emit_expr(node["name"])
        args = node.get("args")
        if args is None:
            arg_str = ""
        else:
            arg_str = ", ".join(emit_expr(a) for a in args)
        return f"{name}({arg_str})"

    if kind == "Cast":
        type_str = emit_type(node["type"])
        expr = emit_expr(node["expr"])
        return f"({type_str})({expr})"

    if kind == "Sizeof":
        arg = node["arg"]
        if isinstance(arg, dict) and "kind" in arg:
            if arg["kind"] in ("IdentifierType", "TypeDecl", "Typename",
                               "Struct", "Union", "Enum", "PtrDecl",
                               "ArrayDecl", "FuncDecl"):
                return f"sizeof({emit_type(arg)})"
        return f"sizeof({emit_expr(arg)})"

    if kind == "ArrayRef":
        name = emit_expr(node["name"])
        subscript = emit_expr(node["subscript"])
        return f"{name}[{subscript}]"

    if kind == "MemberDot":
        name = emit_expr(node["name"])
        return f"{name}.{node['member']}"

    if kind == "MemberArrow":
        name = emit_expr(node["name"])
        return f"{name}->{node['member']}"

    if kind == "InitList":
        exprs = ", ".join(emit_expr(e) for e in node["exprs"])
        return f"{{{exprs}}}"

    if kind == "NamedInitializer":
        parts = []
        for d in node["name"]:
            if isinstance(d, dict):
                parts.append(f"[{emit_expr(d)}]")
            else:
                parts.append(f".{d}")
        expr = emit_expr(node["expr"])
        return f"{''.join(parts)} = {expr}"

    if kind == "CompoundLiteral":
        type_str = emit_type(node["type"])
        init = emit_expr(node["init"])
        return f"({type_str}){init}"

    if kind == "DeclList":
        # Used in for-loop init
        parts = []
        for d in node["decls"]:
            parts.append(emit_decl(d))
        return ", ".join(parts)

    # Fallback
    return f"/* unknown expr {kind} */"


def _inject_quals(type_node, quals):
    """Inject qualifiers into the innermost type of a type tree.

    For example, PtrDecl(PtrDecl(TypeDecl(void))) with quals=["volatile"]
    becomes PtrDecl(PtrDecl(TypeDecl(void, quals=["volatile"]))).
    """
    if not quals:
        return type_node
    if type_node is None:
        return {"kind": "IdentifierType", "names": [], "quals": quals}

    kind = type_node.get("kind")
    if kind in ("PtrDecl", "ArrayDecl"):
        return dict(type_node, type=_inject_quals(type_node["type"], quals))
    if kind == "TypeDecl":
        existing = type_node.get("quals", [])
        return dict(type_node, quals=existing + quals)
    if kind == "IdentifierType":
        return dict(type_node, quals=quals)
    return type_node


# ===========================================================================
# Parameter emitter
# ===========================================================================

def emit_param(node):
    """Convert an IR parameter dict to a C parameter declaration string."""
    kind = node.get("kind")

    if kind == "EllipsisParam":
        return "..."

    if kind == "TypenameParam":
        quals = node.get("quals", [])
        type_node = node["type"]
        if quals:
            type_node = _inject_quals(type_node, quals)
        return emit_type(type_node, node.get("name"))

    if kind == "Decl":
        return emit_decl(node)

    return f"/* unknown param {kind} */"


# ===========================================================================
# Declaration emitter  (IR decl dict → C declaration string)
# ===========================================================================

def emit_decl(node, indent=0):
    """Convert an IR decl dict to a C declaration string."""
    if node is None:
        return ""

    kind = node.get("kind")

    if kind == "Decl":
        name = node.get("name", "")
        quals = node.get("quals", [])
        storage = node.get("storage", [])
        type_node = node["type"]
        init = node.get("init")
        bitsize = node.get("bitsize")

        # Determine if the type involves pointers — if so, inject quals
        # into the type tree (postfix style: char const **x); otherwise
        # use prefix style (const char x).
        def _has_pointer(tn):
            if tn is None:
                return False
            if tn.get("kind") == "PtrDecl":
                return True
            if tn.get("kind") in ("TypeDecl", "ArrayDecl"):
                return _has_pointer(tn.get("type"))
            return False

        if quals and _has_pointer(type_node):
            type_node = _inject_quals(type_node, quals)
            quals = []

        type_str = emit_type(type_node, name)

        # If the type is an ArrayDecl with null dim but there's an
        # initializer, we need to emit [] (size inferred from init).
        if init is not None and type_node.get("kind") == "ArrayDecl" and type_node.get("dim") is None:
            type_str += "[]"

        prefix = " ".join(storage + quals)
        if prefix:
            prefix += " "

        if init is not None:
            init_str = emit_expr(init)
            result = f"{prefix}{type_str} = {init_str}"
        else:
            result = f"{prefix}{type_str}"

        if bitsize is not None:
            result += f" : {emit_expr(bitsize)}"

        return result

    if kind == "TypeDef":
        type_str = emit_type(node["type"], node.get("name"))
        return f"typedef {type_str}"

    if kind == "FuncDef":
        return emit_func_def(node)

    if kind == "Struct":
        return emit_type(node)

    if kind == "Union":
        return emit_type(node)

    if kind == "Enum":
        return emit_type(node)

    # Fallback
    return f"/* unknown decl {kind} */"


# ===========================================================================
# Function definition emitter
# ===========================================================================

def emit_func_def(node):
    """Convert an IR FuncDef dict to a C function definition string."""
    name = node.get("name", "")
    params = node.get("params")
    body = node.get("body")

    if params is None:
        param_str = ""
    elif len(params) == 0:
        param_str = "void"
    else:
        param_parts = []
        for p in params:
            param_parts.append(emit_param(p))
        param_str = ", ".join(param_parts)

    # Build the return type string. The return_type may be a TypeDecl whose
    # declname is the function name — strip it so we can add the name ourselves.
    ret_type_node = node["return_type"]
    ret_str = emit_type(ret_type_node)
    # If the return type ends with the function name (TypeDecl case), strip it.
    if name and ret_str.endswith(" " + name):
        ret_str = ret_str[:-(len(name) + 1)]
    elif name and ret_str == name:
        ret_str = ""
    signature = f"{ret_str} {name}({param_str})"

    if body is None:
        # Function declaration (no body)
        return f"{signature};"

    body_str = emit_stmt(body, indent=0)
    return f"{signature}\n{body_str}"


# ===========================================================================
# Statement emitter  (IR stmt dict → C statement string)
# ===========================================================================

def emit_stmt(node, indent=0):
    """Convert an IR statement dict to a C statement string."""
    if node is None:
        return ""

    # ExprList used as a statement (e.g. "a, b;") — convert_expr returns a list
    if isinstance(node, list):
        return f"{emit_expr(node)};"

    kind = node.get("kind")
    ind = "  " * indent

    if kind == "Compound":
        items = node.get("block_items") or []
        if not items:
            return "{}"
        lines = ["{"]
        for item in items:
            lines.append(emit_stmt(item, indent + 1))
        lines.append("}")
        return "\n".join(lines)

    if kind == "If":
        cond = emit_expr(node["cond"])
        iftrue = emit_stmt(node["iftrue"], indent + 1)
        iffalse = node.get("iffalse")

        # Check if iftrue is a compound statement (starts with "{")
        if iftrue.strip().startswith("{"):
            result = f"if ({cond})\n{iftrue}"
        else:
            result = f"if ({cond})\n{ind}  {iftrue.lstrip()}"

        if iffalse is not None:
            iffalse_str = emit_stmt(iffalse, indent + 1)
            if iffalse_str.strip().startswith("{"):
                result += f"\n{ind}else\n{iffalse_str}"
            else:
                result += f"\n{ind}else\n{ind}  {iffalse_str.lstrip()}"
        return result

    if kind == "While":
        cond = emit_expr(node["cond"])
        stmt = emit_stmt(node["stmt"], indent + 1)
        if stmt.strip().startswith("{"):
            return f"while ({cond})\n{stmt}"
        return f"while ({cond})\n{ind}  {stmt.lstrip()}"

    if kind == "DoWhile":
        cond = emit_expr(node["cond"])
        stmt = emit_stmt(node["stmt"], indent + 1)
        if stmt.strip().startswith("{"):
            return f"do\n{stmt}\n{ind}while ({cond});"
        return f"do\n{ind}  {stmt.lstrip()}\n{ind}while ({cond});"

    if kind == "For":
        init = _emit_for_init(node.get("init"))
        cond = emit_expr(node.get("cond")) or ""
        next_ = emit_expr(node.get("next")) or ""
        stmt = emit_stmt(node["stmt"], indent + 1)
        header = f"for ({init}; {cond}; {next_})"
        if stmt.strip().startswith("{"):
            return f"{header}\n{stmt}"
        return f"{header}\n{ind}  {stmt.lstrip()}"

    if kind == "Switch":
        cond = emit_expr(node["cond"])
        stmt = emit_stmt(node["stmt"], indent + 1)
        return f"switch ({cond})\n{stmt}"

    if kind == "Case":
        expr = emit_expr(node["expr"])
        stmts = node.get("stmts") or []
        lines = [f"case {expr}:"]
        for s in stmts:
            lines.append(emit_stmt(s, indent + 1))
        return "\n".join(lines)

    if kind == "Default":
        stmts = node.get("stmts") or []
        lines = ["default:"]
        for s in stmts:
            lines.append(emit_stmt(s, indent + 1))
        return "\n".join(lines)

    if kind == "Return":
        expr = node.get("expr")
        if expr is not None:
            return f"return {emit_expr(expr)};"
        return "return;"

    if kind == "Break":
        return "break;"

    if kind == "Continue":
        return "continue;"

    if kind == "Goto":
        return f"goto {node['name']};"

    if kind == "Label":
        stmt = emit_stmt(node["stmt"], indent)
        return f"{node['name']}:\n{stmt}"

    if kind == "EmptyStatement":
        return ";"

    # Expression statement
    if kind in ("BinaryOp", "UnaryOp", "TernaryOp", "Assignment",
                "FuncCall", "Cast", "ArrayRef", "StructRef", "Constant",
                "ID", "CompoundLiteral", "InitList", "NamedInitializer",
                "MemberDot", "MemberArrow"):
        return f"{emit_expr(node)};"

    if kind == "Decl":
        return f"{emit_decl(node, indent)};"

    if kind == "DeclList":
        return f"{emit_expr(node)};"

    if kind == "TypeDef":
        return f"{emit_decl(node)};"

    # Fallback
    return f"/* unknown stmt {kind} */"


def _emit_for_init(node):
    """Convert a for-loop init to a string."""
    if node is None:
        return ""
    if isinstance(node, list):
        return ", ".join(emit_expr(e) for e in node)
    if isinstance(node, dict):
        if node.get("kind") == "DeclList":
            parts = []
            for d in node["decls"]:
                parts.append(emit_decl(d))
            return ", ".join(parts)
        return emit_expr(node)
    return str(node)


# ===========================================================================
# Top-level file emitter
# ===========================================================================

def emit_file(ir):
    """Convert the top-level IR File node to C code."""
    ext = ir.get("ext") or []
    parts = []
    for node in ext:
        kind = node.get("kind")

        if kind == "FuncDef":
            parts.append(emit_func_def(node))
        elif kind == "Decl":
            parts.append(f"{emit_decl(node)};")
        elif kind == "TypeDef":
            parts.append(f"{emit_decl(node)};")
        elif kind == "Struct":
            parts.append(f"{emit_type(node)};")
        elif kind == "Union":
            parts.append(f"{emit_type(node)};")
        elif kind == "Enum":
            parts.append(f"{emit_type(node)};")
        elif kind == "Pragma":
            parts.append(f"#pragma {node.get('string', '')}")
        elif kind == "EmptyStatement":
            parts.append(";")
        else:
            parts.append(f"/* unknown ext {kind} */")

    return "\n".join(parts)


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert the custom IR (JSON) back to C code."
    )
    parser.add_argument(
        "input_file", nargs="?", default=DEFAULT_INPUT,
        help="Path to the JSON IR file (default: sqlite3.c.json)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output file path (default: stdout)"
    )
    args = parser.parse_args()

    input_file = args.input_file
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.", file=sys.stderr)
        sys.exit(1)

    print(f"Reading IR from {input_file}", file=sys.stderr)
    with open(input_file, 'r') as f:
        ir = json.load(f)

    print("Emitting C code...", file=sys.stderr)
    c_code = emit_file(ir)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(c_code)
        print(f"C code written to {args.output}", file=sys.stderr)
    else:
        print(c_code)


if __name__ == '__main__':
    main()
