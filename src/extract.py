"""
Extract function calls and attribute accesses (including imported and qualified),
excluding local function references (e.g., nested functions), into a unified per-function list.
"""

import ast
import builtins
import importlib.util
import json
import os
import sys
from collections import defaultdict

# from call_graph_generation import get_imports_from_file, resolve_module_to_file


class ClassDefVisitor(ast.NodeVisitor):
    def __init__(self):
        self.class_defs = {}

    def visit_ClassDef(self, node):
        class_name = node.name
        if class_name not in self.class_defs:
            self.class_defs[class_name] = (node.lineno, node.end_lineno)
        self.generic_visit(node)


class FuncDefVisitor(ast.NodeVisitor):
    def __init__(self):
        self.func_defs = {}

    def visit_FunctionDef(self, node):
        func_name = node.name
        if func_name not in self.func_defs:
            self.func_defs[func_name] = (node.lineno, node.end_lineno)
        self.generic_visit(node)


class FuncCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.func_calls = {}

    def get_full_attr_name(self, node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_full_attr_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Call):
            return self.get_full_attr_name(node.func)
        else:
            return ast.unparse(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
            if func_name not in self.func_calls:
                self.func_calls[func_name] = node.lineno
            self.generic_visit(node)
        elif isinstance(node.func, ast.Attribute):
            func_name = self.get_full_attr_name(node.func)
            if func_name not in self.func_calls:
                self.func_calls[func_name] = node.lineno
            self.generic_visit(node)


def read_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except (UnicodeDecodeError, FileNotFoundError):
        return None


def get_imported_files(file_path, repo_path):
    imports = get_imports_from_file(file_path, repo_path)
    imported_files = {}
    for import_path in imports:
        file = resolve_module_to_file(import_path, repo_path)
        for name in import_path.get("names", []):
            if name not in imported_files and file is not None:
                imported_files[name] = file
    return imported_files


def map_func_to_class(all_func_defs, all_class_defs, rel_class_name):
    """
    map each function to its enclosing class if applicable.
    returns:
      - func_to_class: mapping from func name → (class name or None, func_lineno, func_endlineno)
      - full_func_list: mapping from full name (qualified with class/module) → (start, end)
    """
    func_to_class = {}

    for func_name, (func_lineno, func_endlineno) in all_func_defs.items():
        matched_class = None
        for class_name, (class_start, class_end) in all_class_defs.items():
            if class_start <= func_lineno <= class_end:
                matched_class = class_name
                break
        func_to_class[func_name] = (matched_class, func_lineno, func_endlineno)

    full_func_list = {}
    for func_name, (class_name, func_lineno, func_endlineno) in func_to_class.items():
        if class_name:
            full_func_name = f"{rel_class_name}.{class_name}.{func_name}"
        else:
            full_func_name = f"{rel_class_name}.{func_name}"
        full_func_list[full_func_name] = (func_lineno, func_endlineno)

    return func_to_class, full_func_list


def check_builtin(func_call):
    if func_call in dir(builtins):
        return True  # f"<builtin:{func_call}>"

    parts = func_call.split(".")
    if parts:
        try:
            top_level_module = parts[0]
            spec = importlib.util.find_spec(top_level_module)
            if spec is None:
                return False  # f"<unresolved:{func_call}>"
            elif "site-packages" in (spec.origin or ""):
                return True  # f"<thirdparty:{func_call}>"
            else:
                return True  # f"<stdlib:{func_call}>"
        except Exception:
            pass

    return False  # f"<unresolved:{func_call}>"


def check_func_call(
    full_func_def_list,
    all_func_calls,
    imported_files,
    rel_class_name,
    local_assignments=None,
):
    resolved_calls = {}
    local_assignments = local_assignments or {}

    for func_call, lineno in all_func_calls.items():
        full_func_call_names = []

        if "." in func_call:
            # e.g., queryset.get
            parts = func_call.split(".")
            prefix, method = ".".join(parts[:-1]), parts[-1]

            # Try to resolve based on imported symbols
            for imported_symbol, imported_path in imported_files.items():
                if imported_symbol.endswith(func_call) or imported_symbol.endswith(
                    f"{prefix}.{method}"
                ):
                    full_func_call_name = imported_symbol
                    full_func_call_names.append(full_func_call_name)
                    break
                if imported_symbol.endswith(prefix):
                    full_func_call_name = f"{imported_symbol}.{method}"
                    full_func_call_names.append(full_func_call_name)
                    break

            # Check if prefix is from a known local assignment (e.g., queryset = _get_queryset(...))
            if prefix in local_assignments:
                source_funcs = (
                    local_assignments[prefix]
                    if isinstance(local_assignments[prefix], list)
                    else [local_assignments[prefix]]
                )

                for source_func in source_funcs:
                    for full_func_def in full_func_def_list:
                        if full_func_def.endswith(source_func):
                            full_func_call_name = full_func_def
                            full_func_call_names.append(full_func_call_name)
                            break
                    else:
                        # if it's imported
                        for imported_symbol, imported_path in imported_files.items():
                            if imported_symbol.endswith(source_func):
                                full_func_call_name = imported_symbol
                                full_func_call_names.append(full_func_call_name)
                                break

        else:
            # Direct call, like setup()
            if f"{rel_class_name}.{func_call}" in full_func_def_list:
                full_func_call_name = f"{rel_class_name}.{func_call}"
                full_func_call_names.append(full_func_call_name)
            else:
                for imported_symbol in imported_files:
                    if imported_symbol.split(".")[-1] == func_call:
                        full_func_call_name = imported_symbol
                        full_func_call_names.append(full_func_call_name)
                        break

                if func_call in local_assignments:
                    # print(f"Found local assignment for {func_call} at line {lineno}")
                    source_funcs = (
                        local_assignments[func_call]
                        if isinstance(local_assignments[func_call], list)
                        else [local_assignments[func_call]]
                    )

                    for source_func in source_funcs:
                        for full_func_def in full_func_def_list:
                            if full_func_def.endswith(source_func):
                                full_func_call_name = full_func_def
                                full_func_call_names.append(full_func_call_name)
                                break
                        else:
                            # if it's imported
                            for (
                                imported_symbol,
                                imported_path,
                            ) in imported_files.items():
                                if imported_symbol.endswith(source_func):
                                    full_func_call_name = imported_symbol
                                    full_func_call_names.append(full_func_call_name)
                                    break

        if len(full_func_call_names) > 0 and not check_builtin(func_call):
            resolved_calls[func_call] = {
                "line": lineno,
                "full_name": full_func_call_names,
            }
        else:
            continue

    return resolved_calls


import ast


def get_local_assignments(source_code):
    """
    extract local and class-level assignments where variables are assigned
    to function calls or directly to symbols, including:
    - a = func(...)
    - a, b = func(...)
    - self.a = func(...)
    - MyClass.a = func(...)
    - a = A if cond else B (both A and B are recorded)
    returns: dict mapping variable names to assigned sources (string or list of strings).
    """
    assignments = {}

    class AssignVisitor(ast.NodeVisitor):
        def visit_Assign(self, node):
            sources = []

            # RHS is a function call
            if isinstance(node.value, ast.Call):
                func_name = get_func_name(node.value.func)
                if func_name:
                    sources.append(func_name)

            # RHS is an if-expression (ternary): A if cond else B
            elif isinstance(node.value, ast.IfExp):
                true_branch = get_name_like_expr(node.value.body)
                false_branch = get_name_like_expr(node.value.orelse)
                sources = [s for s in (true_branch, false_branch) if s]

            # RHS is a name or attribute (e.g., redirect_class = SomeClass)
            elif isinstance(node.value, (ast.Name, ast.Attribute)):
                val = get_name_like_expr(node.value)
                if val:
                    sources.append(val)

            # Assign targets
            for target in node.targets:
                # a = ...
                if isinstance(target, ast.Name):
                    assignments[target.id] = (
                        sources[0] if len(sources) == 1 else sources
                    )
                # a, b = ...
                elif isinstance(target, (ast.Tuple, ast.List)):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            assignments[elt.id] = (
                                sources[0] if len(sources) == 1 else sources
                            )
                # self.a = ... or Class.a = ...
                elif isinstance(target, ast.Attribute):
                    attr_name = get_attribute_full_name(target)
                    if attr_name:
                        assignments[attr_name] = (
                            sources[0] if len(sources) == 1 else sources
                        )

    def get_func_name(func_node):
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return get_attribute_full_name(func_node)
        return None

    def get_name_like_expr(node):
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return get_attribute_full_name(node)
        return None

    def get_attribute_full_name(attr_node):
        parts = []
        while isinstance(attr_node, ast.Attribute):
            parts.insert(0, attr_node.attr)
            attr_node = attr_node.value
        if isinstance(attr_node, ast.Name):
            parts.insert(0, attr_node.id)
            return ".".join(parts)
        return None

    tree = ast.parse(source_code)
    AssignVisitor().visit(tree)
    return assignments


def analyze_function_calls(file_path, repo_path, imported_files):
    rel_class_name = (
        os.path.relpath(file_path, repo_path)
        .replace(".py", "")
        .replace("/__init__", "")
        .replace("/", ".")
        .replace("..", ".")
    )
    code_content = read_file(file_path)
    tree = ast.parse(code_content)

    func_defs_visitor = FuncDefVisitor()
    func_defs_visitor.visit(tree)
    all_func_defs = func_defs_visitor.func_defs

    func_calls_visitor = FuncCallVisitor()
    func_calls_visitor.visit(tree)
    all_func_calls = func_calls_visitor.func_calls

    class_defs_visitor = ClassDefVisitor()
    class_defs_visitor.visit(tree)
    all_class_defs = class_defs_visitor.class_defs

    func_to_class, full_func_def_list = map_func_to_class(
        all_func_defs, all_class_defs, rel_class_name
    )

    # if len(func_to_class) > 0:
    #     print(f"all class definitions in {file_path}: {full_func_def_list}")

    # print(f"all function definitions in {file_path}: {full_func_def_list}")
    # print(f"all function calls in {file_path}: {all_func_calls}")
    # print(f"all imported files in {file_path}: {json.dumps(imported_files, indent=2)}")

    local_assignments = get_local_assignments(code_content)

    # print(f"local assignments in {file_path}: {json.dumps(local_assignments, indent=2)}")

    resolved_calls = check_func_call(
        full_func_def_list,
        all_func_calls,
        imported_files,
        rel_class_name,
        local_assignments,
    )

    caller_callee = get_caller_callee(
        full_func_def_list, resolved_calls, rel_class_name
    )

    return caller_callee


def get_caller_callee(full_func_def_list, resolved_calls, rel_class_name):
    caller_callee = {}
    for func_call, call_info in resolved_calls.items():
        lineno = call_info["line"]
        full_name = call_info["full_name"]
        found = False
        for func_def in full_func_def_list:
            func_def_lineno, func_def_end_lineno = full_func_def_list[func_def]
            if func_def_lineno <= lineno <= func_def_end_lineno:
                if func_def not in caller_callee:
                    caller_callee[func_def] = []
                if func_call not in caller_callee[func_def]:
                    caller_callee[func_def].append(full_name)
                # print(f"[INFO] {rel_class_name} - {func_def} calls {func_call} at line {lineno}")
                found = True
                break
        if not found:
            if rel_class_name not in caller_callee:
                caller_callee[rel_class_name] = []
            if func_call not in caller_callee[rel_class_name]:
                caller_callee[rel_class_name].append(full_name)

    # print(f"[INFO] Caller-callee relationships in {rel_class_name}:")
    # print(json.dumps(caller_callee, indent=2))

    return caller_callee


def extract_call_relationships(repo_path, repo_shortcut):
    all_functions = set()
    function_nodes_by_file = {}
    combined_info = defaultdict(set)
    main_path = os.path.join(repo_path, repo_shortcut)
    all_caller_callee = {}
    for root, _, files in os.walk(main_path):
        for file in files:
            if file.endswith(".py"):
                try:
                    abs_path = os.path.join(root, file)
                    rel_path = os.path.relpath(abs_path, repo_path)
                    imported_files = get_imported_files(abs_path, repo_path)
                    # print(f"[INFO] Processing file: {abs_path} with imports: {imported_files}")
                    caller_callee = analyze_function_calls(
                        abs_path, repo_path, imported_files
                    )

                    for caller, callee in caller_callee.items():
                        if caller not in all_caller_callee:
                            all_caller_callee[caller] = []
                        for callee_func in callee:
                            all_caller_callee[caller].extend(callee_func)
                            all_caller_callee[caller] = list(
                                set(all_caller_callee[caller])
                            )  # Remove duplicates
                except Exception as e:
                    print(f"[ERROR] Failed to process {abs_path}: {e}")
                    continue
    return all_caller_callee


def save_combined_graph(combined_info, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined_info, f, indent=4)
    print(f"[INFO] Merged function graph saved to {out_path}")


def get_call_paths(repo_path, result_json, repo_shortcut):
    print(f"[INFO] Extracting function calls from {repo_path}...")
    combined_info = extract_call_relationships(repo_path, repo_shortcut)
    save_combined_graph(combined_info, result_json)


if __name__ == "__main__":
    repo_path = sys.argv[1]
    result_json = sys.argv[2]  # "combined_graph.json"
    repo_shortcut = sys.argv[3]  # django
    get_call_paths(repo_path, result_json, repo_shortcut)
