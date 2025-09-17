import ast
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Set

from tqdm import tqdm


class TestMethodFinder:
    def __init__(self, repo_path: str, method: str):
        self.repo_path = Path(repo_path)
        self.method = method
        self.method_short_names = {method.split(".")[-1]}
        self.test_dirs = ["test", "tests"]
        self.test_functions: List[str] = []

    def is_test_file(self, filepath: Path) -> bool:
        return (
            any(d in filepath.parts for d in self.test_dirs)
            and filepath.name.startswith("test")
            and filepath.suffix == ".py"
        )

    def find_tests(self):
        for filepath in self.repo_path.rglob("*.py"):
            if not self.is_test_file(filepath):
                continue
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    tree = ast.parse(f.read(), filename=str(filepath))
                    self._extract_test_methods(tree, filepath)
            except Exception as e:
                print(f"[WARN] Failed parsing {filepath}: {e}")

    def _extract_test_methods(self, tree: ast.AST, filepath: Path):
        class TestVisitor(ast.NodeVisitor):
            def __init__(self, target_names: Set[str]):
                self.target_names = target_names
                self.matches: List[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef):
                if not node.name.startswith("test"):
                    return
                func_calls = set()
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Attribute):
                            func_calls.add(child.func.attr)
                        elif isinstance(child.func, ast.Name):
                            func_calls.add(child.func.id)
                if self.target_names & func_calls:
                    self.matches.append(node.name)

        visitor = TestVisitor(self.method_short_names)
        visitor.visit(tree)
        if visitor.matches:
            mod_path = (
                filepath.relative_to(self.repo_path)
                .with_suffix("")
                .as_posix()
                .replace("/", ".")
            )
            for match in visitor.matches:
                full_name = f"{mod_path}.{match}"
                self.test_functions.append(full_name)

    def get_results(self) -> List[str]:
        return self.test_functions


def process_method(repo_path, method):
    finder = TestMethodFinder(repo_path, method)
    finder.find_tests()
    return method, finder.get_results()


def locate_tests(
    repo_path: str, methods: List[str], logger=None
) -> Dict[str, List[str]]:
    results: Dict[str, List[str]] = {}
    print(f"Locating tests for methods: {len(methods)}")
    max_workers = 1

    task_args = [(repo_path, method) for method in methods]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_method, *args) for args in task_args]
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing"
        ):
            method, tests = future.result()
            results[method] = tests

    if logger:
        logger.info(f"Potential Tests: {json.dumps(results, indent=2)}")
    return results
