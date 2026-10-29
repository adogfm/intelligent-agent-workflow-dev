"""Local AST-based checks used as a deterministic Agent tool."""

from __future__ import annotations

import ast
from dataclasses import dataclass

from models import ReviewIssue, Severity


@dataclass(frozen=True)
class CodeCheckFinding:
    """One deterministic finding produced by CodeChecker."""

    rule_id: str
    line: int | None
    problem: str
    severity: Severity
    reason: str
    suggestion: str

    def to_review_issue(self) -> ReviewIssue:
        location = f"第 {self.line} 行：" if self.line is not None else ""
        return ReviewIssue(
            problem=f"{location}{self.problem}",
            severity=self.severity,
            reason=self.reason,
            suggestion=self.suggestion,
        )


class CodeChecker:
    """Parse Python source without executing it and run basic AST checks."""

    def check(self, code: str) -> list[CodeCheckFinding]:
        try:
            tree = ast.parse(code)
            compile(tree, "reviewed_code.py", "exec", dont_inherit=True)
        except SyntaxError as exc:
            return [
                CodeCheckFinding(
                    rule_id="syntax-error",
                    line=exc.lineno,
                    problem=f"语法错误：{exc.msg}",
                    severity="high",
                    reason="代码未通过 Python 语法解析或编译，因此不能正常运行。",
                    suggestion="根据报错行和偏移位置修正语法后重新审查。",
                )
            ]

        visitor = _CodeCheckVisitor()
        visitor.visit(tree)
        return visitor.findings


class _CodeCheckVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[CodeCheckFinding] = []

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is None:
            self.findings.append(
                CodeCheckFinding(
                    rule_id="bare-except",
                    line=node.lineno,
                    problem="使用了裸 except",
                    severity="medium",
                    reason="裸 except 会捕获 KeyboardInterrupt、SystemExit 等异常，可能掩盖程序退出信号和真实故障。",
                    suggestion="只捕获预期异常，例如 `except ValueError:`；确需兜底时使用 `except Exception:` 并记录异常。",
                )
            )
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if (
            isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod))
            and _is_zero_literal(node.right)
            and not (isinstance(node.op, ast.Mod) and _text_result_type(node.left) is not None)
        ):
            operator = {
                ast.Div: "/",
                ast.FloorDiv: "//",
                ast.Mod: "%",
            }[type(node.op)]
            self.findings.append(
                CodeCheckFinding(
                    rule_id="division-by-zero",
                    line=node.lineno,
                    problem=f"运算符 `{operator}` 的右操作数是常量零",
                    severity="high",
                    reason=(
                        "数值取模以零为右操作数时会抛出 ZeroDivisionError；"
                        "左操作数类型不明确时需结合运行时类型判断。"
                        if isinstance(node.op, ast.Mod) else
                        "执行到该表达式时会抛出 ZeroDivisionError。"
                    ),
                    suggestion="修正分母来源，或在运算前检查分母不为零并明确处理零值情况。",
                )
            )
        self.generic_visit(node)


def _text_result_type(node: ast.expr) -> type[str] | type[bytes] | None:
    """Recognize builtin text expressions without evaluating user code."""

    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        return str if isinstance(node.value, str) else bytes
    if isinstance(node, ast.JoinedStr):
        return str
    if isinstance(node, ast.NamedExpr):
        return _text_result_type(node.value)
    if isinstance(node, ast.IfExp):
        kind = _text_result_type(node.body)
        return kind if kind is not None and kind is _text_result_type(node.orelse) else None
    if isinstance(node, ast.BinOp):
        left = _text_result_type(node.left)
        if isinstance(node.op, ast.Add):
            return left if left is not None and left is _text_result_type(node.right) else None
        if isinstance(node.op, ast.Mult):
            if left is not None and _is_integer_literal(node.right):
                return left
            if _is_integer_literal(node.left):
                return _text_result_type(node.right)
        if isinstance(node.op, ast.Mod) and left is not None:
            # An unknown RHS may overload __rmod__ and return a numeric value.
            if (
                isinstance(node.right, (ast.Constant, ast.JoinedStr, ast.Tuple, ast.List, ast.Set, ast.Dict))
                or _is_integer_literal(node.right)
                or _text_result_type(node.right) is not None
            ):
                return left
    return None


def _is_integer_literal(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, int)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_integer_literal(node.operand)
    return False


def _is_zero_literal(node: ast.expr) -> bool:
    """Return True for numeric zero literals, including signed zero."""

    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float, complex)) and not isinstance(node.value, bool) and node.value == 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        return _is_zero_literal(node.operand)
    return False
