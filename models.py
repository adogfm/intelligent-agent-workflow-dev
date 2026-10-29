"""Structured input/output models."""

from __future__ import annotations

import ast
import textwrap
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


Severity = Literal["critical", "high", "medium", "low"]


class GeneratedCodeSyntaxError(ValueError):
    """Typed validation failure for syntactically invalid generated Python."""

    def __init__(self, code: str, reason: str) -> None:
        super().__init__(f"生成代码不是有效的 Python 源代码：{reason}")
        self.code = code
        self.reason = reason


def generated_code_syntax_error(error: ValidationError) -> GeneratedCodeSyntaxError | None:
    """Find the typed code error inside Pydantic's structured diagnostics."""

    for detail in error.errors(include_url=False):
        cause = detail.get("ctx", {}).get("error")
        if detail.get("loc") == ("code",) and isinstance(cause, GeneratedCodeSyntaxError):
            return cause
    return None


class CodeGenerationResult(BaseModel):
    """One Python source artifact and its implementation summary."""

    code: str = Field(strict=True, min_length=1, description="生成的完整 Python 源代码")
    summary: str = Field(strict=True, min_length=1, description="实现说明与必要假设")

    @field_validator("code")
    @classmethod
    def validate_python_code(cls, code: str) -> str:
        if not code.strip():
            raise ValueError("生成代码不能为空")
        try:
            tree = ast.parse(code)
            # Parsing alone accepts invalid scopes such as a top-level return.
            # Compile without executing source or inheriting this module's flags.
            compile(tree, "generated_code.py", "exec", dont_inherit=True)
        except (SyntaxError, ValueError) as exc:
            raise GeneratedCodeSyntaxError(code, str(exc)) from exc
        if not tree.body:
            raise ValueError("生成代码必须包含 Python 语句，不能只有注释")
        return code

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, summary: str) -> str:
        if not summary.strip():
            raise ValueError("实现说明不能为空")
        return summary.strip()


class TestCase(BaseModel):
    """One generated unittest method body; ``self`` and ``module`` are available."""

    name: str = Field(strict=True, pattern=r"^test_[A-Za-z0-9_]+$")
    code: str = Field(strict=True, min_length=1)

    @field_validator("code")
    @classmethod
    def validate_body(cls, code: str) -> str:
        code = textwrap.dedent(code).strip()
        if not code:
            raise ValueError("测试代码不能为空")
        try:
            tree = ast.parse("def test_case(self):\n" + textwrap.indent(code, "    "))
        except (SyntaxError, ValueError) as exc:
            raise ValueError(f"测试代码语法无效：{exc}") from exc
        if not tree.body[0].body:
            raise ValueError("测试代码必须包含 Python 语句")
        return code


class TestGenerationResult(BaseModel):
    test_cases: list[TestCase] = Field(min_length=1)
    summary: str = Field(strict=True, min_length=1)

    @model_validator(mode="after")
    def validate_unique_names(self) -> TestGenerationResult:
        names = [case.name for case in self.test_cases]
        if len(names) != len(set(names)):
            raise ValueError("测试用例名称不能重复")
        if not self.summary.strip():
            raise ValueError("测试生成说明不能为空")
        return self


class TestExpectation(BaseModel):
    """An expected string copied from an original unittest assertion."""

    assertion: str
    expected: str
    kind: Literal["exact", "regex", "contains"]


class TestFailure(BaseModel):
    test: str
    traceback: str
    test_name: str = ""
    test_code: str = ""
    expectations: list[TestExpectation] = Field(default_factory=list)


class TestExecutionResult(BaseModel):
    """Actual subprocess outcome, independent of workflow lifecycle status."""

    status: Literal["passed", "failed", "error", "timeout"]
    tests_run: int = Field(default=0, ge=0)
    passed: int = Field(default=0, ge=0)
    failed: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    expected_failures: int = Field(default=0, ge=0)
    unexpected_successes: int = Field(default=0, ge=0)
    failures: list[TestFailure] = Field(default_factory=list)
    error_details: list[TestFailure] = Field(default_factory=list)
    test_expectations: dict[str, list[TestExpectation]] = Field(default_factory=dict)
    stdout: str = ""
    stderr: str = ""
    duration_seconds: float = Field(default=0, ge=0)
    return_code: int | None = None
    message: str = ""

    def to_text(self) -> str:
        lines = [
            "\n=== Test Agent 测试结果 ===",
            f"状态：{self.status}；运行 {self.tests_run}，通过 {self.passed}，"
            f"失败 {self.failed}，错误 {self.errors}，跳过 {self.skipped}",
        ]
        if self.message:
            lines.append(self.message)
        for failure in [*self.failures, *self.error_details]:
            lines.append(f"{failure.test}:\n{failure.traceback}")
            for expectation in failure.expectations:
                lines.append(
                    f"原测试期望（{expectation.assertion}/{expectation.kind}）："
                    f"{expectation.expected!r}"
                )
        if self.stdout:
            lines.append(f"标准输出：\n{self.stdout}")
        if self.stderr:
            lines.append(f"标准错误：\n{self.stderr}")
        return "\n".join(lines)


class ReviewContext(BaseModel):
    """Current task and test evidence supplied to a contextual Reviewer."""

    requirements: str = Field(default="", strict=True)
    context: str = Field(default="", strict=True)
    test_result: TestExecutionResult | None = None
    test_cases: list[TestCase] = Field(default_factory=list)


class ReviewIssue(BaseModel):
    problem: str = Field(description="发现的具体问题")
    severity: Severity = Field(description="问题严重程度")
    reason: str = Field(description="该问题为何重要")
    suggestion: str = Field(description="具体、可执行的修改建议")
    category: Literal["requirement", "test_failure", "code_defect", "security_standard", "improvement"] = "code_defect"
    scope: Literal["in_scope", "out_of_scope"] = "in_scope"
    evidence: str = ""
    requirement_ref: str = ""


class ReviewReport(BaseModel):
    summary: str = Field(description="整体审查结论")
    issues: list[ReviewIssue] = Field(description="发现的问题；没有问题时为空列表")
    recommendations: list[ReviewIssue] = Field(default_factory=list, description="可选建议，不阻断修复循环")
    score: int | None = Field(default=None, ge=0, le=100, description="Agent 计算的需求/测试/缺陷评分")
    test_status: Literal["not_run", "passed", "failed", "error", "timeout"] = "not_run"

    def to_text(self) -> str:
        lines = ["\n=== Python 代码审查结果 ===", self.summary]
        if self.score is not None:
            lines.append(f"审查评分：{self.score}/100；测试状态：{self.test_status}")
        if not self.issues:
            lines.append("\n未发现明确问题。")

        for index, issue in enumerate(self.issues, start=1):
            lines.extend(
                [
                    f"\n{index}. [{issue.severity.upper()}] {issue.problem}",
                    f"   原因：{issue.reason}",
                    f"   建议：{issue.suggestion}",
                ]
            )
        if self.recommendations:
            lines.append("\n=== 可选建议（不阻断 Workflow） ===")
            for item in self.recommendations:
                lines.append(f"- {item.problem}：{item.suggestion}")
        return "\n".join(lines)
