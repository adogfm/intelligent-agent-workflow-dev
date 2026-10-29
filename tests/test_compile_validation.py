"""Reject parseable Python that cannot compile before publishing generated code."""

from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from pydantic import ValidationError

from agent import ReviewAgent
from code_checker import CodeChecker
from coding_agent import CodingAgent, create_coding_workflow
from llm import LLMCodeSyntaxError, OpenAICodeGenerator, parse_code_generation
from main import main
from models import CodeGenerationResult, ReviewReport, generated_code_syntax_error
from workflow import WorkflowExecutionError, WorkflowState, WorkflowStatus


INVALID = (
    ("return", "\nreturn 1\n", 2),
    ("break", "break\n", 1),
    ("continue", "continue\n", 1),
    ("await", "await run()\n", 1),
    ("yield", "yield 1\n", 1),
    ("nonlocal", "def work():\n    nonlocal missing\n", 2),
    ("duplicate_arguments", "def work(value, value):\n    return value\n", 1),
    ("future_position", "value = 1\nfrom __future__ import annotations\n", 2),
)
GOOD = "def add(a, b):\n    return a + b\n"


class CompileValidationTests(unittest.TestCase):
    def test_generation_model_rejects_compile_errors_with_original_diagnostics(self):
        for name, code, _ in INVALID:
            with self.subTest(name=name):
                ast.parse(code)  # These inputs specifically reproduce the AST-only gap.
                with self.assertRaises(ValidationError) as caught:
                    CodeGenerationResult(code=code, summary="生成代码")
                syntax = generated_code_syntax_error(caught.exception)
                self.assertIsNotNone(syntax)
                self.assertEqual(syntax.code, code)
                self.assertTrue(syntax.reason)

    def test_json_parser_preserves_typed_compile_errors(self):
        for name, code, _ in INVALID:
            payload = json.dumps({"code": code, "summary": "生成代码"})
            for content in (payload, f"```json\n{payload}\n```"):
                with self.subTest(name=name, fenced=content.startswith("```")):
                    with self.assertRaises(LLMCodeSyntaxError) as caught:
                        parse_code_generation(content)
                    self.assertEqual(caught.exception.generated_code, code)
                    self.assertTrue(caught.exception.reason)

    def test_checker_reports_compile_errors_with_line_numbers(self):
        for name, code, line in INVALID:
            with self.subTest(name=name):
                findings = CodeChecker().check(code)
                self.assertEqual([item.rule_id for item in findings], ["syntax-error"])
                self.assertEqual(findings[0].severity, "high")
                self.assertEqual(findings[0].line, line)

    def test_valid_scopes_remain_accepted(self):
        valid = (
            GOOD,
            "for item in range(3):\n    break\n",
            "for item in range(3):\n    continue\n",
            "async def work():\n    return await run()\n",
            "def values():\n    yield 1\n",
            "def outer():\n    value = 1\n    def inner():\n        nonlocal value\n        value += 1\n",
            "from __future__ import annotations\ndef work(value: Unknown) -> Unknown:\n    return value\n",
        )
        for code in valid:
            with self.subTest(code=code):
                self.assertEqual(CodeGenerationResult(code=code, summary="合法代码").code, code)
                self.assertEqual(CodeChecker().check(code), [])

    def test_validation_does_not_execute_source(self):
        code = "raise RuntimeError('validation must not execute source')\n"
        self.assertEqual(CodeGenerationResult(code=code, summary="仅编译").code, code)
        self.assertEqual(CodeChecker().check(code), [])

    def test_injected_provider_errors_preserve_state_and_skip_downstream_review(self):
        for name, code, _ in INVALID:
            for mode in ("unvalidated", "mutated", "construct_inside_provider"):
                with self.subTest(name=name, mode=mode):
                    def generate(*args, **kwargs):
                        if mode == "construct_inside_provider":
                            return CodeGenerationResult(code=code, summary="无效代码")
                        if mode == "mutated":
                            result = CodeGenerationResult(code=GOOD, summary="供应商修改结果")
                            result.code = code
                            return result
                        return CodeGenerationResult.model_construct(code=code, summary="绕过校验")

                    generator = Mock()
                    generator.generate.side_effect = generate
                    reviewer = Mock()
                    reviewer.review.return_value = ReviewReport(summary="审查通过", issues=[])
                    previous_report = ReviewReport(summary="旧代码的审查结果", issues=[])
                    state = WorkflowState({
                        "requirements": "实现加法", "code": GOOD, "report": previous_report,
                    })
                    workflow = create_coding_workflow(CodingAgent(generator), ReviewAgent(reviewer))
                    with self.assertRaises(WorkflowExecutionError) as caught:
                        workflow.run(state)
                    cause = caught.exception.__cause__
                    self.assertIsInstance(cause, LLMCodeSyntaxError)
                    self.assertEqual(cause.generated_code, code)
                    self.assertTrue(cause.reason)
                    self.assertEqual(state.status, WorkflowStatus.FAILED)
                    self.assertEqual(state.get("code"), GOOD)
                    self.assertIs(state.get("report"), previous_report)
                    self.assertIsNone(state.get("generation_result"))
                    self.assertEqual(state.version, 0)
                    self.assertEqual(state.history, ())
                    reviewer.review.assert_not_called()

    def test_remote_generation_surfaces_compile_errors_without_retrying(self):
        client = Mock()
        with patch("llm.load_dotenv"):
            generator = OpenAICodeGenerator(api_key="test-key", client=client)
        for name, code, _ in INVALID:
            with self.subTest(name=name):
                client.reset_mock()
                client.chat.completions.create.return_value = SimpleNamespace(choices=[
                    SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                        refusal=None, content=json.dumps({"code": code, "summary": "生成代码"}),
                    )),
                ])
                with self.assertRaises(LLMCodeSyntaxError) as caught:
                    generator.generate("实现加法")
                self.assertEqual(caught.exception.generated_code, code)
                client.chat.completions.create.assert_called_once()

    def test_generation_cli_reports_compile_failure_with_exit_one(self):
        for name, code, _ in INVALID:
            with self.subTest(name=name):
                generator = Mock()
                generator.generate.return_value = CodeGenerationResult.model_construct(
                    code=code, summary="无效的生成代码",
                )
                reviewer = Mock()
                reviewer.review.return_value = ReviewReport(summary="审查通过", issues=[])
                errors = io.StringIO()
                with patch("sys.argv", ["main.py", "--requirements", "实现加法"]), \
                        patch("main.create_code_generator", return_value=generator), \
                        patch("main.create_reviewer", return_value=reviewer), \
                        redirect_stdout(io.StringIO()), redirect_stderr(errors):
                    self.assertEqual(main(), 1)
                self.assertIn("LLM 代码生成失败", errors.getvalue())
                reviewer.review.assert_not_called()


if __name__ == "__main__":
    unittest.main()
