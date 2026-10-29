"""Offline recovery from initial generation syntax and compilation failures."""

from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import httpx
from openai import APIConnectionError, AuthenticationError, RateLimitError

from agent import CodeReviewAgent
from coding_agent import CodingAgent, create_coding_workflow
from fix_agent import FixAgent
from llm import LLMCodeSyntaxError, LLMGenerationError, OpenAICodeGenerator
from main import main
from models import (
    CodeGenerationResult, ReviewIssue, ReviewReport, TestCase as GeneratedCase,
    TestExecutionResult as ExecutionResult, TestFailure as FailureDetail,
    TestGenerationResult as GenerationResult,
)
from test_agent import TestAgent as WorkflowTestAgent
from workflow import WorkflowExecutionError, WorkflowState, WorkflowStatus


ORIGINAL = "def add(a, b):\n    return a - b\n"
VALID = "def add(a, b):\n    return a + b\n"
INVALID = "def add(a, b)\n    return a + b\n"
CONTEXT = "保留 add 接口，不引入第三方依赖。"
CASES = [GeneratedCase(name="test_add", code="self.assertEqual(module.add(2, 3), 5)")]
COMPILE_INVALID = (
    "return 1\n", "break\n", "continue\n", "await run()\n", "yield 1\n",
    "def work():\n    nonlocal missing\n",
    "def work(value, value):\n    return value\n",
    "value = 1\nfrom __future__ import annotations\n",
)


def completion(code: str) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="stop", message=SimpleNamespace(
            refusal=None, content=json.dumps({"code": code, "summary": "生成加法"}),
        ),
    )])


class CodingSyntaxRetryTests(unittest.TestCase):
    def remote(self, responses):
        client = Mock()
        client.chat.completions.create.side_effect = responses
        with patch("llm.load_dotenv"), patch("llm.OpenAI", side_effect=AssertionError("Real SDK client forbidden")):
            generator = OpenAICodeGenerator(
                api_key="offline-test-key", base_url="https://example.invalid/v1",
                model="offline-coding-retry", client=client,
            )
        return generator, client

    def downstream(self):
        reviewer = Mock()
        reviewer.is_mock = False
        reviewer.review.return_value = ReviewReport(summary="审查通过", issues=[])
        tests = Mock()
        tests.generate.return_value = GenerationResult(test_cases=CASES, summary="验证加法")
        runner = Mock()
        runner.run.return_value = ExecutionResult(status="passed", tests_run=1, passed=1)
        fixer = Mock()
        fixer.fix.return_value = CodeGenerationResult(code=VALID, summary="修正加法")
        return reviewer, tests, runner, fixer

    def test_syntax_retry_preserves_requirement_and_context_with_diagnostics(self):
        generator, client = self.remote([completion(INVALID), completion(VALID)])
        result = CodingAgent(generator).generate("  实现相加  ", context=CONTEXT)
        self.assertIsInstance(result, CodeGenerationResult)
        self.assertEqual(result.code, VALID)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        first, retry = [json.loads(call.kwargs["messages"][1]["content"])
                        for call in client.chat.completions.create.call_args_list]
        self.assertEqual(first, {"requirements": "实现相加", "context": CONTEXT})
        self.assertEqual(retry["requirements"], first["requirements"])
        self.assertTrue(retry["context"].startswith(CONTEXT + "\n\n"))
        diagnostics = json.loads(retry["context"].rsplit("\n", 1)[1])
        self.assertEqual(diagnostics["syntax_retry"], 1)
        self.assertEqual(diagnostics["invalid_generated_code"], INVALID)
        with self.assertRaises(SyntaxError) as caught:
            ast.parse(INVALID)
        self.assertEqual(diagnostics["syntax_error"], str(caught.exception))

    def test_success_on_last_retry_uses_only_latest_diagnostic(self):
        invalid_codes = (INVALID, "return 1\n", "break\n")
        generator, client = self.remote([completion(code) for code in (*invalid_codes, VALID)])
        result = CodingAgent(generator).generate("实现相加", context=CONTEXT)
        self.assertEqual(result.code, VALID)
        self.assertEqual(client.chat.completions.create.call_count, 4)
        for retry, call in enumerate(client.chat.completions.create.call_args_list[1:], 1):
            payload = json.loads(call.kwargs["messages"][1]["content"])
            prefix, data = payload["context"].rsplit("\n", 1)
            self.assertTrue(prefix.startswith(CONTEXT + "\n\n"))
            self.assertEqual(prefix.count("以下为上一次生成"), 1)
            diagnostics = json.loads(data)
            self.assertEqual(diagnostics["syntax_retry"], retry)
            self.assertEqual(diagnostics["invalid_generated_code"], invalid_codes[retry - 1])
            self.assertEqual(payload["requirements"], "实现相加")

    def test_exhaustion_preserves_artifacts_and_never_runs_downstream(self):
        invalid_codes = (INVALID, "return 1\n", "break\n", COMPILE_INVALID[6])
        generator, client = self.remote([completion(code) for code in (*invalid_codes, VALID)])
        reviewer, tests, runner, fixer = self.downstream()
        previous_generation = CodeGenerationResult(code=ORIGINAL, summary="旧生成结果")
        previous_report = ReviewReport(summary="旧审查结果", issues=[])
        state = WorkflowState({
            "requirements": "实现相加", "context": CONTEXT, "code": ORIGINAL,
            "generation_result": previous_generation, "report": previous_report, "test_cases": CASES,
        })
        workflow = create_coding_workflow(
            CodingAgent(generator), CodeReviewAgent(reviewer),
            test_agent=WorkflowTestAgent(tests, runner), fix_agent=FixAgent(fixer),
        )
        with self.assertRaises(WorkflowExecutionError) as caught:
            workflow.run(state)
        cause = caught.exception.__cause__
        self.assertIsInstance(cause, LLMCodeSyntaxError)
        self.assertEqual(cause.generated_code, invalid_codes[-1])
        self.assertIn("duplicate argument", cause.reason)
        self.assertIn("4 次生成调用", str(cause))
        self.assertIn("3 次重试", str(cause))
        self.assertEqual(client.chat.completions.create.call_count, 4)
        self.assertEqual(state.status, WorkflowStatus.FAILED)
        self.assertEqual(state.current_step, "coding")
        self.assertEqual(state.get("code"), ORIGINAL)
        self.assertIs(state.get("generation_result"), previous_generation)
        self.assertIs(state.get("report"), previous_report)
        self.assertEqual(state.get("context"), CONTEXT)
        self.assertEqual(state.get("test_cases"), CASES)
        self.assertEqual(state.version, 0)
        self.assertEqual(state.history, ())
        reviewer.review.assert_not_called()
        tests.generate.assert_not_called()
        runner.run.assert_not_called()
        fixer.fix.assert_not_called()

    def test_injected_provider_failures_retry_before_one_atomic_commit(self):
        for mode in ("unvalidated", "mutated", "construct_inside_provider"):
            with self.subTest(mode=mode):
                generator = Mock()
                state = WorkflowState({"requirements": "实现相加", "context": CONTEXT, "code": ORIGINAL})
                calls = 0

                def generate(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    self.assertEqual(state.get("code"), ORIGINAL)
                    self.assertIsNone(state.get("generation_result"))
                    self.assertEqual(state.version, 0)
                    if calls > 1:
                        return CodeGenerationResult(code=VALID, summary="合法代码")
                    if mode == "construct_inside_provider":
                        return CodeGenerationResult(code=INVALID, summary="无效代码")
                    if mode == "mutated":
                        result = CodeGenerationResult(code=VALID, summary="修改后的结果")
                        result.code = INVALID
                        return result
                    return CodeGenerationResult.model_construct(code=INVALID, summary="绕过校验")

                generator.generate.side_effect = generate
                run = create_coding_workflow(CodingAgent(generator)).run(state)
                self.assertEqual(generator.generate.call_count, 2)
                self.assertEqual(state.get("code"), VALID)
                self.assertEqual(state.get("context"), CONTEXT)
                self.assertEqual(state.get("generation_result"), run.output)
                self.assertEqual(state.version, 1)
                self.assertEqual(len(state.history), 1)
                self.assertEqual([item.step for item in run.executions], ["coding"])

    def test_compile_errors_not_detected_by_ast_are_retried(self):
        for code in COMPILE_INVALID:
            with self.subTest(code=code):
                generator, client = self.remote([completion(code), completion(VALID)])
                result = CodingAgent(generator).generate("实现相加")
                self.assertEqual(result.code, VALID)
                self.assertEqual(client.chat.completions.create.call_count, 2)
                payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
                diagnostics = json.loads(payload["context"].rsplit("\n", 1)[1])
                self.assertEqual(diagnostics["invalid_generated_code"], code)
                self.assertTrue(diagnostics["syntax_error"])

    def test_non_syntax_sdk_failures_are_not_retried(self):
        request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
        responses = (
            APIConnectionError(request=request),
            AuthenticationError("invalid key", response=httpx.Response(401, request=request), body=None),
            RateLimitError("rate limit", response=httpx.Response(429, request=request), body=None),
            SimpleNamespace(choices=[]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="length")]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="content_filter")]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(refusal="拒绝"))]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(refusal=None, content=None))]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(refusal=None, content="not-json"))]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                refusal=None, content=json.dumps({"code": 123, "summary": "类型不正确"}),
            ))]),
            SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                refusal=None, content=json.dumps({"code": VALID, "summary": " "}),
            ))]),
        )
        for response in responses:
            with self.subTest(response=response):
                generator, client = self.remote([response, completion(VALID)])
                with self.assertRaises(LLMGenerationError) as caught:
                    CodingAgent(generator).generate("实现相加")
                self.assertNotIsInstance(caught.exception, LLMCodeSyntaxError)
                self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_injected_non_syntax_validation_errors_are_not_retried(self):
        for result in (
            "not a CodeGenerationResult",
            CodeGenerationResult.model_construct(code=VALID, summary=" "),
            CodeGenerationResult.model_construct(summary="缺少代码字段"),
        ):
            with self.subTest(result=result):
                generator = Mock()
                generator.generate.side_effect = [result, CodeGenerationResult(code=VALID, summary="合法代码")]
                with self.assertRaises(LLMGenerationError) as caught:
                    CodingAgent(generator).generate("实现相加")
                self.assertNotIsInstance(caught.exception, LLMCodeSyntaxError)
                generator.generate.assert_called_once()

    def test_non_syntax_error_after_syntax_failure_stops_immediately(self):
        generator, client = self.remote([
            completion(INVALID), LLMGenerationError("服务不可用"), completion(VALID),
        ])
        state = WorkflowState({"requirements": "实现相加", "code": ORIGINAL})
        with self.assertRaises(WorkflowExecutionError) as caught:
            create_coding_workflow(CodingAgent(generator)).run(state)
        self.assertNotIsInstance(caught.exception.__cause__, LLMCodeSyntaxError)
        self.assertEqual(str(caught.exception.__cause__), "服务不可用")
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertEqual(state.get("code"), ORIGINAL)
        self.assertEqual(state.version, 0)

    def test_retry_limit_is_configurable_and_zero_disables_retries(self):
        for limit in (0, 1, 2):
            with self.subTest(limit=limit):
                generator, client = self.remote([completion(INVALID)] * (limit + 1) + [completion(VALID)])
                with self.assertRaises(LLMCodeSyntaxError) as caught:
                    CodingAgent(generator, max_syntax_retries=limit).generate("实现相加")
                self.assertEqual(client.chat.completions.create.call_count, limit + 1)
                self.assertIn(f"最多 {limit} 次重试", str(caught.exception))
        for value in (-1, True, False, 1.5, "3", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                CodingAgent(Mock(), max_syntax_retries=value)

    def test_first_pass_and_validation_without_execution_keep_existing_behavior(self):
        for code in (VALID, "raise RuntimeError('validation must not execute source')\n"):
            with self.subTest(code=code):
                generator, client = self.remote([completion(code)])
                result = CodingAgent(generator, max_syntax_retries=0).generate("实现需求", context=CONTEXT)
                self.assertEqual(result.code, code)
                client.chat.completions.create.assert_called_once()
                payload = json.loads(client.chat.completions.create.call_args.kwargs["messages"][1]["content"])
                self.assertEqual(payload, {"requirements": "实现需求", "context": CONTEXT})

    def test_recovered_generation_continues_without_consuming_repair_budget(self):
        for limit in (0, 1, 3):
            with self.subTest(limit=limit):
                generator, client = self.remote([completion(INVALID), completion(VALID)])
                reviewer, tests, runner, fixer = self.downstream()
                run = create_coding_workflow(
                    CodingAgent(generator), CodeReviewAgent(reviewer),
                    test_agent=WorkflowTestAgent(tests, runner), fix_agent=FixAgent(fixer), max_fix_attempts=limit,
                ).run({"requirements": "实现相加"})
                self.assertEqual([item.step for item in run.executions], ["coding", "review", "test"])
                self.assertEqual(client.chat.completions.create.call_count, 2)
                self.assertEqual(run.state.get("fix_attempts"), 0)
                self.assertEqual(run.state.get("max_fix_attempts"), limit)
                self.assertEqual(run.state.get("repair_stop_reason"), "resolved")
                self.assertTrue(run.report.succeeded)
                reviewer.review.assert_called_once()
                tests.generate.assert_called_once()
                runner.run.assert_called_once_with(VALID, CASES)
                fixer.fix.assert_not_called()

    def test_recovered_generation_leaves_business_repair_attempt_available(self):
        generator, client = self.remote([completion(INVALID), completion(ORIGINAL)])
        reviewer, tests, runner, fixer = self.downstream()
        reviewer.review.side_effect = lambda code, findings, **kwargs: ReviewReport(
            summary="审查加法", issues=[] if code == VALID else [ReviewIssue(
                problem="使用减法", severity="high", reason="不符合加法需求", suggestion="改为相加",
            )],
        )
        runner.run.side_effect = lambda code, cases: ExecutionResult(
            status="passed" if code == VALID else "failed", tests_run=1,
            passed=int(code == VALID), failed=int(code != VALID),
            failures=[] if code == VALID else [FailureDetail(test="test_add", traceback="AssertionError")],
        )
        run = create_coding_workflow(
            CodingAgent(generator), CodeReviewAgent(reviewer),
            test_agent=WorkflowTestAgent(tests, runner), fix_agent=FixAgent(fixer), max_fix_attempts=1,
        ).run({"requirements": "实现相加"})
        self.assertEqual([item.step for item in run.executions], ["coding", "review", "test", "fix", "test", "review"])
        self.assertEqual(client.chat.completions.create.call_count, 2)
        fixer.fix.assert_called_once()
        tests.generate.assert_called_once()
        self.assertEqual(run.state.get("fix_attempts"), 1)
        self.assertEqual(run.state.get("code"), VALID)
        self.assertEqual(run.state.get("generation_result").code, ORIGINAL)
        self.assertTrue(run.report.succeeded)

    def test_reuse_starts_each_generation_with_original_context(self):
        generator, client = self.remote([completion(INVALID), completion(VALID)] * 2)
        agent = CodingAgent(generator, max_syntax_retries=1)
        for context in ("第一次任务", "第二次任务"):
            agent.generate("实现相加", context=context)
        requests = [json.loads(call.kwargs["messages"][1]["content"])
                    for call in client.chat.completions.create.call_args_list]
        self.assertEqual(requests[2]["context"], "第二次任务")
        self.assertTrue(requests[3]["context"].startswith("第二次任务\n\n"))
        self.assertNotIn("第一次任务", requests[3]["context"])
        self.assertEqual(json.loads(requests[3]["context"].rsplit("\n", 1)[1])["syntax_retry"], 1)

    def test_cli_continues_after_recovery_and_returns_one_on_exhaustion(self):
        for recovered in (True, False):
            with self.subTest(recovered=recovered):
                responses = [completion(INVALID)] * (1 if recovered else 4) + [completion(VALID)]
                generator, client = self.remote(responses)
                reviewer, tests, runner, fixer = self.downstream()
                output, errors = io.StringIO(), io.StringIO()
                with patch("sys.argv", ["main.py", "--requirements", "实现相加", "--fix-code",
                                       "--run-tests", "--max-fix-attempts", "0"]), \
                        patch("main.create_code_generator", return_value=generator), \
                        patch("main.create_reviewer", return_value=reviewer), \
                        patch("main.create_test_generator", return_value=tests), \
                        patch("main.create_code_fixer", return_value=fixer), \
                        patch("main.TestRunner", return_value=runner), \
                        redirect_stdout(output), redirect_stderr(errors):
                    status = main()
                self.assertEqual(status, 0 if recovered else 1)
                self.assertEqual(client.chat.completions.create.call_count, 2 if recovered else 4)
                fixer.fix.assert_not_called()
                if recovered:
                    self.assertEqual(errors.getvalue(), "")
                    self.assertIn("Coding Agent", output.getvalue())
                    reviewer.review.assert_called_once()
                    runner.run.assert_called_once_with(VALID, CASES)
                else:
                    self.assertIn("LLM 代码生成失败", errors.getvalue())
                    self.assertIn("4 次生成调用", errors.getvalue())
                    self.assertNotIn("Coding Agent", output.getvalue())
                    reviewer.review.assert_not_called()
                    tests.generate.assert_not_called()
                    runner.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
