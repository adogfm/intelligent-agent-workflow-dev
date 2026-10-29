"""Offline tests with real subprocess execution of small known programs."""

from __future__ import annotations

import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agent import CodeReviewAgent
from coding_agent import CodingAgent, create_coding_workflow
from llm import (
    LLMTestGenerationError,
    MockCodeGenerator,
    MockReviewer,
    MockTestGenerator,
    OpenAITestGenerator,
    create_test_generator,
    parse_test_generation,
)
from main import main
from models import CodeGenerationResult, TestCase as GeneratedCase, TestGenerationResult as GenerationResult
from test_agent import TestAgent as WorkflowTestAgent, TestRunner as SubprocessTestRunner, create_test_workflow
from workflow import Agent, WorkflowExecutionError, WorkflowState, WorkflowStatus


ADD_CODE = "def add(a, b):\n    return a + b\n"


def case(name: str = "test_add", code: str = "self.assertEqual(module.add(2, 3), 5)") -> GeneratedCase:
    return GeneratedCase(name=name, code=code)


class TestRunnerTests(unittest.TestCase):
    def test_real_execution_reports_passes_failures_errors_and_skips(self) -> None:
        cases = [
            case(),
            case("test_wrong", "self.assertEqual(module.add(1, 1), 3)"),
            case("test_error", "module.missing()"),
            case("test_skip", "self.skipTest('not applicable')"),
        ]
        result = SubprocessTestRunner().run(ADD_CODE, cases)

        self.assertEqual(result.status, "failed")
        self.assertEqual((result.tests_run, result.passed, result.failed, result.errors, result.skipped),
                         (4, 1, 1, 1, 1))
        self.assertIn("AssertionError", result.failures[0].traceback)
        self.assertIn("AttributeError", result.error_details[0].traceback)
        self.assertEqual(result.return_code, 1)
        self.assertGreater(result.duration_seconds, 0)

    def test_success_captures_output_and_returns_zero(self) -> None:
        result = SubprocessTestRunner().run("print('module output')\n" + ADD_CODE, [case()])
        self.assertEqual(result.status, "passed")
        self.assertEqual(result.passed, 1)
        self.assertEqual(result.return_code, 0)
        self.assertIn("module output", result.stdout)

    def test_import_errors_and_syntax_errors_are_execution_results(self) -> None:
        for code in ("raise RuntimeError('import failed')", "def broken(:"):
            with self.subTest(code=code):
                result = SubprocessTestRunner().run(code, [case()])
                self.assertEqual(result.status, "error")
                self.assertEqual(result.tests_run, 0)
                self.assertEqual(result.errors, 1)
                self.assertTrue(result.error_details)

    def test_all_skipped_tests_do_not_report_success(self) -> None:
        result = SubprocessTestRunner().run(ADD_CODE, [case(code="self.skipTest('skip')")])
        self.assertEqual(result.status, "error")
        self.assertEqual(result.skipped, 1)
        self.assertEqual(result.passed, 0)

    def test_subtest_failures_count_one_failed_case(self) -> None:
        result = SubprocessTestRunner().run(ADD_CODE, [case(code=(
            "for value in (1, 2):\n"
            "    with self.subTest(value=value):\n"
            "        self.assertEqual(module.add(value, 0), 0)"
        ))])
        self.assertEqual(result.tests_run, 1)
        self.assertEqual(result.failed, 1)
        self.assertEqual(len(result.failures), 2)

    def test_timeout_terminates_hanging_code(self) -> None:
        result = SubprocessTestRunner(timeout_seconds=0.5).run("while True:\n    pass\n", [case()])
        self.assertEqual(result.status, "timeout")
        self.assertIn("已终止", result.message)
        self.assertIsNotNone(result.return_code)

    def test_missing_result_is_not_treated_as_a_pass(self) -> None:
        result = SubprocessTestRunner().run("import os\nos._exit(0)", [case()])
        self.assertEqual(result.status, "error")
        self.assertEqual(result.return_code, 0)

    def test_output_is_truncated_and_api_keys_are_not_inherited(self) -> None:
        code = "import os\nsecret_present = 'DASHSCOPE_API_KEY' in os.environ\nprint('x' * 1024)\n"
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "test-key"}):
            result = SubprocessTestRunner(max_output_bytes=128).run(
                code, [case(code="self.assertFalse(module.secret_present)")]
            )
        self.assertEqual(result.status, "passed")
        self.assertIn("输出已截断", result.stdout)
        self.assertLess(len(result.stdout), 160)

    def test_invalid_timeout_and_empty_test_cases_are_rejected(self) -> None:
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                SubprocessTestRunner(timeout_seconds=timeout)
        with self.assertRaises(ValueError):
            SubprocessTestRunner().run(ADD_CODE, [])
        with self.assertRaises(ValueError):
            SubprocessTestRunner().run(ADD_CODE, [GeneratedCase.model_construct(name="not_valid", code="pass")])


class TestAgentWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = Mock()
        self.generator.generate.return_value = GenerationResult(
            test_cases=[case()], summary="验证加法"
        )
        self.agent = WorkflowTestAgent(self.generator)

    def test_coding_review_test_workflow_commits_cases_and_result(self) -> None:
        coding_provider = Mock()
        coding_provider.generate.return_value = CodeGenerationResult(code=ADD_CODE, summary="实现相加")
        workflow = create_coding_workflow(
            CodingAgent(coding_provider), CodeReviewAgent(MockReviewer()), test_agent=self.agent
        )
        state = WorkflowState({"requirements": "实现相加", "context": "只使用标准库"})

        run = workflow.run(state)

        self.assertIsInstance(self.agent, Agent)
        self.assertEqual([execution.step for execution in run.executions], ["coding", "review", "test"])
        self.generator.generate.assert_called_once_with(
            ADD_CODE, requirements="实现相加", context="只使用标准库"
        )
        self.assertEqual(state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(state.get("test_cases")[0].name, "test_add")
        self.assertEqual(state.get("test_result").status, "passed")
        self.assertEqual(run.output, state.get("test_result"))
        self.assertEqual(state.version, 3)
        self.assertEqual([change.source for change in state.history], ["coding", "review", "test"])
        self.assertIsNotNone(state.get("generation_result"))
        self.assertIsNotNone(state.get("report"))

    def test_coding_test_workflow_without_review(self) -> None:
        coding_provider = Mock()
        coding_provider.generate.return_value = CodeGenerationResult(code=ADD_CODE, summary="实现相加")
        run = create_coding_workflow(CodingAgent(coding_provider), test_agent=self.agent).run(
            {"requirements": "实现相加"}
        )
        self.assertEqual([item.step for item in run.executions], ["coding", "test"])
        self.assertEqual(run.output.status, "passed")

    def test_failed_assertions_are_committed_as_outcomes(self) -> None:
        self.generator.generate.return_value = GenerationResult(
            test_cases=[case(code="self.assertEqual(module.add(2, 3), 99)")], summary="错误期望"
        )
        run = create_test_workflow(self.agent).run({"code": ADD_CODE})
        self.assertEqual(run.state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(run.state.get("test_result").status, "failed")
        self.assertEqual(len(run.state.get("test_cases")), 1)

    def test_timeout_and_cases_are_committed(self) -> None:
        run = create_test_workflow(WorkflowTestAgent(self.generator, SubprocessTestRunner(timeout_seconds=0.5))).run(
            {"code": "while True:\n    pass"}
        )
        self.assertEqual(run.state.get("test_result").status, "timeout")
        self.assertEqual(len(run.state.get("test_cases")), 1)

    def test_generation_failure_does_not_overwrite_existing_results(self) -> None:
        self.generator.generate.side_effect = LLMTestGenerationError("生成失败")
        runner = Mock()
        state = WorkflowState({"code": ADD_CODE, "test_cases": "old cases", "test_result": "old result"})
        with self.assertRaises(WorkflowExecutionError):
            create_test_workflow(WorkflowTestAgent(self.generator, runner)).run(state)
        self.assertEqual(state.status, WorkflowStatus.FAILED)
        self.assertEqual(state.get("test_cases"), "old cases")
        self.assertEqual(state.get("test_result"), "old result")
        self.assertEqual(state.version, 0)
        runner.run.assert_not_called()

    def test_invalid_input_and_provider_output_do_not_execute(self) -> None:
        runner = Mock()
        for values in ({}, {"code": " "}, {"code": 123}, {"code": ADD_CODE, "context": None}):
            with self.subTest(values=values), self.assertRaises(WorkflowExecutionError):
                create_test_workflow(WorkflowTestAgent(self.generator, runner)).run(values)
        self.generator.generate.assert_not_called()
        for output in ("invalid", GenerationResult.model_construct(test_cases=[], summary="empty")):
            self.generator.generate.return_value = output
            with self.subTest(output=output), self.assertRaises(WorkflowExecutionError):
                create_test_workflow(WorkflowTestAgent(self.generator, runner)).run({"code": ADD_CODE})
        runner.run.assert_not_called()


class TestGeneratorTests(unittest.TestCase):
    def test_json_and_fences_preserve_executable_cases(self) -> None:
        payload = json.dumps({"test_cases": [case().model_dump()], "summary": "验证相加"})
        for content in (payload, f"```json\n{payload}\n```"):
            with self.subTest(content=content):
                self.assertEqual(parse_test_generation(content).test_cases[0].name, "test_add")

    def test_invalid_outputs_are_rejected(self) -> None:
        payloads = [None, "", "not-json", '{}',
                    '{"test_cases": [], "summary": "empty"}',
                    '{"test_cases": [{"name": "not_a_test", "code": "pass"}], "summary": "bad"}',
                    '{"test_cases": [{"name": "test_bad", "code": "def broken(:"}], "summary": "bad"}']
        payloads.append(json.dumps({"test_cases": [case().model_dump()] * 2, "summary": "duplicates"}))
        for content in payloads:
            with self.subTest(content=content), self.assertRaises(LLMTestGenerationError):
                parse_test_generation(content)

    def test_mock_and_real_provider_selection(self) -> None:
        with patch("llm.load_dotenv"), patch.dict(os.environ, {}, clear=True):
            self.assertIsInstance(create_test_generator(), MockTestGenerator)
        for key in ("DASHSCOPE_API_KEY", "OPENAI_API_KEY"):
            with self.subTest(key=key), patch("llm.load_dotenv"), patch.dict(
                os.environ, {key: "test-key"}, clear=True
            ), patch("llm.OpenAITestGenerator") as generator:
                self.assertIs(create_test_generator(), generator.return_value)
                generator.assert_called_once_with(api_key="test-key")

    def test_mock_generates_real_smoke_test_with_clear_limitations(self) -> None:
        run = create_test_workflow(WorkflowTestAgent(MockTestGenerator())).run({"code": ADD_CODE})
        self.assertEqual(run.output.status, "passed")
        self.assertIn("不验证", run.state.get("test_generation_result").summary)

    def test_real_provider_receives_source_requirements_and_context(self) -> None:
        client = Mock()
        payload = json.dumps({"test_cases": [case().model_dump()], "summary": "验证相加"})
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=payload, refusal=None)
        )])
        with patch("llm.load_dotenv"):
            generator = OpenAITestGenerator(api_key="test-key", model="qwen-test", client=client)
        generated = generator.generate(ADD_CODE, requirements="实现相加", context="标准库")
        self.assertEqual(generated.test_cases[0].name, "test_add")
        request = client.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "qwen-test")
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertEqual(json.loads(request["messages"][1]["content"]), {
            "code": ADD_CODE, "requirements": "实现相加", "context": "标准库",
        })

    def test_empty_truncated_filtered_and_refused_responses_fail(self) -> None:
        client = Mock()
        with patch("llm.load_dotenv"):
            generator = OpenAITestGenerator(api_key="test-key", client=client)
        for choices in ([],
                        [SimpleNamespace(finish_reason="length")],
                        [SimpleNamespace(finish_reason="content_filter")],
                        [SimpleNamespace(finish_reason="stop", message=SimpleNamespace(refusal="拒绝"))]):
            with self.subTest(choices=choices), self.assertRaises(LLMTestGenerationError):
                client.chat.completions.create.return_value = SimpleNamespace(choices=choices)
                generator.generate(ADD_CODE)


class TestCliTests(unittest.TestCase):
    def test_cli_passing_tests_do_not_hide_mock_review_blocker(self) -> None:
        output = io.StringIO()
        with patch("sys.argv", ["main.py", "--requirements", "实现相加", "--run-tests"]), patch(
            "main.create_code_generator", return_value=MockCodeGenerator()
        ), patch("main.create_reviewer", return_value=MockReviewer()), patch(
            "main.create_test_generator", return_value=MockTestGenerator()
        ), redirect_stdout(output):
            self.assertEqual(main(), 1)
        self.assertIn("Test Agent 测试结果", output.getvalue())
        self.assertIn("状态：passed", output.getvalue())
        self.assertIn("示例问题", output.getvalue())

    def test_cli_failed_test_returns_one(self) -> None:
        generator = Mock()
        generator.generate.return_value = GenerationResult(
            test_cases=[case(code="self.fail('failed assertion')")], summary="失败用例"
        )
        with patch("sys.argv", ["main.py", "--requirements", "实现相加", "--run-tests"]), patch(
            "main.create_code_generator", return_value=MockCodeGenerator()
        ), patch("main.create_reviewer", return_value=MockReviewer()), patch(
            "main.create_test_generator", return_value=generator
        ), redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 1)

    def test_cli_invalid_options_fail_before_provider_loading(self) -> None:
        for arguments in (["--run-tests"], ["--requirements", "实现相加", "--run-tests", "--test-timeout", "0"]):
            with self.subTest(arguments=arguments), patch("sys.argv", ["main.py", *arguments]), \
                    patch("main.create_test_generator") as factory, redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 2)
                factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
