"""Bounded repair loop tests; generation is mocked and regression execution is real."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from agent import CodeReviewAgent
from coding_agent import CodingAgent, create_coding_workflow
from fix_agent import FixAgent
from llm import LLMFixError
from main import main
from models import (
    CodeGenerationResult, ReviewIssue, ReviewReport, TestCase as GeneratedCase,
    TestExecutionResult as ExecutionResult, TestFailure as FailureDetail, TestGenerationResult as GenerationResult,
)
from test_agent import TestAgent as WorkflowTestAgent
from workflow import (
    END,
    RepairDecision,
    WorkflowConfigurationError,
    WorkflowExecutionError,
    WorkflowOutcome,
    WorkflowState,
    WorkflowStatus,
    create_repair_workflow,
)


BAD = "def add(a, b):\n    return a - b\n"
INTERMEDIATE = "def add(a, b):\n    return a + b + 1\n"
GOOD = "def add(a, b):\n    return a + b\n"
CASES = [GeneratedCase(name="test_add", code="self.assertEqual(module.add(2, 3), 5)")]


def review_report(code: str) -> ReviewReport:
    return ReviewReport(summary=code, issues=[] if code == GOOD else [ReviewIssue(
        problem="加法不符合需求", severity="high", reason="结果错误", suggestion="正确相加"
    )])


def make_test_result(code: str) -> ExecutionResult:
    return ExecutionResult(
        status="passed" if code == GOOD else "failed", tests_run=1,
        passed=int(code == GOOD), failed=int(code != GOOD), stdout=code,
        failures=[] if code == GOOD else [FailureDetail(test="test_add", traceback="AssertionError")],
    )


class RepairLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reviewer = Mock()
        self.reviewer.review.side_effect = lambda code, findings, **kwargs: review_report(code)
        self.fixer = Mock()
        self.fixer.fix.return_value = CodeGenerationResult(code=GOOD, summary="修复加法")
        self.generator = Mock()
        self.generator.generate.return_value = GenerationResult(test_cases=CASES, summary="测试相加")
        self.runner = Mock()
        self.runner.run.side_effect = lambda code, cases: make_test_result(code)

    def engine(self, limit: int = 3, *, real_runner: bool = False):
        return create_repair_workflow(
            CodeReviewAgent(self.reviewer), FixAgent(self.fixer),
            WorkflowTestAgent(self.generator, None if real_runner else self.runner), max_fix_attempts=limit,
        )

    def test_multiple_repairs_converge_and_record_each_round(self) -> None:
        self.fixer.fix.side_effect = [
            CodeGenerationResult(code=INTERMEDIATE, summary="第一次修复"),
            CodeGenerationResult(code=GOOD, summary="第二次修复"),
        ]
        run = self.engine().run({"code": BAD, "requirements": "实现相加"})
        state = run.state
        self.assertEqual([item.step for item in run.executions],
                         ["review", "test", "fix", "test", "review", "fix", "test", "review"])
        self.assertEqual(state.get("fix_attempts"), 2)
        self.assertEqual(state.repair_iteration_count, 2)
        self.assertEqual(state.repair_history, state.get("repair_history"))
        self.assertEqual(state.iteration_count, len(run.executions))
        self.assertEqual(len(state.agent_results), len(run.executions))
        self.assertEqual(len(state.stage_history), len(run.executions))
        self.assertEqual(
            [item.repair_round for item in state.agent_results],
            [0, 0, 1, 1, 1, 2, 2, 2],
        )
        self.assertEqual(
            [item.repair_round for item in state.stage_history],
            [0, 0, 1, 1, 1, 2, 2, 2],
        )
        self.assertEqual(
            [
                sum(item.event == "agent.completed" for item in state.logs_for_round(round_index))
                for round_index in range(3)
            ],
            [2, 3, 3],
        )
        self.assertEqual(state.get("repair_stop_reason"), "resolved")
        self.assertTrue(state.get("repair_succeeded"))
        decision = state.get("repair_decision")
        self.assertIsInstance(decision, RepairDecision)
        self.assertEqual((decision.review_passed, decision.test_passed), (True, True))
        self.assertEqual((decision.next_step, decision.reason), (END, "resolved"))
        self.assertEqual(
            [item.event for item in state.execution_logs].count("repair.decision"),
            len(run.executions),
        )
        self.assertEqual(state.get("code"), GOOD)
        report = run.report
        self.assertEqual(report.outcome, WorkflowOutcome.SUCCEEDED)
        self.assertEqual(report.repair_iteration_count, 2)
        self.assertEqual(len(report.repair_history), 2)
        self.assertEqual(len(report.test_results), 3)
        self.assertEqual(report.final_test_result.status, "passed")
        self.assertEqual(
            sum(item.event == "agent.completed" for item in report.logs_for_round(2)),
            3,
        )
        self.assertEqual(report.to_dict()["repair_history"][-1]["status"], "resolved")
        first, second = state.get("repair_history")
        self.assertEqual((first.attempt, first.code_before, first.code_after, first.status),
                         (1, BAD, INTERMEDIATE, "unresolved"))
        self.assertEqual((second.attempt, second.code_before, second.code_after, second.status),
                         (2, INTERMEDIATE, GOOD, "resolved"))
        self.assertEqual(first.test_result_before.status, "failed")
        self.assertEqual(second.test_result_after.status, "passed")
        for record in (first, second):
            self.assertEqual([step.step for step in record.executions], ["review", "fix", "test", "review"])
            self.assertTrue(record.fix_result.diff)
            self.assertTrue(all(step.duration_seconds >= 0 for step in record.executions))
        self.assertLess(first.executions[-1].state_version, second.executions[-1].state_version)
        self.assertEqual(self.fixer.fix.call_args_list[1].args[0], INTERMEDIATE)
        self.assertEqual(self.fixer.fix.call_args_list[1].kwargs["test_result"].stdout, INTERMEDIATE)
        self.generator.generate.assert_called_once()
        self.assertIsNone(state.get("active_repair_round"))
        # Later mutable diagnostics must not rewrite previous round snapshots.
        state.get("report").summary = "changed later"
        self.assertTrue(first.report_after.summary.endswith(INTERMEDIATE))
        self.assertTrue(second.report_after.summary.endswith(GOOD))
        self.assertTrue(report.final_review_result.summary.endswith(GOOD))

    def test_actual_failed_regression_case_passes_after_two_repairs(self) -> None:
        self.fixer.fix.side_effect = [
            CodeGenerationResult(code=INTERMEDIATE, summary="中间结果"),
            CodeGenerationResult(code=GOOD, summary="最终修复"),
        ]
        coding = Mock()
        coding.generate.return_value = CodeGenerationResult(code=BAD, summary="初始代码")
        workflow = create_coding_workflow(
            CodingAgent(coding), CodeReviewAgent(self.reviewer),
            test_agent=WorkflowTestAgent(self.generator), fix_agent=FixAgent(self.fixer), max_fix_attempts=3,
        )
        run = workflow.run({"requirements": "实现相加"})
        self.assertEqual(run.executions[0].step, "coding")
        self.assertEqual(run.state.get("test_result").status, "passed")
        self.assertEqual([record.test_result_after.status for record in run.state.get("repair_history")],
                         ["failed", "passed"])
        self.assertEqual(run.state.get("test_cases"), CASES)
        self.generator.generate.assert_called_once()
        self.assertEqual(self.fixer.fix.call_count, 2)

    def test_limit_stops_unresolved_work_and_preserves_final_diagnostics(self) -> None:
        self.fixer.fix.side_effect = [
            CodeGenerationResult(code=INTERMEDIATE, summary="第一次修复"),
            CodeGenerationResult(code=BAD, summary="第二次修复"),
        ]
        run = self.engine(2).run({"code": BAD})
        self.assertEqual(run.state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(run.state.get("repair_stop_reason"), "max_fix_attempts")
        self.assertFalse(run.state.get("repair_succeeded"))
        self.assertEqual(run.state.get("fix_attempts"), 2)
        self.assertEqual(len(run.state.get("repair_history")), 2)
        self.assertTrue(run.state.get("report").summary.endswith(BAD))
        self.assertEqual(run.state.get("test_result").status, "failed")
        self.assertEqual(run.state.get("repair_decision").reason, "max_fix_attempts")
        self.assertEqual(self.fixer.fix.call_count, 2)

    def test_resolution_on_the_last_allowed_attempt_takes_precedence(self) -> None:
        run = self.engine(1).run({"code": BAD})
        self.assertEqual(run.state.get("repair_stop_reason"), "resolved")
        self.assertTrue(run.state.get("repair_succeeded"))
        self.assertEqual(run.state.get("fix_attempts"), 1)

    def test_zero_budget_runs_baseline_diagnostics_but_never_fixes(self) -> None:
        for code in (BAD, GOOD):
            with self.subTest(code=code):
                run = self.engine(0).run({"code": code})
                self.assertEqual([item.step for item in run.executions], ["review", "test"])
                self.assertEqual(run.state.get("fix_attempts"), 0)
                self.assertEqual(run.state.get("repair_history"), ())
                self.assertEqual(run.state.get("repair_stop_reason"),
                                 "resolved" if code == GOOD else "max_fix_attempts")
        self.fixer.fix.assert_not_called()

    def test_clean_baseline_exits_without_consuming_attempts(self) -> None:
        run = self.engine().run({"code": GOOD})
        self.assertTrue(run.state.get("repair_succeeded"))
        self.assertEqual(run.state.get("fix_attempts"), 0)
        self.assertEqual(run.state.get("repair_history"), ())
        self.fixer.fix.assert_not_called()

    def test_unchanged_fix_stops_early_and_records_no_progress(self) -> None:
        self.fixer.fix.return_value = CodeGenerationResult(code=BAD, summary="未改变代码")
        run = self.engine().run({"code": BAD})
        self.assertEqual(run.state.get("repair_stop_reason"), "no_change")
        self.assertEqual(run.state.get("fix_attempts"), 1)
        record = run.state.get("repair_history")[0]
        self.assertEqual(record.status, "unchanged")
        self.assertEqual(record.code_before, record.code_after)
        self.assertEqual(record.test_result_after.status, "failed")
        self.assertEqual([step.step for step in record.executions], ["review", "fix"])
        self.assertEqual(self.runner.run.call_count, 1)

    def test_passing_tests_do_not_hide_remaining_review_issues(self) -> None:
        self.runner.run.side_effect = lambda code, cases: ExecutionResult(status="passed", tests_run=1, passed=1)
        self.fixer.fix.return_value = CodeGenerationResult(code=INTERMEDIATE, summary="仍有问题")
        run = self.engine(1).run({"code": BAD})
        self.assertEqual(run.state.get("test_result").status, "passed")
        self.assertEqual(run.state.get("repair_stop_reason"), "max_fix_attempts")
        self.assertFalse(run.state.get("repair_succeeded"))

    def test_clean_review_does_not_hide_test_failures(self) -> None:
        self.reviewer.review.side_effect = lambda code, findings, **kwargs: ReviewReport(summary="没有问题", issues=[])
        run = self.engine().run({"code": BAD})
        self.fixer.fix.assert_called_once()
        self.assertEqual(self.fixer.fix.call_args.kwargs["report"].issues, [])
        self.assertEqual(self.fixer.fix.call_args.kwargs["test_result"].status, "failed")
        self.assertTrue(run.state.get("repair_succeeded"))

    def test_engine_reuse_resets_budget_and_seeded_loop_metadata(self) -> None:
        engine = self.engine(1)
        for _ in range(2):
            run = engine.run({"code": BAD, "fix_attempts": 999, "repair_history": "invalid seed",
                              "_repair_tested": True, "test_result": make_test_result(GOOD)})
            self.assertEqual(run.state.get("fix_attempts"), 1)
            self.assertEqual(len(run.state.get("repair_history")), 1)
            self.assertEqual(run.state.get("repair_history")[0].attempt, 1)
            self.assertTrue(run.state.get("repair_succeeded"))
        self.assertEqual(self.runner.run.call_count, 4)

    def test_failed_fix_is_recorded_without_overwriting_code(self) -> None:
        self.fixer.fix.side_effect = LLMFixError("修复服务失败")
        state = WorkflowState({"code": BAD})
        with self.assertRaises(WorkflowExecutionError) as caught:
            self.engine().run(state)
        self.assertIsInstance(caught.exception.__cause__, LLMFixError)
        self.assertEqual(state.status, WorkflowStatus.FAILED)
        self.assertEqual(state.get("code"), BAD)
        self.assertEqual(state.get("fix_attempts"), 1)
        record = state.get("repair_history")[0]
        self.assertEqual(record.status, "error")
        self.assertEqual(record.error, "修复服务失败")
        self.assertEqual(record.executions[-1].agent, "fix")
        self.assertEqual(state.get("repair_stop_reason"), "error")
        self.assertEqual(state.get("repair_decision").reason, "error")
        self.assertTrue(any(
            item.event == "stage.failed" and item.repair_round == 1
            for item in state.execution_logs
        ))
        self.assertTrue(any(
            item.event == "repair.decision" and item.repair_round == 1
            and item.level == "error"
            for item in state.execution_logs
        ))

    def test_verification_error_records_the_partial_round(self) -> None:
        self.runner.run.side_effect = [make_test_result(BAD), RuntimeError("测试执行器不可用")]
        state = WorkflowState({"code": BAD})
        with self.assertRaises(WorkflowExecutionError):
            self.engine().run(state)
        record = state.get("repair_history")[0]
        self.assertEqual(record.code_after, GOOD)
        self.assertEqual(record.status, "error")
        self.assertEqual(record.executions[-1].step, "test")
        self.assertEqual(record.executions[-1].error, "测试执行器不可用")
        self.assertIsNone(record.test_result_after)
        self.assertIsNone(state.get("active_repair_round"))

    def test_later_fix_error_preserves_preceding_round_records(self) -> None:
        self.fixer.fix.side_effect = [
            CodeGenerationResult(code=INTERMEDIATE, summary="仍有问题"), LLMFixError("第二轮服务失败"),
        ]
        state = WorkflowState({"code": BAD})
        with self.assertRaises(WorkflowExecutionError):
            self.engine().run(state)
        first, failed = state.get("repair_history")
        self.assertEqual(first.status, "unresolved")
        self.assertEqual((failed.attempt, failed.code_before, failed.status), (2, INTERMEDIATE, "error"))
        self.assertEqual(state.get("fix_attempts"), 2)
        self.assertEqual(state.get("code"), INTERMEDIATE)

    def test_review_error_after_retest_does_not_claim_resolution(self) -> None:
        self.reviewer.review.side_effect = [review_report(BAD), RuntimeError("复审不可用")]
        state = WorkflowState({"code": BAD})
        with self.assertRaises(WorkflowExecutionError):
            self.engine().run(state)
        record = state.get("repair_history")[0]
        self.assertEqual(record.status, "error")
        self.assertEqual(record.test_result_after.status, "passed")
        self.assertIsNone(record.report_after)
        self.assertEqual(record.executions[-1].step, "review")
        self.assertFalse(state.get("repair_succeeded"))

    def test_agent_configuration_error_also_records_the_attempt(self) -> None:
        self.fixer.fix.side_effect = WorkflowConfigurationError("节点配置错误")
        state = WorkflowState({"code": BAD})
        with self.assertRaises(WorkflowConfigurationError):
            self.engine().run(state)
        self.assertEqual(state.status, WorkflowStatus.FAILED)
        self.assertEqual(state.get("fix_attempts"), 1)
        self.assertEqual(state.get("repair_history")[0].status, "error")
        self.assertEqual(state.get("repair_stop_reason"), "error")

    def test_large_budget_does_not_hit_the_generic_engine_step_limit(self) -> None:
        attempt = 0

        def fix(code, **kwargs):
            nonlocal attempt
            attempt += 1
            return CodeGenerationResult(code=f"def add(a, b):\n    return a + b + {attempt}\n", summary="未收敛")

        self.fixer.fix.side_effect = fix
        run = self.engine(35).run({"code": BAD})
        self.assertGreater(len(run.executions), 100)
        self.assertEqual(run.state.get("fix_attempts"), 35)
        self.assertEqual(run.state.get("repair_stop_reason"), "max_fix_attempts")
        self.assertEqual(len(run.state.get("repair_history")), 35)

    def test_invalid_limits_and_missing_nodes_are_rejected(self) -> None:
        for limit in (-1, True, 1.5, "3"):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.engine(limit)
        coding = Mock()
        with self.assertRaises(ValueError):
            create_coding_workflow(CodingAgent(coding), fix_agent=FixAgent(self.fixer), max_fix_attempts=3)


class RepairCliTests(unittest.TestCase):
    def test_cli_prints_rounds_and_unresolved_review_returns_one(self) -> None:
        coding = Mock()
        coding.generate.return_value = CodeGenerationResult(code=BAD, summary="初始代码")
        reviewer = Mock()
        reviewer.review.side_effect = lambda code, findings, **kwargs: review_report(code)
        generator = Mock()
        generator.generate.return_value = GenerationResult(test_cases=CASES, summary="验证加法")
        fixer = Mock()
        fixer.fix.return_value = CodeGenerationResult(code=INTERMEDIATE, summary="仍不正确")
        runner = Mock()
        runner.run.return_value = ExecutionResult(status="passed", tests_run=1, passed=1)
        output = io.StringIO()
        with patch("sys.argv", ["main.py", "--requirements", "实现相加", "--fix-code", "--run-tests",
                                "--max-fix-attempts", "1"]), \
                patch("main.create_code_generator", return_value=coding), \
                patch("main.create_reviewer", return_value=reviewer), \
                patch("main.create_test_generator", return_value=generator), \
                patch("main.create_code_fixer", return_value=fixer), \
                patch("main.TestRunner", return_value=runner), redirect_stdout(output):
            self.assertEqual(main(), 1)
        self.assertIn("自动修复次数：1", output.getvalue())
        self.assertIn("第 1 轮：unresolved", output.getvalue())
        self.assertIn("max_fix_attempts", output.getvalue())

    def test_zero_attempts_does_not_require_a_fix_result_to_print(self) -> None:
        coding = Mock()
        coding.generate.return_value = CodeGenerationResult(code=GOOD, summary="正确代码")
        reviewer = Mock()
        reviewer.review.return_value = ReviewReport(summary="没有问题", issues=[])
        generator = Mock()
        generator.generate.return_value = GenerationResult(test_cases=CASES, summary="验证加法")
        runner = Mock()
        runner.run.return_value = make_test_result(GOOD)
        with patch("sys.argv", ["main.py", "--requirements", "实现相加", "--fix-code", "--run-tests",
                                "--max-fix-attempts", "0"]), \
                patch("main.create_code_generator", return_value=coding), \
                patch("main.create_reviewer", return_value=reviewer), \
                patch("main.create_test_generator", return_value=generator), \
                patch("main.create_code_fixer") as fixer, \
                patch("main.TestRunner", return_value=runner), redirect_stdout(io.StringIO()):
            self.assertEqual(main(), 0)
            fixer.return_value.fix.assert_not_called()

    def test_invalid_loop_options_fail_before_loading_providers(self) -> None:
        arguments = [
            ["--max-fix-attempts", "3"],
            ["--requirements", "需求", "--fix-code", "--max-fix-attempts", "3"],
            ["--requirements", "需求", "--run-tests", "--max-fix-attempts", "3"],
            ["--requirements", "需求", "--fix-code", "--run-tests", "--max-fix-attempts", "-1"],
        ]
        for options in arguments:
            with self.subTest(options=options), patch("sys.argv", ["main.py", *options]), \
                    patch("main.create_code_generator") as factory, redirect_stderr(io.StringIO()):
                self.assertEqual(main(), 2)
                factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
