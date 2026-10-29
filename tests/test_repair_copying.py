"""Repair records must support the same isolated values as ordinary workflows."""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
import unittest
from unittest.mock import Mock

from agent import ReviewAgent
from fix_agent import FixAgent
from models import (
    CodeGenerationResult, ReviewIssue, ReviewReport,
    TestExecutionResult as ExecutionResult, TestGenerationResult as GenerationResult,
)
from test_agent import TestAgent as WorkflowTestAgent
from workflow import (
    AgentResult, BaseAgent, WorkflowExecutionError, WorkflowOutcome,
    WorkflowState, WorkflowStatus, create_repair_workflow,
)


BAD = "def add(a, b):\n    return a - b\n"
GOOD = "def add(a, b):\n    return a + b\n"
BLOCKED = ReviewReport(summary="修复加法", issues=[ReviewIssue(
    problem="使用了减法", severity="high", reason="不符合加法需求", suggestion="使用加法",
)])


def proxy_payload(value):
    shared = {"items": ["original"]}
    return MappingProxyType({"result": value, "left": shared, "right": shared})


@dataclass
class Envelope:
    payload: object


class OutputAgent(BaseAgent):
    def __init__(self, inner, factory):
        self.name = inner.name
        self.inner = inner
        self.factory = factory
        self.retained = []

    def run(self, state):
        result = self.inner.execute(state)
        output = self.factory(result.output)
        self.retained.append(output)
        return AgentResult(updates=result.updates, output=output)


class CallbackAgent(BaseAgent):
    def __init__(self, name, callback):
        self.name = name
        self.callback = callback

    def run(self, state):
        return self.callback(state)


class RepairCopyingTests(unittest.TestCase):
    def engine(self, factory=proxy_payload, stages=("review", "fix", "test"), fixed_code=GOOD):
        reviewer = Mock()
        reviewer.review.side_effect = lambda code, findings, **kwargs: (
            BLOCKED if code == BAD else ReviewReport(summary="审查通过", issues=[])
        )
        fixer = Mock()
        fixer.fix.return_value = CodeGenerationResult(code=fixed_code, summary="修复加法")
        generator = Mock()
        generator.generate.return_value = GenerationResult(
            test_cases=[{"name": "test_add", "code": "self.assertEqual(module.add(2, 3), 5)"}],
            summary="验证原加法断言",
        )
        runner = Mock()
        runner.run.side_effect = lambda code, cases: ExecutionResult(
            status="passed" if code == GOOD else "failed", tests_run=1,
            passed=int(code == GOOD), failed=int(code != GOOD),
        )
        agents = {"review": ReviewAgent(reviewer), "fix": FixAgent(fixer),
                  "test": WorkflowTestAgent(generator, runner)}
        wrappers = {}
        for stage in stages:
            wrappers[stage] = agents[stage] = OutputAgent(agents[stage], factory)
        return create_repair_workflow(agents["review"], agents["fix"], agents["test"]), wrappers

    def test_readonly_outputs_work_in_every_repair_stage(self):
        for stages in (("review",), ("fix",), ("test",), ("review", "fix", "test")):
            with self.subTest(stages=stages):
                engine, _ = self.engine(stages=stages)
                run = engine.run({"code": BAD})
                self.assertTrue(run.report.succeeded)
                self.assertEqual(run.state.get("code"), GOOD)
                self.assertEqual(run.state.get("fix_attempts"), 1)
                self.assertEqual([item.step for item in run.executions],
                                 ["review", "test", "fix", "test", "review"])
                for execution in run.executions:
                    if execution.step in stages:
                        self.assertIsInstance(execution.output, MappingProxyType)
                if "fix" in stages:
                    record = run.state.repair_history[0]
                    self.assertIsInstance(record.fix_result, MappingProxyType)
                    self.assertTrue(record.fix_result["result"].changed)
                json.dumps(run.report.to_dict())

    def test_nested_and_cyclic_proxies_work_in_records_and_reports(self):
        def cyclic(value):
            backing = {"result": value}
            proxy = MappingProxyType(backing)
            backing["self"] = proxy
            return proxy

        factories = (
            lambda value: {"payload": proxy_payload(value)},
            lambda value: Envelope(proxy_payload(value)),
            cyclic,
        )
        for factory in factories:
            with self.subTest(factory=factory):
                engine, _ = self.engine(factory=factory)
                run = engine.run({"code": BAD})
                report = run.report
                self.assertTrue(report.succeeded)
                self.assertEqual(report.repair_history[0].status, "resolved")
                encoded = json.dumps(report.to_dict())
                if factory is cyclic:
                    output = report.repair_history[0].fix_result
                    self.assertIs(output["self"], output)
                    self.assertIn("<circular reference:", encoded)

    def test_clean_baseline_proxy_outputs_do_not_consume_repair_budget(self):
        engine, wrappers = self.engine()
        run = engine.run({"code": GOOD})
        self.assertTrue(run.report.succeeded)
        self.assertEqual(run.state.get("fix_attempts"), 0)
        self.assertEqual(run.state.repair_history, ())
        self.assertEqual([item.step for item in run.executions], ["review", "test"])
        self.assertEqual(wrappers["fix"].retained, [])

    def test_retained_outputs_cannot_mutate_execution_round_or_report(self):
        engine, wrappers = self.engine()
        run = engine.run({"code": BAD})
        report = run.report
        record = run.state.repair_history[0]
        snapshots = (
            *(item.output for item in run.executions),
            *(item.output for item in run.state.agent_results),
            *(item.output for item in record.executions),
            record.fix_result, report.repair_history[0].fix_result,
            run.state.get("_repair_last_review").output,
        )
        for wrapper in wrappers.values():
            for output in wrapper.retained:
                output["left"]["items"].append("provider mutation")
        for snapshot in snapshots:
            self.assertIsInstance(snapshot, MappingProxyType)
            self.assertIs(snapshot["left"], snapshot["right"])
            self.assertEqual(snapshot["left"]["items"], ["original"])
            with self.assertRaises(TypeError):
                snapshot["new"] = True
        record.fix_result["left"]["items"].append("state mutation")
        self.assertEqual(report.repair_history[0].fix_result["left"]["items"], ["original"])
        self.assertEqual(run.executions[2].output["left"]["items"], ["original"])

    def test_unchanged_proxy_fix_still_stops_as_unresolved(self):
        engine, _ = self.engine(fixed_code=BAD)
        run = engine.run({"code": BAD})
        self.assertEqual(run.report.outcome, WorkflowOutcome.UNRESOLVED)
        self.assertEqual(run.state.get("repair_stop_reason"), "no_change")
        self.assertEqual(run.state.get("fix_attempts"), 1)
        self.assertIsInstance(run.state.repair_history[0].fix_result, MappingProxyType)

    def test_failure_preserves_original_error_and_proxy_diagnostics(self):
        class DiagnosticReview(ReviewReport):
            details: object

        class DiagnosticResult(ExecutionResult):
            details: object

        review = DiagnosticReview(**BLOCKED.model_dump(), details=proxy_payload(None))
        test = DiagnosticResult(status="failed", tests_run=1, failed=1, details=proxy_payload(None))

        def fail(state):
            raise RuntimeError("original fix failure")

        engine = create_repair_workflow(
            CallbackAgent("review", lambda state: AgentResult(updates={"report": review}, output=review)),
            CallbackAgent("fix", fail),
            CallbackAgent("test", lambda state: AgentResult(updates={"test_result": test}, output=test)),
        )
        with self.assertRaises(WorkflowExecutionError) as caught:
            engine.run({"code": BAD})
        state = caught.exception.state
        self.assertIsInstance(caught.exception.__cause__, RuntimeError)
        self.assertEqual(str(caught.exception.__cause__), "original fix failure")
        self.assertEqual(state.status, WorkflowStatus.FAILED)
        self.assertEqual(state.get("code"), BAD)
        self.assertEqual(state.get("repair_stop_reason"), "error")
        record = state.repair_history[0]
        self.assertEqual((record.attempt, record.status, record.error), (1, "error", "original fix failure"))
        self.assertIsInstance(record.report_before.details, MappingProxyType)
        self.assertIsInstance(record.test_result_before.details, MappingProxyType)

    def test_unsafe_fix_output_is_rejected_before_code_commit(self):
        engine, _ = self.engine(factory=lambda value: iter([value]), stages=("fix",))
        state = WorkflowState({"code": BAD})
        with self.assertRaises(WorkflowExecutionError) as caught:
            engine.run(state)
        self.assertIsInstance(caught.exception.__cause__, TypeError)
        self.assertEqual(state.get("code"), BAD)
        self.assertIsNone(state.get("fix_result"))
        self.assertEqual(state.get("repair_stop_reason"), "error")
        self.assertEqual(state.repair_history[0].status, "error")

    def test_verification_failure_retains_proxy_round_and_original_error(self):
        engine, wrappers = self.engine()
        wrappers["test"].inner._runner.run.side_effect = [
            ExecutionResult(status="failed", tests_run=1, failed=1),
            RuntimeError("retest unavailable"),
        ]
        with self.assertRaises(WorkflowExecutionError) as caught:
            engine.run({"code": BAD})
        state = caught.exception.state
        self.assertEqual(str(caught.exception.__cause__), "retest unavailable")
        self.assertEqual(state.get("code"), GOOD)
        record = state.repair_history[0]
        self.assertEqual((record.status, record.error), ("error", "retest unavailable"))
        self.assertIsInstance(record.fix_result, MappingProxyType)
        self.assertEqual(record.executions[-1].step, "test")
        self.assertIsNone(record.test_result_after)
        self.assertEqual(caught.exception.state.status, WorkflowStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
