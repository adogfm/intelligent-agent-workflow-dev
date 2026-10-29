"""Task-scoped review, deterministic scoring, and advisory-only convergence."""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from agent import CodeReviewAgent
from fix_agent import FixAgent
from llm import OpenAICodeFixer, OpenAIReviewer
from models import (
    ReviewContext, ReviewIssue, ReviewReport, TestCase as GeneratedCase, TestExecutionResult as ExecutionResult,
    TestFailure as FailureDetail, TestGenerationResult as GenerationResult,
)
from planner import DEFAULT_REVIEW_TASK
from test_agent import TestAgent as WorkflowTestAgent
from workflow import WorkflowState, create_repair_workflow


CODE = "def add(a, b):\n    return a + b\n"
CASES = [GeneratedCase(name="test_add", code="self.assertEqual(module.add(2, 3), 5)")]
PASSED = ExecutionResult(status="passed", tests_run=1, passed=1)
FAILED = ExecutionResult(status="failed", tests_run=1, failed=1, failures=[FailureDetail(
    test="test_add", test_name="test_add", traceback="AssertionError", test_code=CASES[0].code,
)])


def issue(category="security_standard", *, scope="in_scope", requirement_ref="", severity="high"):
    return ReviewIssue(
        problem="建议增加生产级密码哈希/审计规范", severity=severity,
        reason="通用规范建议", suggestion="改用 bcrypt 并添加审计", category=category,
        scope=scope, evidence="示例代码不包含存储或认证功能", requirement_ref=requirement_ref,
    )


class ReviewPolicyTests(unittest.TestCase):
    def agent(self, report):
        reviewer = Mock()
        reviewer.review.return_value = report
        return CodeReviewAgent(reviewer), reviewer

    def test_unrequested_security_standards_become_nonblocking_recommendations(self) -> None:
        original = ReviewReport(summary="功能符合需求，存在通用建议", issues=[issue()])
        agent, _ = self.agent(original)
        run = agent.review_detailed(CODE, review_context=ReviewContext(
            requirements="实现 add，返回两个数的和", test_result=PASSED, test_cases=CASES,
        ))
        self.assertEqual(run.report.issues, [])
        self.assertEqual(run.report.recommendations, [original.issues[0]])
        self.assertEqual(run.report.score, 100)
        self.assertEqual(run.report.test_status, "passed")
        self.assertIn("不阻断 Workflow", run.report.to_text())

    def test_explicit_security_requirement_is_still_blocking(self) -> None:
        requested = "密码必须使用 bcrypt 哈希存储"
        agent, _ = self.agent(ReviewReport(summary="未满足要求", issues=[issue(requirement_ref=requested)]))
        report = agent.review_detailed(CODE, review_context=ReviewContext(
            requirements=requested, test_result=PASSED,
        )).report
        self.assertEqual(len(report.issues), 1)
        self.assertEqual(report.score, 60)
        self.assertEqual(report.recommendations, [])

    def test_generic_or_unmatched_reference_cannot_expand_scope(self) -> None:
        agent, _ = self.agent(ReviewReport(summary="建议", issues=[issue(requirement_ref="生产安全标准")]))
        report = agent.review_detailed(CODE, review_context=ReviewContext(
            requirements="实现求和函数", test_result=PASSED,
        )).report
        self.assertEqual(report.issues, [])
        self.assertEqual(report.score, 100)

    def test_generic_password_reference_and_excluded_standards_are_not_requirements(self) -> None:
        for requirements, reference in (
            ("实现 register_user 校验密码", "校验密码"),
            ("实现 register_user，不要求 bcrypt 哈希存储", "bcrypt 哈希存储"),
            ("implement registration; bcrypt is not required", "bcrypt"),
        ):
            with self.subTest(requirements=requirements):
                agent, _ = self.agent(ReviewReport(summary="通用标准", issues=[issue(requirement_ref=reference)]))
                report = agent.review_detailed(CODE, review_context=ReviewContext(
                    requirements=requirements, test_result=PASSED,
                )).report
                self.assertEqual(report.issues, [])
                self.assertEqual(report.score, 100)

    def test_actual_test_failure_blocks_even_if_reviewer_claims_no_problems(self) -> None:
        agent, _ = self.agent(ReviewReport(summary="模型认为无问题", issues=[]))
        report = agent.review_detailed(CODE, review_context=ReviewContext(
            requirements="实现相加", test_result=FAILED, test_cases=CASES,
        )).report
        self.assertEqual(len(report.issues), 1)
        self.assertEqual(report.issues[0].category, "test_failure")
        self.assertIn("test_add", report.issues[0].evidence)
        self.assertEqual(report.score, 60)
        self.assertEqual(report.test_status, "failed")

    def test_unrun_and_successful_tests_do_not_become_invented_failures(self) -> None:
        for result in (None, PASSED):
            with self.subTest(result=result):
                agent, _ = self.agent(ReviewReport(summary="希望增加覆盖", issues=[issue("test_failure")]))
                report = agent.review_detailed(CODE, review_context=ReviewContext(test_result=result)).report
                self.assertEqual(report.issues, [])
                self.assertEqual(len(report.recommendations), 1)
                self.assertEqual(report.score, 100)
                self.assertEqual(report.test_status, "not_run" if result is None else "passed")

    def test_requirements_and_real_defects_remain_blocking_when_tests_pass(self) -> None:
        agent, _ = self.agent(ReviewReport(summary="遗漏需求和缺陷", issues=[
            issue("requirement", severity="low"), issue("code_defect", severity="high"),
        ]))
        report = agent.review_detailed(CODE, review_context=ReviewContext(test_result=PASSED)).report
        self.assertEqual(len(report.issues), 2)
        self.assertEqual(report.score, 35)
        self.assertEqual(report.recommendations, [])

    def test_static_tool_findings_cannot_be_downgraded_by_optional_advice(self) -> None:
        agent, _ = self.agent(ReviewReport(summary="其他建议不阻断", issues=[issue(scope="out_of_scope")]))
        run = agent.review_detailed("value = 1 / 0", review_context=ReviewContext(test_result=PASSED))
        self.assertEqual(len(run.tool_findings), 1)
        self.assertEqual(len(run.report.issues), 1)
        self.assertIn("常量零", run.report.issues[0].problem)
        self.assertEqual(run.report.score, 75)
        self.assertEqual(len(run.report.recommendations), 1)

    def test_improvements_do_not_reduce_score_and_severity_still_controls_real_defects(self) -> None:
        agent, _ = self.agent(ReviewReport(summary="仅可选建议", issues=[issue("improvement", severity="critical")]))
        self.assertEqual(agent.review(CODE).score, 100)
        for severity, score in (("critical", 60), ("high", 75), ("medium", 85), ("low", 95)):
            with self.subTest(severity=severity):
                agent, _ = self.agent(ReviewReport(summary="缺陷", issues=[issue("code_defect", severity=severity)]))
                self.assertEqual(agent.review(CODE).score, score)

    def test_contextual_reviewer_receives_requirements_and_current_test_evidence(self) -> None:
        class RecordingReviewer:
            is_mock = True

            def review(self, *args, **kwargs):
                raise AssertionError("contextual review should be used")

            def review_with_context(self, code, findings, *, task, plan, review_context):
                self.context = review_context
                return ReviewReport(summary="根据实际任务审查", issues=[])

        reviewer = RecordingReviewer()
        state = WorkflowState({"code": CODE, "requirements": "实现 add", "context": "仅演示，不存储密码",
                               "test_cases": CASES, "test_result": FAILED})
        result = CodeReviewAgent(reviewer).execute(state.snapshot())
        self.assertEqual(reviewer.context.requirements, "实现 add")
        self.assertEqual(reviewer.context.context, "仅演示，不存储密码")
        self.assertEqual(reviewer.context.test_cases, CASES)
        self.assertEqual(reviewer.context.test_result, FAILED)
        self.assertEqual(result.output.test_status, "failed")

    def test_legacy_reviewer_signature_remains_supported(self) -> None:
        class LegacyReviewer:
            is_mock = True

            def review(self, code, findings, *, task, plan):
                return ReviewReport(summary="旧接口", issues=[])

        result = CodeReviewAgent(LegacyReviewer()).execute(WorkflowState({
            "code": CODE, "requirements": "实现 add", "test_result": PASSED,
        }).snapshot())
        self.assertEqual(result.output.score, 100)
        self.assertEqual(result.output.issues, [])


class ScopedReviewerWorkflowTests(unittest.TestCase):
    def remote(self, payload):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
            finish_reason="stop", message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False), refusal=None),
        )])
        with patch("llm.load_dotenv"):
            reviewer = OpenAIReviewer(api_key="test-key", model="qwen-test", client=client)
        return reviewer, client

    def test_remote_review_input_includes_original_requirement_and_failure_details(self) -> None:
        reviewer, client = self.remote({"summary": "根据上下文审查", "issues": []})
        CodeReviewAgent(reviewer).execute(WorkflowState({
            "code": CODE, "requirements": "add 按原断言相加", "context": "无需部署安全规范",
            "test_cases": CASES, "test_result": FAILED,
        }).snapshot())
        request = client.chat.completions.create.call_args.kwargs
        user_input = request["messages"][1]["content"]
        context = json.loads(user_input.split("<workflow_context>\n", 1)[1].split("\n</workflow_context>", 1)[0])
        self.assertEqual(context["requirements"], "add 按原断言相加")
        self.assertEqual(context["test_result"]["failures"][0]["traceback"], "AssertionError")
        self.assertEqual(context["test_cases"], [case.model_dump() for case in CASES])
        self.assertIn("任务未要求的生产级安全标准", request["messages"][0]["content"])

    def test_advisory_only_remote_review_does_not_keep_fix_loop_running(self) -> None:
        reviewer, _ = self.remote({"summary": "功能符合需求", "issues": [issue().model_dump()]})
        generator = Mock()
        generator.generate.return_value = GenerationResult(test_cases=CASES, summary="验证相加")
        runner = Mock()
        runner.run.return_value = PASSED
        fixer = Mock()
        workflow = create_repair_workflow(
            CodeReviewAgent(reviewer), FixAgent(fixer), WorkflowTestAgent(generator, runner), max_fix_attempts=3,
        )
        run = workflow.run({"code": CODE, "requirements": "实现 add，只需返回两个数的和"})
        self.assertEqual([item.step for item in run.executions], ["review", "test"])
        self.assertEqual(run.state.get("repair_stop_reason"), "resolved")
        self.assertTrue(run.state.get("repair_succeeded"))
        self.assertEqual(run.state.get("fix_attempts"), 0)
        self.assertEqual(run.state.get("report").score, 100)
        self.assertEqual(len(run.state.get("report").recommendations), 1)
        fixer.fix.assert_not_called()

    def test_high_score_does_not_bypass_an_actual_blocking_defect(self) -> None:
        reviewer, _ = self.remote({"summary": "低严重度但明确的缺陷", "issues": [issue("code_defect", severity="low").model_dump()]})
        generator = Mock()
        generator.generate.return_value = GenerationResult(test_cases=CASES, summary="验证相加")
        runner = Mock()
        runner.run.return_value = PASSED
        workflow = create_repair_workflow(
            CodeReviewAgent(reviewer), FixAgent(Mock()), WorkflowTestAgent(generator, runner), max_fix_attempts=0,
        )
        run = workflow.run({"code": CODE})
        self.assertEqual(run.state.get("report").score, 95)
        self.assertFalse(run.state.get("repair_succeeded"))
        self.assertEqual(run.state.get("repair_stop_reason"), "max_fix_attempts")


class ErrorMessagePolicyTests(unittest.TestCase):
    """Capture real request construction with injected offline SDK clients."""

    def capture_requests(self, requirements, case, result=None):
        code = "def register_user(password):\n    if not password:\n        raise ValueError('新提示！')\n"
        review_client, fix_client = Mock(), Mock()
        # Transport fixtures only; generated code is not evaluated as model behavior.
        for client, payload in (
            (review_client, {"summary": "离线请求捕获", "issues": []}),
            (fix_client, {"code": code, "summary": "离线请求捕获"}),
        ):
            client.chat.completions.create.return_value = SimpleNamespace(choices=[SimpleNamespace(
                finish_reason="stop", message=SimpleNamespace(refusal=None, content=json.dumps(payload)),
            )])
        context = "按本次需求处理错误文案，不扩展注册功能。"
        options = {"api_key": "offline-test-key", "base_url": "https://example.invalid/v1", "model": "offline-policy"}
        with patch("llm.load_dotenv"), patch("llm.OpenAI", side_effect=AssertionError("Real SDK client forbidden")):
            reviewer = OpenAIReviewer(client=review_client, **options)
            fixer = OpenAICodeFixer(client=fix_client, **options)
            report = CodeReviewAgent(reviewer).review_detailed(code, review_context=ReviewContext(
                requirements=requirements, context=context, test_cases=[case], test_result=result,
            )).report
            fixer.fix(code, report=report, test_result=result, test_cases=[case],
                      requirements=requirements, context=context)
        review_client.chat.completions.create.assert_called_once()
        fix_client.chat.completions.create.assert_called_once()
        review_messages = review_client.chat.completions.create.call_args.kwargs["messages"]
        fix_messages = fix_client.chat.completions.create.call_args.kwargs["messages"]
        review_payload = json.loads(review_messages[1]["content"].split(
            "<workflow_context>\n", 1,
        )[1].split("\n</workflow_context>", 1)[0])
        fix_payload = json.loads(fix_messages[1]["content"])
        for payload in (review_payload, fix_payload):
            self.assertEqual(payload["requirements"], requirements)
            self.assertEqual(payload["context"], context)
            self.assertEqual(payload["test_cases"], [case.model_dump()])
        return (review_messages[0]["content"], fix_messages[0]["content"]), review_payload, fix_payload, report

    def test_review_and_fix_share_requirement_scoped_message_rules(self):
        for requirements in (
            "将 register_user 的空密码错误提示由旧提示改为新提示！。",
            "修复 register_user，保留原有错误提示的中英文、大小写和标点。",
            "修复 register_user 的输入校验，保持其他行为不变。",
        ):
            with self.subTest(requirements=requirements):
                case = GeneratedCase(name="test_message", code=(
                    "with self.assertRaisesRegex(ValueError, '^新提示！$'):\n"
                    "    module.register_user('')"
                ))
                systems, _, _, _ = self.capture_requests(requirements, case)
                for system in systems:
                    self.assertIn("用户需求或与需求一致的原测试指定错误提示时，必须保持中英文、大小写和标点一致", system)
                    self.assertIn("用户明确要求修改提示时遵循新要求", system)
                    self.assertIn("不为匹配当前实现改写测试", system)
                    self.assertNotIn("不修改原测试或原错误提示", system)

    def test_exact_and_regex_message_contracts_reach_both_stages_unchanged(self):
        cases = (
            (GeneratedCase(name="test_exact_message", code=(
                "with self.assertRaises(ValueError) as caught:\n"
                "    module.register_user('')\n"
                "self.assertEqual(str(caught.exception), 'Password required.')"
            )), "exact", "Password required."),
            (GeneratedCase(name="test_regex_message", code=(
                "with self.assertRaisesRegex(ValueError, '^口令不能为空！$'):\n"
                "    module.register_user('')"
            )), "regex", "^口令不能为空！$"),
        )
        for case, kind, expected in cases:
            with self.subTest(kind=kind):
                _, review_payload, fix_payload, _ = self.capture_requests("保留原测试指定的错误提示。", case)
                self.assertEqual(review_payload["test_cases"][0]["code"], case.code)
                contract = fix_payload["test_contract"][0]
                self.assertEqual(contract["code"], case.code)
                self.assertEqual((contract["expectations"][0]["kind"], contract["expectations"][0]["expected"]),
                                 (kind, expected))

    def test_conflicting_old_contract_keeps_disclosure_rule_and_real_failure(self):
        requirements = "将 register_user 的空密码错误提示由旧提示改为新提示！。"
        case = GeneratedCase(name="test_old_message", code=(
            "with self.assertRaisesRegex(ValueError, '^旧提示$'):\n"
            "    module.register_user('')"
        ))
        result = ExecutionResult(status="failed", tests_run=1, failed=1, failures=[FailureDetail(
            test=case.name, test_name=case.name, traceback="AssertionError: old message differs",
        )])
        systems, review_payload, fix_payload, report = self.capture_requests(requirements, case, result)
        for system in systems:
            self.assertIn("若与明确用户需求冲突，以用户需求为准", system)
            self.assertIn("在 summary 中说明冲突，不改写测试或宣称冲突已解决", system)
        self.assertEqual(review_payload["test_result"]["status"], "failed")
        self.assertEqual(fix_payload["test_result"]["status"], "failed")
        self.assertEqual(fix_payload["failed_test_contexts"][0]["test_code"], case.code)
        self.assertEqual(fix_payload["test_contract"][0]["expectations"][0]["expected"], "^旧提示$")
        self.assertEqual(report.test_status, "failed")
        self.assertTrue(any(issue.category == "test_failure" for issue in report.issues))


if __name__ == "__main__":
    unittest.main()
