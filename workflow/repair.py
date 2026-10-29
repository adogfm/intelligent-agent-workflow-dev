"""Bounded Review -> Fix -> Test -> Review iteration with per-round records."""

from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Any, Literal, Mapping

from models import ReviewReport, TestExecutionResult
from workflow.agent import Agent, AgentResult, BaseAgent
from workflow.engine import (
    END,
    WorkflowConfigurationError,
    WorkflowEngine,
    WorkflowError,
    WorkflowRun,
    WorkflowStep,
)
from workflow.state import StateView, WorkflowState, WorkflowStatus, _copy_state_data


@dataclass(frozen=True)
class RepairStepExecution:
    step: str
    agent: str
    state_version: int
    duration_seconds: float
    output: Any = None
    error: str | None = None


@dataclass(frozen=True)
class RepairDecision:
    """Auditable Review/Test decision made after one repair stage."""

    after_stage: Literal["review", "fix", "test", "error"]
    review_passed: bool | None
    test_passed: bool | None
    test_status: str | None
    attempt: int
    max_attempts: int
    next_step: str
    reason: Literal[
        "baseline_test_required",
        "repair_required",
        "code_changed",
        "review_required",
        "resolved",
        "max_fix_attempts",
        "no_change",
        "error",
    ]


@dataclass(frozen=True)
class RepairRound:
    attempt: int
    code_before: str
    code_after: str
    report_before: ReviewReport | None
    report_after: ReviewReport | None
    test_result_before: TestExecutionResult | None
    test_result_after: TestExecutionResult | None
    fix_result: Any
    executions: tuple[RepairStepExecution, ...]
    status: Literal["running", "resolved", "unresolved", "unchanged", "error"]
    error: str | None = None


def _review_passed(values: Mapping[str, Any]) -> bool | None:
    report = values.get("report")
    return None if not isinstance(report, ReviewReport) else not report.issues


def _test_passed(values: Mapping[str, Any]) -> bool | None:
    result = values.get("test_result")
    if not isinstance(result, TestExecutionResult):
        return None
    return (
        result.status == "passed"
        and result.tests_run > 0 and result.passed > 0
        and not result.failed and not result.errors and not result.unexpected_successes
    )


def _resolved(values: Mapping[str, Any]) -> bool:
    return _review_passed(values) is True and _test_passed(values) is True


def _decision_after(
    stage: Literal["review", "fix", "test"],
    values: Mapping[str, Any],
    max_attempts: int,
) -> RepairDecision:
    attempt = values.get("fix_attempts", 0)
    review_passed = _review_passed(values)
    test_passed = _test_passed(values)
    test_result = values.get("test_result")
    test_status = test_result.status if isinstance(test_result, TestExecutionResult) else None
    stop_reason = values.get("repair_stop_reason")

    if stage == "review" and not values.get("_repair_tested"):
        next_step, reason = "test", "baseline_test_required"
    elif stage == "fix" and values.get("fix_changed"):
        next_step, reason = "test", "code_changed"
    elif stage == "test" and values.get("active_repair_round") is not None:
        next_step, reason = "review", "review_required"
    elif stop_reason is not None:
        next_step, reason = END, stop_reason
    else:
        next_step, reason = "fix", "repair_required"

    return RepairDecision(
        after_stage=stage,
        review_passed=review_passed,
        test_passed=test_passed,
        test_status=test_status,
        attempt=attempt,
        max_attempts=max_attempts,
        next_step=next_step,
        reason=reason,
    )


class _RepairStage(BaseAgent):
    """Add round metadata to a domain agent's existing atomic state update."""

    def __init__(self, stage: str, agent: Agent, max_fix_attempts: int) -> None:
        self._stage = stage
        self._agent = agent
        self._max_fix_attempts = max_fix_attempts
        self.name = agent.name
        self.last_failure: RepairStepExecution | None = None

    def run(self, state: StateView) -> AgentResult:
        started = time.monotonic()
        self.last_failure = None
        try:
            return self._execute(state, started)
        except Exception as exc:
            self.last_failure = RepairStepExecution(
                self._stage, self.name, state.version, time.monotonic() - started, error=str(exc),
            )
            raise

    def _execute(self, state: StateView, started: float) -> AgentResult:
        if self._stage == "fix" and state["fix_attempts"] >= self._max_fix_attempts:
            raise ValueError("已达到最大修复次数，不能继续调用 Fix Agent。")
        view = state
        if self._stage == "test" and state.get("test_cases"):
            # Retest the same assertions across all rounds, including no-change fixes.
            view = StateView(
                {**state.to_dict(), "reuse_test_cases": True}, version=state.version,
                status=state.status, current_step=state.current_step,
                iteration_count=state.iteration_count,
                current_repair_round=state.current_repair_round,
                stages=state.stages,
                stage_history=state.stage_history, agent_results=state.agent_results,
                test_results=state.test_results, execution_logs=state.execution_logs,
            )
        result = self._agent.execute(view)
        if not isinstance(result, AgentResult):
            raise TypeError(f"Agent {self.name!r} 必须返回 AgentResult")
        updates = dict(result.updates)
        values = {**state.to_dict(), **updates}
        # Repair history must support the same isolated value types as state.
        execution = RepairStepExecution(
            self._stage, self.name, state.version + 1,
            time.monotonic() - started, _copy_state_data(result.output),
        )
        active = state.get("active_repair_round")
        history = state["repair_history"]

        if self._stage == "fix":
            attempt = state["fix_attempts"] + 1
            previous_review = state.get("_repair_last_review")
            executions = ((previous_review,) if previous_review is not None else ()) + (execution,)
            changed = state["code"] != values["code"]
            active = RepairRound(
                attempt, state["code"], values["code"],
                _copy_state_data(state.get("report")), _copy_state_data(values.get("report")),
                _copy_state_data(state.get("test_result")), _copy_state_data(values.get("test_result")),
                _copy_state_data(result.output), executions, "running" if changed else "unchanged",
            )
            updates.update({"fix_attempts": attempt, "fix_changed": changed})
            if not changed:
                updates.update({
                    "repair_history": (*history, active), "active_repair_round": None,
                    "repair_stop_reason": "no_change", "repair_succeeded": False,
                })
            else:
                updates["active_repair_round"] = active
        elif self._stage == "test":
            if not isinstance(values.get("test_result"), TestExecutionResult):
                raise TypeError("Test Agent 必须写入 TestExecutionResult。")
            updates["_repair_tested"] = True
            if active is not None:
                updates["active_repair_round"] = replace(
                    active, test_result_after=_copy_state_data(values["test_result"]),
                    executions=(*active.executions, execution),
                )
            elif _resolved(values):
                updates.update({"repair_stop_reason": "resolved", "repair_succeeded": True})
            elif state["fix_attempts"] >= self._max_fix_attempts:
                updates.update({"repair_stop_reason": "max_fix_attempts", "repair_succeeded": False})
        else:
            if not isinstance(values.get("report"), ReviewReport):
                raise TypeError("Review Agent 必须写入 ReviewReport。")
            updates["_repair_last_review"] = execution
            if active is not None:
                closed = replace(
                    active, report_after=_copy_state_data(values["report"]),
                    status="resolved" if _resolved(values) else "unresolved",
                    executions=(*active.executions, execution),
                )
                updates.update({"repair_history": (*history, closed), "active_repair_round": None})
            if values.get("_repair_tested") and _resolved(values):
                updates.update({"repair_stop_reason": "resolved", "repair_succeeded": True})
            elif values.get("_repair_tested") and state["fix_attempts"] >= self._max_fix_attempts:
                # With zero attempts, still obtain a baseline test result first.
                if isinstance(values.get("test_result"), TestExecutionResult):
                    updates.update({"repair_stop_reason": "max_fix_attempts", "repair_succeeded": False})
        decision_values = {**state.to_dict(), **updates}
        updates["repair_decision"] = _decision_after(
            self._stage, decision_values, self._max_fix_attempts,
        )
        # Routing belongs to the bounded loop, rather than an agent's next_step.
        return AgentResult(updates=updates, output=result.output)


class RepairWorkflowEngine(WorkflowEngine):
    """A normal WorkflowEngine graph with a separate code-repair budget."""

    def __init__(
        self, review_agent: Agent, fix_agent: Agent, test_agent: Agent,
        *, max_fix_attempts: int = 3,
    ) -> None:
        if isinstance(max_fix_attempts, bool) or not isinstance(max_fix_attempts, int) or max_fix_attempts < 0:
            raise ValueError("max_fix_attempts 必须是非负整数。")
        if not all(isinstance(agent, Agent) for agent in (review_agent, fix_agent, test_agent)):
            raise WorkflowConfigurationError("Review、Fix、Test 节点必须实现统一 Agent 接口。")
        super().__init__("repair", max_steps=max(100, 3 * max_fix_attempts + 5))
        self.max_fix_attempts = max_fix_attempts
        self.add_step("review", _RepairStage("review", review_agent, max_fix_attempts), router=self._after_review)
        self.add_step("fix", _RepairStage("fix", fix_agent, max_fix_attempts), router=self._after_fix)
        self.add_step("test", _RepairStage("test", test_agent, max_fix_attempts), router=self._after_test)

    def run(self, initial_state: WorkflowState | Mapping[str, Any] | None = None) -> WorkflowRun:
        state = initial_state if isinstance(initial_state, WorkflowState) else WorkflowState(initial_state)
        if state.status is not WorkflowStatus.PENDING:
            raise RuntimeError(f"Workflow 状态不能从 {state.status.value} 重新启动")
        # Budgets and history belong to this run, not untrusted initial metadata.
        state.apply({
            "fix_attempts": 0, "max_fix_attempts": self.max_fix_attempts,
            "repair_history": (), "active_repair_round": None,
            "repair_stop_reason": None, "repair_succeeded": None, "_repair_last_review": None,
            "_repair_tested": False, "repair_decision": None,
        }, source="repair_init")
        try:
            return super().run(state)
        except WorkflowError as exc:
            self._record_failure(state, exc)
            raise

    def _record_failure(self, state: WorkflowState, error: WorkflowError) -> None:
        active = state.get("active_repair_round")
        history = state.get("repair_history")
        cause = error.__cause__ or error
        failure = RepairStepExecution(
            state.current_step or "unknown", "unknown", state.version, 0.0, error=str(cause),
        )
        step = self._steps.get(state.current_step)
        if step is not None:
            failure = replace(failure, agent=step.agent.name)
        if step is not None and isinstance(step.agent, _RepairStage) and step.agent.last_failure is not None:
            failure = step.agent.last_failure
        attempts = state.get("fix_attempts")
        if active is None and state.current_step == "fix":
            attempts += 1
            previous_review = state.get("_repair_last_review")
            active = RepairRound(
                attempts, state.get("code"), state.get("code"),
                _copy_state_data(state.get("report")), _copy_state_data(state.get("report")),
                _copy_state_data(state.get("test_result")), _copy_state_data(state.get("test_result")),
                None, (previous_review,) if previous_review is not None else (), "running",
            )
        if active is not None:
            failed = replace(active, status="error", error=str(cause),
                             executions=(*active.executions, failure))
            history = (*history, failed)
        state.apply({
            "fix_attempts": attempts, "repair_history": history, "active_repair_round": None,
            "repair_stop_reason": "error", "repair_succeeded": False,
            "repair_decision": RepairDecision(
                after_stage="error",
                review_passed=_review_passed(state.to_dict()),
                test_passed=_test_passed(state.to_dict()),
                test_status=(
                    state.get("test_result").status
                    if isinstance(state.get("test_result"), TestExecutionResult)
                    else None
                ),
                attempt=attempts,
                max_attempts=self.max_fix_attempts,
                next_step=END,
                reason="error",
            ),
        }, source="repair_error")
        state.log(
            "repair.decision",
            f"轮次 {attempts} 因 Agent 异常终止：{cause}",
            level="error",
            stage=state.current_step,
            repair_round=attempts,
        )

    @staticmethod
    def _after_review(state: StateView, result: AgentResult) -> str:
        return RepairWorkflowEngine._decision_target(state)

    @staticmethod
    def _after_fix(state: StateView, result: AgentResult) -> str:
        return RepairWorkflowEngine._decision_target(state)

    @staticmethod
    def _after_test(state: StateView, result: AgentResult) -> str:
        return RepairWorkflowEngine._decision_target(state)

    @staticmethod
    def _decision_target(state: StateView) -> str:
        decision = state.get("repair_decision")
        if not isinstance(decision, RepairDecision):
            raise WorkflowConfigurationError("修复阶段缺少有效的条件分支决策。")
        return decision.next_step

    def _repair_round_for_step(
        self,
        state: WorkflowState,
        step: WorkflowStep,
    ) -> int | None:
        if step.name == "fix":
            return state.repair_iteration_count + 1
        if step.name in {"review", "test"}:
            return state.repair_iteration_count
        return None

    def _record_route(
        self,
        state: WorkflowState,
        step: WorkflowStep,
        next_step: str,
    ) -> None:
        super()._record_route(state, step, next_step)
        decision = state.get("repair_decision")
        if isinstance(decision, RepairDecision):
            state.log(
                "repair.decision",
                (
                    f"轮次 {decision.attempt} 在 {decision.after_stage} 后选择 "
                    f"{decision.next_step!r}：{decision.reason}；"
                    f"review={decision.review_passed}，test={decision.test_status}"
                ),
                stage=step.name,
                agent=step.agent.name,
            )


def create_repair_workflow(
    review_agent: Agent, fix_agent: Agent, test_agent: Agent,
    *, max_fix_attempts: int = 3,
) -> RepairWorkflowEngine:
    """Review existing code and repair it until resolved or the budget is spent."""

    return RepairWorkflowEngine(review_agent, fix_agent, test_agent, max_fix_attempts=max_fix_attempts)
