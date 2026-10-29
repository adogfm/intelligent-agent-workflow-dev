"""Offline tests for the reusable workflow framework."""

from __future__ import annotations

import json
import unittest
import weakref
from collections import deque
from dataclasses import dataclass
from types import MappingProxyType, SimpleNamespace

from pydantic import BaseModel

from agent import CodeReviewAgent, ReviewAgent, create_review_workflow
from coding_agent import CodingAgent
from fix_agent import FixAgent
from llm import MockReviewer
from models import ReviewIssue, ReviewReport
from test_agent import TestAgent as WorkflowTestAgent
from workflow import (
    END,
    AgentResult,
    BaseAgent,
    StageStatus,
    StateView,
    WorkflowEngine,
    WorkflowExecutionError,
    WorkflowOutcome,
    WorkflowState,
    WorkflowStatus,
    generate_workflow_report,
)


class FunctionAgent(BaseAgent):
    def __init__(self, name: str, function) -> None:
        self.name = name
        self._function = function

    def run(self, state: StateView) -> AgentResult:
        return self._function(state)


class WorkflowStateTests(unittest.TestCase):
    def test_snapshot_rejects_weakrefs_in_bound_method_receivers(self) -> None:
        class Receiver:
            def __init__(self, target: object) -> None:
                self.target = target

            def read(self):
                return self.target

        class SlottedReceiver:
            __slots__ = ("target",)

            def __init__(self, target: object) -> None:
                self.target = target

            def read(self):
                return self.target

        original = ReviewReport(summary="original", issues=[])
        references = (
            ("ref", weakref.ref(original)),
            ("weak method", weakref.WeakMethod(original.model_dump)),
        )
        for kind, reference in references:
            receivers = (
                ("attribute", Receiver(reference)),
                ("nested container", Receiver({"links": [reference]})),
                ("nested object", Receiver(SimpleNamespace(reference=reference))),
                ("slots", SlottedReceiver(reference)),
            )
            for receiver_location, receiver in receivers:
                callback = receiver.read
                wrappers = (
                    ("direct", callback),
                    ("nested container", {"callbacks": [(callback,)]}),
                    ("object attribute", SimpleNamespace(callback=callback)),
                )
                for callback_location, value in wrappers:
                    with self.subTest(reference=kind, receiver=receiver_location, callback=callback_location):
                        state = WorkflowState({"payload": original, "value": value})
                        with self.assertRaisesRegex(TypeError, r"weakref\.ref"):
                            state.snapshot()
                        self.assertEqual(original.summary, "original")
                        self.assertEqual(state.version, 0)

    def test_bound_method_without_weakrefs_preserves_snapshot_isolation(self) -> None:
        class Receiver:
            def __init__(self, target: dict) -> None:
                self.target = target
                self.callback = self.mutate

            def mutate(self) -> None:
                self.target["items"].append("local")

        payload = {"items": ["original"]}
        receiver = Receiver(payload)
        state = WorkflowState({"payload": payload, "callback": receiver.callback})

        snapshot = state.snapshot()
        copied_receiver = snapshot["callback"].__self__
        self.assertIsNot(copied_receiver, receiver)
        self.assertIs(copied_receiver.target, snapshot["payload"])
        self.assertIs(copied_receiver.callback.__self__, copied_receiver)
        snapshot["callback"]()
        exported = snapshot.to_dict()
        exported["callback"]()

        self.assertEqual(state.get("payload"), {"items": ["original"]})
        self.assertEqual(snapshot["payload"], {"items": ["original", "local"]})
        self.assertEqual(exported["payload"], {"items": ["original", "local", "local"]})

    def test_snapshot_rejects_nested_weakrefs(self) -> None:
        @dataclass
        class Payload:
            items: list[str]

            def mutate(self) -> None:
                self.items.append("uncommitted")

        @dataclass
        class Record:
            reference: object

        class SlottedRecord:
            __slots__ = ("reference",)

            def __init__(self, reference: object) -> None:
                self.reference = reference

        class ItemList(list):
            pass

        class ItemDict(dict):
            pass

        original = Payload(["original"])
        reference = weakref.ref(original)
        items = ItemList()
        items.reference = reference
        mapping = ItemDict()
        mapping.reference = reference
        cases = (
            ("direct", reference),
            ("nested", {"items": [reference]}),
            ("proxy", MappingProxyType({"reference": reference})),
            ("dataclass", Record(reference)),
            ("slots", SlottedRecord(reference)),
            ("list attribute", items),
            ("mapping attribute", mapping),
            ("deque", deque([reference])),
            ("weak method", weakref.WeakMethod(original.mutate)),
        )

        for name, value in cases:
            with self.subTest(location=name):
                state = WorkflowState({"payload": original, "value": value})
                with self.assertRaisesRegex(TypeError, "weakref.ref"):
                    state.snapshot()
                self.assertEqual(original.items, ["original"])
                self.assertEqual(state.version, 0)

    def test_weakref_update_is_rejected_without_partial_commit(self) -> None:
        original = ReviewReport(summary="original", issues=[])
        state = WorkflowState({"value": "original"})

        with self.assertRaisesRegex(TypeError, "weakref.ref"):
            state.apply({"value": "replacement", "reference": weakref.ref(original)}, source="prepare")

        self.assertEqual(state.to_dict(), {"value": "original"})
        self.assertEqual(state.version, 0)
        self.assertEqual(state.history, ())
        self.assertEqual(state.execution_logs, ())

    def test_committed_updates_detach_input_and_preserve_internal_aliases(self) -> None:
        payload = {"items": ["committed"]}
        test_result = {"items": ["passed"]}
        state = WorkflowState()
        state.apply({"payload": payload, "alias": payload, "test_result": test_result}, source="prepare")

        payload["items"].append("uncommitted")
        test_result["items"].append("uncommitted")

        self.assertEqual(state.get("payload"), {"items": ["committed"]})
        self.assertIs(state.get("payload"), state.get("alias"))
        self.assertEqual(state.test_results, ({"items": ["passed"]},))
        self.assertIs(state.get("test_result"), state.latest_test_result)
        self.assertEqual(state.version, 1)

    def test_agent_record_detaches_updates_and_output_from_retained_input(self) -> None:
        payload = {"items": ["committed"]}
        state = WorkflowState()
        state.register_stages({"prepare": "prepare-agent"})
        state.start()
        state.begin_step("prepare", agent="prepare-agent")
        state.record_agent_result(
            agent="prepare-agent", updates={"payload": payload}, output=payload,
            updated_keys=("payload",), next_step=None,
        )

        payload["items"].append("uncommitted")

        record = state.agent_results[0]
        self.assertEqual(record.updates["payload"], {"items": ["committed"]})
        self.assertIs(record.updates["payload"], record.output)
        with self.assertRaises(TypeError):
            record.updates["other"] = True

    def test_nested_mappingproxies_in_snapshot_and_export_are_isolated(self) -> None:
        items = ["original"]
        proxy = MappingProxyType({"items": items})
        state = WorkflowState({"payload": {"proxy": proxy, "items": items}})
        snapshot = state.snapshot()
        exported = snapshot.to_dict()

        self.assertIsInstance(snapshot["payload"]["proxy"], MappingProxyType)
        self.assertIs(snapshot["payload"]["proxy"]["items"], snapshot["payload"]["items"])
        snapshot["payload"]["proxy"]["items"].append("snapshot")
        exported["payload"]["proxy"]["items"].append("export")

        self.assertEqual(items, ["original"])
        self.assertEqual(snapshot["payload"]["items"], ["original", "snapshot"])
        self.assertEqual(exported["payload"]["items"], ["original", "export"])
        with self.assertRaises(TypeError):
            snapshot["payload"]["proxy"]["other"] = True

    def test_mappingproxies_inside_objects_preserve_types_and_isolation(self) -> None:
        @dataclass(frozen=True)
        class Record:
            data: object

        class SlottedRecord:
            __slots__ = ("data",)

            def __init__(self, data: object) -> None:
                self.data = data

        for wrapper in (Record, SlottedRecord, lambda data: SimpleNamespace(data=data)):
            with self.subTest(wrapper=wrapper):
                items = ["original"]
                original = wrapper(MappingProxyType({"items": items}))
                copied = WorkflowState({"record": original}).snapshot()["record"]

                self.assertIs(type(copied), type(original))
                self.assertIsInstance(copied.data, MappingProxyType)
                copied.data["items"].append("local")
                self.assertEqual(items, ["original"])

    def test_mappingproxy_cycles_and_shared_references_survive_copy(self) -> None:
        backing = {"items": []}
        proxy = MappingProxyType(backing)
        backing["self"] = proxy
        state = WorkflowState()
        state.apply({"proxy": proxy, "alias": proxy}, source="prepare")
        backing["items"].append("uncommitted")
        self.assertEqual(state.get("proxy")["items"], [])
        snapshot = state.snapshot()

        self.assertIs(snapshot["proxy"], snapshot["alias"])
        self.assertIs(snapshot["proxy"], snapshot["proxy"]["self"])
        exported = snapshot.to_dict()
        self.assertIs(exported["proxy"], exported["proxy"]["self"])
        exported["proxy"]["items"].append("export")
        self.assertEqual(snapshot["proxy"]["items"], [])
        self.assertEqual(state.get("proxy")["items"], [])

    def test_copy_failure_leaves_commit_and_history_unchanged(self) -> None:
        class Uncopyable:
            def __deepcopy__(self, memo):
                raise TypeError("not copyable")

        state = WorkflowState({"payload": {"items": ["original"]}})
        with self.assertRaisesRegex(TypeError, "not copyable"):
            state.apply({"payload": {"items": ["new"]}, "bad": Uncopyable()}, source="prepare")

        self.assertEqual(state.get("payload"), {"items": ["original"]})
        self.assertIsNone(state.get("bad"))
        self.assertEqual(state.version, 0)
        self.assertEqual(state.history, ())
        self.assertEqual(state.execution_logs, ())

    def test_nested_snapshot_values_are_detached_from_state(self) -> None:
        report = ReviewReport(summary="original", issues=[])
        state = WorkflowState({
            "payload": {"items": [{"tags": {"original"}}]},
            "report": report,
            "repair_history": [{"notes": ["original"]}],
        })
        snapshot = state.snapshot()
        other_snapshot = state.snapshot()

        snapshot["payload"]["items"][0]["tags"].add("local")
        snapshot.get("payload")["items"].append({"tags": set()})
        snapshot.require("report", ReviewReport).issues.append(ReviewIssue(
            problem="local", severity="low", reason="local", suggestion="local",
        ))
        snapshot.repair_history[0]["notes"].append("local")

        self.assertEqual(state.get("payload"), {"items": [{"tags": {"original"}}]})
        self.assertEqual(other_snapshot["payload"], state.get("payload"))
        self.assertEqual(report.issues, [])
        self.assertEqual(state.repair_history, ({"notes": ["original"]},))
        self.assertEqual(state.version, 0)
        self.assertEqual(state.history, ())

    def test_snapshot_and_export_remain_independent_after_nested_changes(self) -> None:
        state = WorkflowState({"payload": {"items": ["original"]}})
        snapshot = state.snapshot()
        exported = snapshot.to_dict()

        exported["payload"]["items"].append("exported")
        state.get("payload")["items"].append("owner")

        self.assertEqual(snapshot["payload"], {"items": ["original"]})
        self.assertEqual(exported["payload"], {"items": ["original", "exported"]})
        self.assertEqual(state.get("payload"), {"items": ["original", "owner"]})

    def test_snapshot_detaches_nested_agent_and_test_history(self) -> None:
        state = WorkflowState({"test_result": {"items": ["original"]}})
        state.register_stages({"prepare": "prepare-agent"})
        state.start()
        state.begin_step("prepare", agent="prepare-agent")
        state.record_agent_result(
            agent="prepare-agent",
            updates={"payload": {"items": ["original"]}},
            output={"items": ["original"]},
            updated_keys=("payload",), next_step=None,
        )
        snapshot = state.snapshot()

        snapshot.agent_results[0].updates["payload"]["items"].append("local")
        snapshot.agent_results[0].output["items"].append("local")
        snapshot.test_results[0]["items"].append("local")
        snapshot.latest_test_result["items"].append("local")

        self.assertEqual(state.agent_results[0].updates["payload"], {"items": ["original"]})
        self.assertEqual(state.agent_results[0].output, {"items": ["original"]})
        self.assertEqual(state.test_results, ({"items": ["original"]},))
        self.assertEqual(state.latest_test_result, {"items": ["original"]})
        with self.assertRaises(TypeError):
            snapshot.agent_results[0].updates["other"] = True

    def test_snapshot_is_read_only_and_tracks_atomic_changes(self) -> None:
        state = WorkflowState({"code": "print('ok')"})
        state.apply({"count": 1, "ready": True}, source="prepare")

        snapshot = state.snapshot()

        self.assertEqual(snapshot.require("count", int), 1)
        self.assertEqual(state.version, 1)
        self.assertEqual(state.history[0].source, "prepare")
        self.assertEqual(state.history[0].updated_keys, ("count", "ready"))
        with self.assertRaises(TypeError):
            snapshot._values["count"] = 2
        with self.assertRaisesRegex(ValueError, "已完成或失败"):
            generate_workflow_report(state)

    def test_tracks_test_results_repair_history_and_domain_iterations(self) -> None:
        first = {"result": "first"}
        second = {"result": "second"}
        state = WorkflowState({"test_result": first})

        state.apply({"test_result": None}, source="fix")
        state.apply({"test_result": second}, source="test")
        state.apply({"repair_history": ("round-1",), "fix_attempts": 1}, source="repair")

        self.assertEqual(state.test_results, (first, second))
        self.assertEqual(state.latest_test_result, second)
        self.assertIsNot(state.latest_test_result, second)
        self.assertEqual(state.repair_history, ("round-1",))
        self.assertEqual(state.repair_iteration_count, 1)
        self.assertEqual(state.snapshot().repair_history, ("round-1",))

    def test_runtime_metadata_collections_are_read_only_snapshots(self) -> None:
        state = WorkflowState()
        state.register_stages({"prepare": "prepare-agent"})
        state.start()
        state.begin_step("prepare", agent="prepare-agent")
        snapshot = state.snapshot()

        self.assertEqual(snapshot.stage_statuses["prepare"], StageStatus.RUNNING)
        self.assertEqual(snapshot.iteration_count, 1)
        with self.assertRaises(TypeError):
            snapshot.stages["other"] = snapshot.stages["prepare"]


class WorkflowEngineTests(unittest.TestCase):
    def test_nested_bound_method_weakref_is_rejected_before_agent_execution(self) -> None:
        class Receiver:
            def __init__(self, target: object) -> None:
                self.reference = weakref.ref(target)

            def mutate(self) -> None:
                self.reference().summary = "uncommitted"

        for location in ("nested container", "object attribute"):
            with self.subTest(location=location):
                original = ReviewReport(summary="original", issues=[])
                receiver = Receiver(original)
                value = {"callbacks": [receiver.mutate]} if location == "nested container" else SimpleNamespace(callback=receiver.mutate)
                state = WorkflowState({"payload": original, "value": value})
                calls = []

                def fail(snapshot: StateView) -> AgentResult:
                    calls.append("executed")
                    callback = snapshot["value"]["callbacks"][0] if location == "nested container" else snapshot["value"].callback
                    callback()
                    raise ValueError("failed after callback mutation")

                with self.assertRaisesRegex(WorkflowExecutionError, r"weakref\.ref"):
                    WorkflowEngine().add_step("fail", FunctionAgent("broken", fail)).run(state)

                self.assertEqual(calls, [])
                self.assertEqual(state.get("payload").summary, "original")
                self.assertEqual(state.version, 0)
                self.assertEqual(state.history, ())
                self.assertEqual(state.agent_results, ())
                self.assertEqual(state.stage_statuses["fail"], StageStatus.FAILED)

    def test_bound_method_weakref_result_is_rejected_before_commit(self) -> None:
        class Receiver:
            def __init__(self, target: object) -> None:
                self.reference = weakref.ref(target)

            def read(self):
                return self.reference()

        original = ReviewReport(summary="original", issues=[])
        receiver = Receiver(original)
        for location in ("updates", "output"):
            with self.subTest(location=location):
                value = SimpleNamespace(callbacks={"items": [receiver.read]})
                result = AgentResult(
                    updates={"value": 42, "callbacks": value} if location == "updates" else {"value": 42},
                    output=value if location == "output" else None,
                )
                state = WorkflowState({"value": 0})
                agent = FunctionAgent("producer", lambda snapshot: result)

                with self.assertRaisesRegex(WorkflowExecutionError, r"weakref\.ref"):
                    WorkflowEngine().add_step("produce", agent).run(state)

                self.assertEqual(state.to_dict(), {"value": 0})
                self.assertEqual(state.version, 0)
                self.assertEqual(state.history, ())
                self.assertEqual(state.agent_results, ())
                self.assertNotIn("state.updated", [item.event for item in state.execution_logs])

    def test_initial_weakref_is_rejected_before_agent_can_mutate_state(self) -> None:
        original = ReviewReport(summary="original", issues=[])
        state = WorkflowState({"payload": original, "reference": weakref.ref(original)})
        calls = []

        def fail(snapshot: StateView) -> AgentResult:
            calls.append("executed")
            snapshot["reference"]().summary = "uncommitted"
            raise ValueError("failed after weakref mutation")

        with self.assertRaisesRegex(WorkflowExecutionError, "weakref.ref"):
            WorkflowEngine().add_step("fail", FunctionAgent("broken", fail)).run(state)

        self.assertEqual(calls, [])
        self.assertEqual(state.get("payload").summary, "original")
        self.assertEqual(state.version, 0)
        self.assertEqual(state.history, ())
        self.assertEqual(state.agent_results, ())
        self.assertEqual(state.stage_statuses["fail"], StageStatus.FAILED)
        self.assertEqual(generate_workflow_report(state).outcome, WorkflowOutcome.FAILED)

    def test_weakref_agent_result_is_rejected_before_commit(self) -> None:
        original = ReviewReport(summary="original", issues=[])
        reference = weakref.ref(original)
        for location in ("updates", "output"):
            with self.subTest(location=location):
                result = AgentResult(
                    updates={"value": 42, "reference": reference} if location == "updates" else {"value": 42},
                    output={"reference": reference} if location == "output" else None,
                )
                state = WorkflowState({"value": 0})
                agent = FunctionAgent("producer", lambda snapshot: result)

                with self.assertRaisesRegex(WorkflowExecutionError, "weakref.ref"):
                    WorkflowEngine().add_step("produce", agent).run(state)

                self.assertEqual(state.to_dict(), {"value": 0})
                self.assertEqual(state.version, 0)
                self.assertEqual(state.history, ())
                self.assertEqual(state.agent_results, ())
                events = [item.event for item in state.execution_logs]
                self.assertNotIn("state.updated", events)
                self.assertNotIn("agent.completed", events)

    def test_execution_outputs_detach_reused_agent_values(self) -> None:
        class ReusingAgent(BaseAgent):
            name = "reusing"

            def __init__(self) -> None:
                self.output = {"items": []}

            def run(self, state: StateView) -> AgentResult:
                self.output["items"].append(state.iteration_count)
                return AgentResult(updates={"payload": self.output}, output=self.output)

        agent = ReusingAgent()
        run = WorkflowEngine().add_step("first", agent, next_step="second").add_step(
            "second", agent,
        ).run()

        self.assertEqual(run.executions[0].output, {"items": [1]})
        self.assertEqual(run.output, {"items": [1, 2]})
        agent.output["items"].append("agent")
        run.executions[0].output["items"].append("execution")
        self.assertEqual(run.output, {"items": [1, 2]})
        run.output["items"].append("caller")
        self.assertEqual(run.state.get("payload"), {"items": [1, 2]})
        self.assertEqual(run.state.agent_results[0].output, {"items": [1]})
        self.assertEqual(run.state.agent_results[1].output, {"items": [1, 2]})
        self.assertEqual(run.report.agent_results[0].output, {"items": [1]})

    def test_completed_output_isolated_from_later_failed_agent_call(self) -> None:
        class RetainingAgent(BaseAgent):
            name = "retaining-output"

            def __init__(self) -> None:
                self.items = ["committed"]
                self.output = MappingProxyType({"items": self.items})
                self.fail = False

            def run(self, state: StateView) -> AgentResult:
                if self.fail:
                    self.items.append("uncommitted")
                    raise ValueError("later invocation failed")
                return AgentResult(output=self.output)

        agent = RetainingAgent()
        engine = WorkflowEngine().add_step("produce", agent)
        completed = engine.run()
        agent.fail = True

        with self.assertRaisesRegex(WorkflowExecutionError, "later invocation failed"):
            engine.run()

        self.assertIsInstance(completed.output, MappingProxyType)
        self.assertEqual(completed.output["items"], ["committed"])
        self.assertEqual(completed.state.agent_results[0].output["items"], ["committed"])
        self.assertEqual(completed.state.status, WorkflowStatus.COMPLETED)

    def test_execution_output_does_not_follow_router_edits(self) -> None:
        output = {"items": ["original"]}

        def route(snapshot: StateView, result: AgentResult) -> str:
            result.output["items"].append("router")
            snapshot.agent_results[-1].output["items"].append("snapshot")
            return END

        run = WorkflowEngine().add_step(
            "produce", FunctionAgent("producer", lambda state: AgentResult(output=output)),
            router=route,
        ).run()

        self.assertEqual(run.output, {"items": ["original"]})
        self.assertEqual(run.state.agent_results[0].output, {"items": ["original"]})

    def test_output_copy_failure_does_not_partially_commit_agent_result(self) -> None:
        class Uncopyable:
            def __deepcopy__(self, memo):
                raise TypeError("uncopyable output")

        agent = FunctionAgent("broken", lambda state: AgentResult(
            updates={"value": 42, "test_result": {"items": ["uncommitted"]}},
            output=Uncopyable(),
        ))
        state = WorkflowState({"value": 0, "test_result": {"items": ["baseline"]}})

        with self.assertRaisesRegex(WorkflowExecutionError, "uncopyable output"):
            WorkflowEngine().add_step("fail", agent).run(state)

        self.assertEqual(state.get("value"), 0)
        self.assertEqual(state.get("test_result"), {"items": ["baseline"]})
        self.assertEqual(state.test_results, ({"items": ["baseline"]},))
        self.assertEqual(state.version, 0)
        self.assertEqual(state.history, ())
        self.assertEqual(state.agent_results, ())
        self.assertEqual(state.stage_statuses["fail"], StageStatus.FAILED)
        events = [item.event for item in state.execution_logs]
        self.assertNotIn("state.updated", events)
        self.assertNotIn("agent.completed", events)
        self.assertEqual(generate_workflow_report(state).outcome, WorkflowOutcome.FAILED)

    def test_output_copy_failure_preserves_previous_successful_commit(self) -> None:
        class Uncopyable:
            def __deepcopy__(self, memo):
                raise TypeError("uncopyable output")

        prepare = FunctionAgent("prepare", lambda state: AgentResult(
            updates={"value": {"items": ["committed"]}}, output="prepared",
        ))
        fail = FunctionAgent("broken", lambda state: AgentResult(
            updates={"value": {"items": ["uncommitted"]}}, output=Uncopyable(),
        ))
        state = WorkflowState()
        engine = WorkflowEngine().add_step("prepare", prepare, next_step="fail").add_step("fail", fail)

        with self.assertRaisesRegex(WorkflowExecutionError, "uncopyable output"):
            engine.run(state)

        self.assertEqual(state.get("value"), {"items": ["committed"]})
        self.assertEqual(state.version, 1)
        self.assertEqual([change.source for change in state.history], ["prepare"])
        self.assertEqual(len(state.agent_results), 1)
        self.assertEqual(state.agent_results[0].output, "prepared")
        self.assertEqual(state.stage_statuses["prepare"], StageStatus.COMPLETED)
        self.assertEqual(state.stage_statuses["fail"], StageStatus.FAILED)

    def test_failure_during_second_result_copy_happens_before_commit(self) -> None:
        attempts = []

        class FailsOnSecondCopy:
            def __deepcopy__(self, memo):
                attempts.append("copy")
                if len(attempts) == 2:
                    raise TypeError("second copy failed")
                result = type(self)()
                memo[id(self)] = result
                return result

        agent = FunctionAgent("broken", lambda state: AgentResult(
            updates={"value": FailsOnSecondCopy()},
        ))
        state = WorkflowState({"value": "original"})

        with self.assertRaisesRegex(WorkflowExecutionError, "second copy failed"):
            WorkflowEngine().add_step("fail", agent).run(state)

        self.assertEqual(state.get("value"), "original")
        self.assertEqual(state.version, 0)
        self.assertEqual(state.history, ())
        self.assertEqual(state.agent_results, ())

    def test_failed_agent_cannot_mutate_a_snapshot_retained_across_commit(self) -> None:
        class RetainingAgent(BaseAgent):
            name = "retaining"

            def __init__(self) -> None:
                self.previous = None

            def run(self, snapshot: StateView) -> AgentResult:
                if self.previous is None:
                    self.previous = snapshot
                    payload = snapshot.require("payload", dict)
                    payload["items"].append("committed")
                    return AgentResult(updates={"payload": payload}, output=payload)
                self.previous["payload"]["items"].append("uncommitted")
                raise ValueError("boom after editing retained snapshot")

        agent = RetainingAgent()
        state = WorkflowState({"payload": {"items": ["original"]}})
        engine = WorkflowEngine().add_step("prepare", agent, next_step="fail").add_step("fail", agent)

        with self.assertRaisesRegex(WorkflowExecutionError, "editing retained snapshot"):
            engine.run(state)

        expected = {"items": ["original", "committed"]}
        self.assertEqual(state.get("payload"), expected)
        self.assertEqual(state.agent_results[0].updates["payload"], expected)
        self.assertEqual(state.agent_results[0].output, expected)
        self.assertEqual(state.version, 1)
        self.assertEqual([change.source for change in state.history], ["prepare"])
        self.assertEqual(len(state.agent_results), 1)
        self.assertEqual(state.status, WorkflowStatus.FAILED)

    def test_routing_snapshot_copy_failure_does_not_partially_commit(self) -> None:
        attempts = []

        class FailsOnThirdCopy:
            def __deepcopy__(self, memo):
                attempts.append("copy")
                if len(attempts) == 3:
                    raise TypeError("routing snapshot copy failed")
                result = type(self)()
                memo[id(self)] = result
                return result

        agent = FunctionAgent("broken", lambda state: AgentResult(
            updates={"value": 42}, output=FailsOnThirdCopy(),
        ))
        state = WorkflowState({"value": 0})

        with self.assertRaisesRegex(WorkflowExecutionError, "routing snapshot copy failed"):
            WorkflowEngine().add_step("fail", agent).run(state)

        self.assertEqual(len(attempts), 3)
        self.assertEqual(state.get("value"), 0)
        self.assertEqual(state.version, 0)
        self.assertEqual(state.history, ())
        self.assertEqual(state.agent_results, ())
        self.assertNotIn("state.updated", [item.event for item in state.execution_logs])

    def test_mappingproxy_output_allows_completion_and_isolated_history_reads(self) -> None:
        items = ["original"]
        output = MappingProxyType({"items": items})
        producer = FunctionAgent("producer", lambda state: AgentResult(output=output))
        single = WorkflowEngine().add_step("produce", producer).run()
        self.assertEqual(single.state.status, WorkflowStatus.COMPLETED)
        self.assertIsInstance(single.output, MappingProxyType)

        def consume(snapshot: StateView) -> AgentResult:
            record = snapshot.agent_results[0]
            self.assertIsInstance(record.output, MappingProxyType)
            record.output["items"].append("snapshot")
            with self.assertRaises(TypeError):
                record.output["other"] = True
            return AgentResult()

        run = WorkflowEngine().add_step("produce", producer, next_step="consume").add_step(
            "consume", FunctionAgent("consumer", consume),
        ).run()
        items.append("owner")

        self.assertEqual(run.state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(run.state.agent_results[0].output["items"], ["original"])
        self.assertEqual(single.state.agent_results[0].output["items"], ["original"])
        self.assertEqual(run.report.to_dict()["agent_results"][0]["output"], {"items": ["original"]})

    def test_failed_agent_cannot_mutate_nested_state_or_committed_history(self) -> None:
        prepare = FunctionAgent("prepare-agent", lambda state: AgentResult(
            updates={
                "payload": {"items": ["original"]},
                "report": ReviewReport(summary="original", issues=[]),
                "test_result": {"items": ["original"]},
                "repair_history": [{"notes": ["original"]}],
            },
            output={"items": ["original"]},
        ))

        def fail(snapshot: StateView) -> AgentResult:
            snapshot["payload"]["items"].append("uncommitted")
            snapshot.require("report", ReviewReport).summary = "uncommitted"
            snapshot.to_dict()["payload"]["items"].append("exported")
            snapshot.agent_results[0].updates["payload"]["items"].append("uncommitted")
            snapshot.agent_results[0].output["items"].append("uncommitted")
            snapshot.test_results[0]["items"].append("uncommitted")
            snapshot.latest_test_result["items"].append("uncommitted")
            snapshot.repair_history[0]["notes"].append("uncommitted")
            raise ValueError("boom after nested edits")

        state = WorkflowState()
        engine = WorkflowEngine().add_step("prepare", prepare, next_step="fail").add_step(
            "fail", FunctionAgent("broken", fail),
        )

        with self.assertRaisesRegex(WorkflowExecutionError, "boom after nested edits"):
            engine.run(state)

        self.assertEqual(state.get("payload"), {"items": ["original"]})
        self.assertEqual(state.get("report").summary, "original")
        self.assertEqual(state.repair_history, ({"notes": ["original"]},))
        self.assertEqual(state.test_results, ({"items": ["original"]},))
        self.assertEqual(state.agent_results[0].updates["payload"], {"items": ["original"]})
        self.assertEqual(state.agent_results[0].output, {"items": ["original"]})
        self.assertEqual(state.version, 1)
        self.assertEqual([change.source for change in state.history], ["prepare"])
        self.assertEqual(len(state.agent_results), 1)
        self.assertEqual(state.status, WorkflowStatus.FAILED)

    def test_nested_snapshot_edits_require_explicit_updates_to_commit(self) -> None:
        def discard(snapshot: StateView) -> AgentResult:
            snapshot["payload"]["items"].append("discarded")
            return AgentResult()

        def commit(snapshot: StateView) -> AgentResult:
            self.assertEqual(snapshot["payload"], {"items": ["original"]})
            payload = snapshot.require("payload", dict)
            payload["items"].append("committed")
            return AgentResult(updates={"payload": payload})

        run = WorkflowEngine().add_step(
            "discard", FunctionAgent("discard", discard), next_step="commit",
        ).add_step("commit", FunctionAgent("commit", commit)).run({
            "payload": {"items": ["original"]},
        })

        self.assertEqual(run.state.get("payload"), {"items": ["original", "committed"]})
        self.assertEqual(run.state.version, 1)
        self.assertEqual([change.source for change in run.state.history], ["commit"])

    def test_executes_steps_and_routes_from_result(self) -> None:
        prepare = FunctionAgent(
            "prepare-agent",
            lambda state: AgentResult(updates={"value": 2}, next_step="double"),
        )
        double = FunctionAgent(
            "double-agent",
            lambda state: AgentResult(
                updates={"value": state.require("value", int) * 2},
                output="done",
                next_step=END,
            ),
        )
        engine = WorkflowEngine().add_step("prepare", prepare).add_step("double", double)

        run = engine.run()

        self.assertEqual(run.state.get("value"), 4)
        self.assertEqual(run.state.status, WorkflowStatus.COMPLETED)
        self.assertEqual([item.step for item in run.executions], ["prepare", "double"])
        self.assertEqual(run.output, "done")
        report = run.report
        self.assertEqual(report.outcome, WorkflowOutcome.SUCCEEDED)
        self.assertTrue(report.succeeded)
        self.assertEqual(report.iteration_count, 2)
        self.assertEqual([item.name for item in report.stages], ["prepare", "double"])
        self.assertEqual(len(report.agent_results), 2)
        payload = report.to_dict()
        self.assertEqual(payload["workflow_status"], "completed")
        self.assertEqual(payload["outcome"], "succeeded")
        self.assertEqual(payload["agent_results"][-1]["output"], "done")

    def test_state_retains_stage_agent_and_execution_metadata(self) -> None:
        prepare = FunctionAgent(
            "prepare-agent",
            lambda state: AgentResult(updates={"value": 2}, output="prepared"),
        )
        finish = FunctionAgent(
            "finish-agent",
            lambda state: AgentResult(output=state["value"]),
        )
        run = WorkflowEngine().add_step(
            "prepare", prepare, next_step="finish",
        ).add_step("finish", finish).run()
        state = run.state

        self.assertEqual(state.iteration_count, 2)
        self.assertEqual(state.stage_statuses["prepare"], StageStatus.COMPLETED)
        self.assertEqual(state.stage_statuses["finish"], StageStatus.COMPLETED)
        self.assertEqual([item.name for item in state.stage_history], ["prepare", "finish"])
        self.assertEqual([item.output for item in state.agent_results], ["prepared", 2])
        self.assertEqual(state.agent_results[0].updates["value"], 2)
        self.assertEqual(state.agent_results[0].updated_keys, ("value",))
        with self.assertRaises(TypeError):
            state.agent_results[0].updates["value"] = 3
        self.assertEqual(
            [item.sequence for item in state.execution_logs],
            list(range(1, len(state.execution_logs) + 1)),
        )
        self.assertEqual(state.execution_logs[-1].event, "workflow.completed")

    def test_router_can_branch_using_committed_state(self) -> None:
        decide = FunctionAgent(
            "decider",
            lambda state: AgentResult(updates={"approved": state["score"] >= 60}),
        )
        passed = FunctionAgent("passed", lambda state: AgentResult(output="pass"))
        failed = FunctionAgent("failed", lambda state: AgentResult(output="fail"))
        engine = WorkflowEngine().add_step(
            "decide",
            decide,
            router=lambda state, result: "passed" if state["approved"] else "failed",
        )
        engine.add_step("passed", passed).add_step("failed", failed)

        self.assertEqual(engine.run({"score": 80}).output, "pass")
        self.assertEqual(engine.run({"score": 20}).output, "fail")

    def test_failure_is_wrapped_and_preserved_on_state(self) -> None:
        state = WorkflowState()

        def fail(_state: StateView) -> AgentResult:
            raise ValueError("boom")

        engine = WorkflowEngine().add_step("fail", FunctionAgent("broken", fail))

        with self.assertRaisesRegex(WorkflowExecutionError, "boom"):
            engine.run(state)

        self.assertEqual(state.status, WorkflowStatus.FAILED)
        self.assertEqual(state.current_step, "fail")
        self.assertEqual(state.error, "boom")
        self.assertEqual(state.stage_statuses["fail"], StageStatus.FAILED)
        self.assertEqual(state.stage_history[-1].error, "boom")
        self.assertEqual(state.execution_logs[-1].event, "workflow.failed")
        report = generate_workflow_report(state)
        self.assertEqual(report.outcome, WorkflowOutcome.FAILED)
        self.assertFalse(report.succeeded)
        self.assertEqual(report.error, "boom")
        self.assertEqual(report.stages[0].status, StageStatus.FAILED)

    def test_max_steps_stops_cycles(self) -> None:
        looping = FunctionAgent(
            "looping",
            lambda state: AgentResult(next_step="loop"),
        )
        engine = WorkflowEngine(max_steps=2).add_step("loop", looping)

        with self.assertRaisesRegex(WorkflowExecutionError, "最大步骤数"):
            engine.run()


class WorkflowReportIsolationTests(unittest.TestCase):
    def test_report_diagnostic_fallback_drops_string_subclass_references(self) -> None:
        class Diagnostic(str):
            def __str__(self):
                return self

        class Uncopyable:
            def __init__(self, payload):
                self.payload = payload

            def __deepcopy__(self, memo):
                raise TypeError("uncopyable")

            def __repr__(self):
                diagnostic = Diagnostic("diagnostic")
                diagnostic.payload = self.payload
                return diagnostic

        for should_fail in (False, True):
            with self.subTest(should_fail=should_fail):
                payload = {"items": []}
                value = Uncopyable(payload)
                state = WorkflowState({"payload": payload, "repair_history": (value,),
                                       "test_result": value, "report": value})
                if should_fail:
                    with self.assertRaises(WorkflowExecutionError):
                        WorkflowEngine().add_step("produce", FunctionAgent(
                            "producer", lambda snapshot: AgentResult(),
                        )).run(state)
                else:
                    state.start()
                    state.complete()
                report = generate_workflow_report(state)
                for diagnostic in (report.repair_history[0], report.test_results[0],
                                   report.final_test_result, report.final_review_result):
                    self.assertIs(type(diagnostic), str)
                    self.assertEqual(diagnostic, "diagnostic")
                    self.assertFalse(hasattr(diagnostic, "payload"))
                self.assertEqual(payload, {"items": []})
                self.assertIs(state.repair_history[0], value)
                json.dumps(report.to_dict())

    def test_report_serializer_returns_plain_diagnostic_text(self) -> None:
        class Diagnostic(str):
            def __str__(self):
                return self

        class Printable:
            def __init__(self, payload):
                self.payload = payload

            def __repr__(self):
                diagnostic = Diagnostic("printable diagnostic")
                diagnostic.payload = self.payload
                return diagnostic

        class BrokenRepr:
            def __repr__(self):
                raise ValueError("repr failed")

        for value, expected in ((Printable({"items": []}), "printable diagnostic"),
                                (BrokenRepr(), "<uncopyable BrokenRepr>")):
            with self.subTest(kind=type(value).__name__):
                run = WorkflowEngine().add_step("produce", FunctionAgent(
                    "producer", lambda snapshot: AgentResult(output=value),
                )).run()
                exported = run.report.to_dict()
                diagnostic = exported["agent_results"][0]["output"]
                self.assertIs(type(diagnostic), str)
                self.assertEqual(diagnostic, expected)
                self.assertFalse(hasattr(diagnostic, "payload"))
                json.dumps(exported)

    def test_report_detaches_update_keys_and_all_result_metadata(self) -> None:
        class Label(str):
            pass

        key, stage, agent, target = (Label(value) for value in ("value", "produce", "producer", END))
        for label in (key, stage, agent, target):
            label.tags = []
        state = WorkflowState()
        state.register_stages({stage: agent})
        state.start()
        state.begin_step(stage, agent=agent)
        state.commit_agent_result(
            agent=agent, updates={key: 1}, output=key,
            updated_keys=(key,), next_step=target,
        )
        state.complete_step()
        state.complete()
        report = generate_workflow_report(state)
        record = report.agent_results[0]
        reported_key = next(iter(record.updates))
        self.assertIs(reported_key, record.updated_keys[0])
        self.assertIs(reported_key, record.output)
        reported_key.tags.append("report key")
        record.stage.tags.append("report stage")
        record.agent.tags.append("report agent")
        record.next_step.tags.append("report route")
        retained = state.agent_results[0]
        for label in (next(iter(retained.updates)), retained.updated_keys[0], retained.output,
                      retained.stage, retained.agent, retained.next_step):
            self.assertEqual(label.tags, [])
        retained.updated_keys[0].tags.append("owner key")
        self.assertEqual(reported_key.tags, ["report key"])
        with self.assertRaises(TypeError):
            record.updates["other"] = 2

    def test_report_detaches_stage_and_log_metadata(self) -> None:
        class Label(str):
            pass

        stage, agent, event, message, level = (
            Label(value) for value in ("prepare", "agent", "custom", "message", "info")
        )
        for label in (stage, agent, event, message, level):
            label.tags = []
        for should_fail in (False, True):
            with self.subTest(should_fail=should_fail):
                state = WorkflowState()
                state.register_stages({stage: agent})
                state.start()
                state.begin_step(stage, agent=agent)
                state.log(event, message, level=level, stage=stage, agent=agent)
                if should_fail:
                    state.fail(ValueError("failed"))
                else:
                    state.complete_step()
                    state.complete()
                report = generate_workflow_report(state)
                reported_log = next(item for item in report.execution_logs if item.event == "custom")
                labels = (report.stages[0].name, report.stages[0].agent,
                          report.stages[0].records[0].agent, report.stage_records[0].name,
                          report.stage_records[0].agent, reported_log.stage, reported_log.agent,
                          reported_log.event, reported_log.message, reported_log.level)
                for label in labels:
                    label.tags.append("report")
                for label in (stage, agent, event, message, level):
                    self.assertEqual(label.tags, [])
                for record in (state.stage_history[0], state.stages[stage]):
                    self.assertEqual(record.name.tags, [])
                    self.assertEqual(record.agent.tags, [])
                retained_log = next(item for item in state.execution_logs if item.event == "custom")
                self.assertEqual(retained_log.message.tags, [])
                json.dumps(report.to_dict())

    def test_report_redacts_uncopyable_runtime_metadata(self) -> None:
        class Label(str):
            def __deepcopy__(self, memo):
                raise TypeError("uncopyable label")

        stage, agent, message = (Label(value) for value in ("prepare", "agent", "message"))
        for label in (stage, agent, message):
            label.tags = []
        state = WorkflowState()
        state.register_stages({stage: agent})
        state.start()
        state.begin_step(stage, agent=agent)
        state.log("custom", message, agent=agent)
        state.fail(ValueError("failed"))
        report = generate_workflow_report(state)
        self.assertEqual(type(report.stages[0].name), str)
        self.assertEqual(type(report.stage_records[0].agent), str)
        logged = next(item for item in report.execution_logs if item.event == "custom")
        self.assertEqual(type(logged.message), str)
        self.assertEqual(report.outcome, WorkflowOutcome.FAILED)
        json.dumps(report.to_dict())

    def test_report_fallback_detaches_result_keys_and_metadata(self) -> None:
        class Label(str):
            pass

        class Uncopyable:
            def __deepcopy__(self, memo):
                raise TypeError("uncopyable report value")

        key, agent = Label("payload"), Label("producer")
        key.tags, agent.tags = [], []
        state = WorkflowState()
        state.start()
        state.begin_step("produce", agent=agent)
        state.commit_agent_result(
            agent=agent, updates={key: {}}, output=None,
            updated_keys=(key,), next_step=None,
        )
        state.complete_step()
        state.complete()
        retained = state.agent_results[0]
        # Owner-side edits can make formerly valid data uncopyable.
        retained.updates[key]["unsafe"] = Uncopyable()
        report = generate_workflow_report(state)
        reported = report.agent_results[0]
        reported_key = next(iter(reported.updates))
        reported_key.tags.append("report key")
        reported.updated_keys[0].tags.append("report metadata")
        reported.agent.tags.append("report agent")
        self.assertEqual(next(iter(retained.updates)).tags, [])
        self.assertEqual(retained.updated_keys[0].tags, [])
        self.assertEqual(retained.agent.tags, [])
        self.assertIsInstance(reported.updates[reported_key], str)
        self.assertIsInstance(reported.updated_keys, tuple)
        with self.assertRaises(TypeError):
            reported.updates["other"] = 1
        json.dumps(report.to_dict())

    def test_report_redacts_bound_method_holding_weakref(self) -> None:
        class Receiver:
            def __init__(self, target: object) -> None:
                self.reference = weakref.ref(target)

            def read(self):
                return self.reference()

        original = ReviewReport(summary="original", issues=[])
        receiver = Receiver(original)
        callback = receiver.read
        state = WorkflowState({"repair_history": (callback,)})
        state.start()
        state.complete()

        report = generate_workflow_report(state)

        self.assertIsInstance(report.repair_history[0], str)
        self.assertIs(state.repair_history[0], callback)
        self.assertEqual(original.summary, "original")
        json.dumps(report.to_dict())

    def test_report_replaces_weakref_data_with_immutable_diagnostic(self) -> None:
        original = ReviewReport(summary="original", issues=[])
        reference = weakref.ref(original)
        state = WorkflowState({"repair_history": (reference,)})
        state.start()
        state.complete()

        report = generate_workflow_report(state)

        self.assertIsInstance(report.repair_history[0], str)
        self.assertIs(state.repair_history[0], reference)
        self.assertEqual(original.summary, "original")
        json.dumps(report.to_dict())

    def test_cyclic_outputs_serialize_in_completed_and_failed_reports(self) -> None:
        def mappingproxy_cycle():
            backing = {}
            proxy = MappingProxyType(backing)
            backing["self"] = proxy
            return proxy

        def dict_cycle():
            value = {}
            value["self"] = value
            return value

        def list_cycle():
            value = []
            value.append(value)
            return value

        @dataclass
        class Node:
            child: object = None

        def dataclass_cycle():
            node = Node()
            node.child = node
            return node

        def fail(state: StateView) -> AgentResult:
            raise ValueError("second step failed")

        for factory in (mappingproxy_cycle, dict_cycle, list_cycle, dataclass_cycle):
            for should_fail in (False, True):
                with self.subTest(factory=factory.__name__, should_fail=should_fail):
                    output = factory()
                    engine = WorkflowEngine().add_step(
                        "produce", FunctionAgent("producer", lambda state: AgentResult(output=output)),
                        next_step="fail" if should_fail else END,
                    )
                    if should_fail:
                        engine.add_step("fail", FunctionAgent("broken", fail))
                        with self.assertRaises(WorkflowExecutionError) as caught:
                            engine.run()
                        state = caught.exception.state
                    else:
                        state = engine.run().state

                    payload = generate_workflow_report(state).to_dict()
                    serialized = json.dumps(payload)
                    self.assertIn("<circular reference:", serialized)
                    self.assertEqual(payload["workflow_status"], "failed" if should_fail else "completed")
                    self.assertEqual(payload["outcome"], "failed" if should_fail else "succeeded")

    def test_report_serialization_repeats_shared_values_without_cycle_markers(self) -> None:
        shared = {"items": ["original"]}
        output = {"left": shared, "right": shared}
        run = WorkflowEngine().add_step(
            "produce", FunctionAgent("producer", lambda state: AgentResult(output=output)),
        ).run()
        report = run.report

        payload = report.to_dict()["agent_results"][0]["output"]

        self.assertEqual(payload, {"left": {"items": ["original"]}, "right": {"items": ["original"]}})
        payload["left"]["items"].append("export")
        self.assertEqual(payload["right"]["items"], ["original"])
        self.assertEqual(report.agent_results[0].output["left"]["items"], ["original"])
        self.assertEqual(run.state.agent_results[0].output["left"]["items"], ["original"])

    def test_cyclic_model_output_uses_serializable_diagnostic(self) -> None:
        class Record(BaseModel):
            payload: object = None

        output = Record()
        output.payload = output
        run = WorkflowEngine().add_step(
            "produce", FunctionAgent("producer", lambda state: AgentResult(output=output)),
        ).run()

        payload = run.report.to_dict()

        self.assertEqual(payload["agent_results"][0]["output"], "<unserializable Record>")
        json.dumps(payload)
        self.assertIs(output.payload, output)

    def test_mappingproxy_report_output_and_state_are_independent(self) -> None:
        producer = FunctionAgent("producer", lambda state: AgentResult(
            output=MappingProxyType({"items": ["original"]}),
        ))
        run = WorkflowEngine().add_step("produce", producer).run()
        report = run.report
        reported = report.agent_results[0].output
        retained = run.state.agent_results[0].output

        self.assertIsInstance(reported, MappingProxyType)
        self.assertIsNot(reported, retained)
        reported["items"].append("report")
        self.assertEqual(retained["items"], ["original"])
        retained["items"].append("state")
        self.assertEqual(reported["items"], ["original", "report"])
        with self.assertRaises(TypeError):
            reported["other"] = True
        self.assertEqual(
            report.to_dict()["agent_results"][0]["output"],
            {"items": ["original", "report"]},
        )

    def test_report_detaches_proxies_in_updates_and_runtime_histories(self) -> None:
        proxy = MappingProxyType({"items": ["original"]})
        state = WorkflowState()
        state.register_stages({"prepare": "prepare"})
        state.start()
        state.begin_step("prepare", agent="prepare")
        state.apply({"payload": proxy, "test_result": proxy,
                     "repair_history": ({"view": proxy},)}, source="prepare")
        state.record_agent_result(
            agent="prepare", updates={"payload": proxy}, output={"view": proxy},
            updated_keys=("payload",), next_step=None,
        )
        state.complete_step()
        state.complete()
        report = generate_workflow_report(state)

        report.agent_results[0].updates["payload"]["items"].append("report updates")
        report.agent_results[0].output["view"]["items"].append("report output")
        report.test_results[0]["items"].append("report tests")
        report.repair_history[0]["view"]["items"].append("report repair")

        self.assertEqual(state.get("payload")["items"], ["original"])
        self.assertEqual(state.agent_results[0].updates["payload"]["items"], ["original"])
        self.assertEqual(state.agent_results[0].output["view"]["items"], ["original"])
        self.assertEqual(state.test_results[0]["items"], ["original"])
        self.assertEqual(state.repair_history[0]["view"]["items"], ["original"])

    def test_uncopyable_report_data_uses_immutable_diagnostics(self) -> None:
        class Uncopyable:
            def __deepcopy__(self, memo):
                raise TypeError("not copyable")

            def __repr__(self):
                return "<uncopyable diagnostic>"

        original = Uncopyable()
        state = WorkflowState({"repair_history": (original,)})
        state.start()
        state.complete()

        report = generate_workflow_report(state)

        self.assertIsInstance(report.repair_history, tuple)
        self.assertIsInstance(report.repair_history[0], str)
        self.assertIn("uncopyable diagnostic", report.repair_history[0])
        self.assertIs(state.repair_history[0], original)
        self.assertIn("uncopyable diagnostic", report.to_dict()["repair_history"][0])


class CodeReviewWorkflowTests(unittest.TestCase):
    def test_code_review_agent_implements_workflow_agent_interface(self) -> None:
        self.assertTrue(all(
            issubclass(agent_type, BaseAgent)
            for agent_type in (CodingAgent, ReviewAgent, FixAgent, WorkflowTestAgent)
        ))
        self.assertTrue(issubclass(CodeReviewAgent, ReviewAgent))
        with self.assertRaisesRegex(TypeError, "只能接收 StateView"):
            CodingAgent(object()).execute(WorkflowState())

        class InvalidResultAgent(BaseAgent):
            name = "invalid-result"

            def run(self, state: StateView) -> AgentResult:
                return None

        with self.assertRaisesRegex(TypeError, "必须返回 AgentResult"):
            InvalidResultAgent().execute(WorkflowState().snapshot())
        workflow = create_review_workflow(CodeReviewAgent(MockReviewer()))

        run = workflow.run({"code": "value = 10 / 0", "task": "全面审查"})

        self.assertEqual(run.state.status, WorkflowStatus.COMPLETED)
        self.assertEqual(run.output, run.state.get("report"))
        self.assertEqual(len(run.state.get("tool_findings")), 1)
        self.assertEqual(run.state.get("tool_findings")[0].rule_id, "division-by-zero")
        self.assertEqual(run.report.outcome, WorkflowOutcome.UNRESOLVED)
        self.assertEqual(run.report.final_review_result, run.state.get("report"))


if __name__ == "__main__":
    unittest.main()
