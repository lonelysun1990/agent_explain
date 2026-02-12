"""
Abstract interface for use-case-specific data required by the Graph RAG strategy.

Implement this protocol for each use case (e.g. staffing) so graph_rag.py stays generic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GraphRAGConfig(Protocol):
    """
    Use-case-specific configuration for building the hierarchical graph.

    Implementations provide formulation docs, entity names, constraint-to-entity
    mapping, objective-to-variable mapping, and variable entity scope (for L2 grouping).
    """

    def get_formulation_docs(self) -> str:
        """Full formulation documentation (## VARIABLES, ## CONSTRAINTS, ## OBJECTIVES)."""
        ...

    def get_entity_name(self, entity_type: str, entity_id: int) -> str:
        """
        Human-readable name for an entity (e.g. employee 0 -> "Josh", project 1 -> "PSO v8").
        entity_type is e.g. "employee" or "project".
        """
        ...

    def get_entity_count(self, entity_type: str) -> int:
        """Number of entities of this type (for iterating L2 nodes)."""
        ...

    def constraint_name_to_entity(
        self, constraint_type: str, constraint_instance_name: str
    ) -> tuple[str, int] | None:
        """
        Map (constraint type, LP constraint name) to (entity_type, entity_id) for grouping.
        E.g. ("demand_balance", "demand_balance_project_0_week_0") -> ("project", 0).
        Return None if this constraint type is not grouped by entity.
        """
        ...

    def get_objective_variable_map(self) -> dict[str, set[str]]:
        """
        Map objective term name -> set of variable family names that appear in that objective.
        E.g. {"cost_of_missing_demand": {"d_miss"}, "idle_time": {"x_idle"}}.
        """
        ...

    def get_variable_entity_scope(self, variable_family: str) -> str:
        """
        Which entity type this variable is primarily indexed by, for L2 grouping.
        Returns "employee" or "project" (or another entity_type string).
        E.g. "d_miss" -> "project", "x" -> "employee".
        """
        ...

    def get_constraint_type_prefixes(self) -> list[tuple[str, str]]:
        """
        LP constraint name prefix -> semantic constraint type, for parsing.
        E.g. [("demand_balance_project_", "demand_balance"), ...].
        Order matters: first match wins.
        """
        ...
