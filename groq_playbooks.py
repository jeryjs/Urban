"""Groq structured output integration for Urban Farming dashboard."""
from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List

from groq import Groq


SUPPORTED_MODEL = "openai/gpt-oss-120b"


class GroqPlaybookError(RuntimeError):
    """Raised when Groq orchestration fails."""


@dataclass(frozen=True)
class PlaybookRequest:
    crop: str
    organic_context: str
    variant_cap: int
    emphasize: List[str]


def _base_schema() -> Dict[str, Any]:
    """Return the canonical JSON schema used for structured outputs."""
    combo_ref_schema = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [
                    "irrigation_playbooks",
                    "nutrient_programs",
                    "lighting_profiles",
                    "resilience_protocols",
                    "commercial_blueprints",
                    "learning_modules",
                ],
            },
            "name": {"type": "string"},
        },
        "required": ["category", "name"],
        "additionalProperties": False,
    }

    def array_schema(item_schema: Dict[str, Any]) -> Dict[str, Any]:
        schema = {
            "type": "array",
            "items": item_schema,
            "minItems": 3,
            "maxItems": 12,
        }
        return schema

    return {
        "type": "object",
        "properties": {
            "meta": {
                "type": "object",
                "properties": {
                    "crop": {"type": "string"},
                    "narrative": {"type": "string"},
                    "variant_target": {"type": "integer"},
                },
                "required": ["crop", "narrative", "variant_target"],
                "additionalProperties": False,
            },
            "irrigation_playbooks": array_schema(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "trigger": {"type": "string"},
                        "sensor_proxy": {"type": "string"},
                        "actions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 3,
                            "maxItems": 8,
                        },
                        "organic_inputs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 5,
                        },
                        "impact_score": {
                            "type": "string",
                            "enum": ["water_saver", "yield_guard", "risk_control"],
                        },
                    },
                    "required": [
                        "name",
                        "trigger",
                        "sensor_proxy",
                        "actions",
                        "organic_inputs",
                        "impact_score",
                    ],
                    "additionalProperties": False,
                }
            ),
            "nutrient_programs": array_schema(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "ikr_inputs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 6,
                        },
                        "dose_pattern": {"type": "string"},
                        "microbe_focus": {"type": "string"},
                        "expected_delta": {"type": "string"},
                    },
                    "required": [
                        "name",
                        "ikr_inputs",
                        "dose_pattern",
                        "microbe_focus",
                        "expected_delta",
                    ],
                    "additionalProperties": False,
                }
            ),
            "lighting_profiles": array_schema(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "photoperiod": {"type": "string"},
                        "spectrum_notes": {"type": "string"},
                        "stage_alignment": {"type": "string"},
                        "energy_tradeoff": {"type": "string"},
                    },
                    "required": [
                        "name",
                        "photoperiod",
                        "spectrum_notes",
                        "stage_alignment",
                        "energy_tradeoff",
                    ],
                    "additionalProperties": False,
                }
            ),
            "resilience_protocols": array_schema(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "shock": {"type": "string"},
                        "buffer_steps": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 6,
                        },
                        "backup_inputs": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 4,
                        },
                        "recovery_signal": {"type": "string"},
                    },
                    "required": [
                        "name",
                        "shock",
                        "buffer_steps",
                        "backup_inputs",
                        "recovery_signal",
                    ],
                    "additionalProperties": False,
                }
            ),
            "commercial_blueprints": array_schema(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "audience": {"type": "string"},
                        "value_prop": {"type": "string"},
                        "revenue_streams": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 5,
                        },
                        "impact_metric": {"type": "string"},
                    },
                    "required": [
                        "name",
                        "audience",
                        "value_prop",
                        "revenue_streams",
                        "impact_metric",
                    ],
                    "additionalProperties": False,
                }
            ),
            "learning_modules": array_schema(
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "format": {"type": "string"},
                        "learning_goal": {"type": "string"},
                        "hands_on_elements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 2,
                            "maxItems": 6,
                        },
                        "assessment": {"type": "string"},
                    },
                    "required": [
                        "name",
                        "format",
                        "learning_goal",
                        "hands_on_elements",
                        "assessment",
                    ],
                    "additionalProperties": False,
                }
            ),
            "cross_linkages": array_schema(
                {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "rationale": {"type": "string"},
                        "linked_assets": {
                            "type": "array",
                            "items": combo_ref_schema,
                            "minItems": 2,
                            "maxItems": 5,
                        },
                    },
                    "required": ["title", "rationale", "linked_assets"],
                    "additionalProperties": False,
                }
            ),
        },
        "required": [
            "meta",
            "irrigation_playbooks",
            "nutrient_programs",
            "lighting_profiles",
            "resilience_protocols",
            "commercial_blueprints",
            "learning_modules",
            "cross_linkages",
        ],
        "additionalProperties": False,
    }


def _dynamic_schema(variant_cap: int) -> Dict[str, Any]:
    variant_cap = max(3, min(variant_cap, 12))
    schema = deepcopy(_base_schema())
    for key in [
        "irrigation_playbooks",
        "nutrient_programs",
        "lighting_profiles",
        "resilience_protocols",
        "commercial_blueprints",
        "learning_modules",
        "cross_linkages",
    ]:
        schema["properties"][key]["maxItems"] = variant_cap
    schema["properties"]["meta"]["properties"]["variant_target"] = {
        "type": "integer",
        "minimum": 3,
        "maximum": variant_cap,
    }
    return schema


def _client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise GroqPlaybookError(
            "Missing GROQ_API_KEY. Set it as an environment variable before using Groq integration."
        )
    return Groq(api_key=api_key)


def fetch_playbooks(req: PlaybookRequest) -> Dict[str, Any]:
    client = _client()
    schema = _dynamic_schema(req.variant_cap)
    emphasize_text = ", ".join(req.emphasize) if req.emphasize else "all modules"

    system_prompt = (
        "You are an agronomy+business AI architect for Indian organic urban farms. "
        "Blend Panchgavya/Jeevamrut cycles, seed-to-market storytelling, and education. "
        "Output dense, implementation-ready catalogs without any prose beyond schema fields."
    )
    user_prompt = (
        f"Crop focus: {req.crop}. Context: {req.organic_context}. "
        f"Generate upto {req.variant_cap} variants per array, prioritizing: {emphasize_text}. "
        "Ensure each entry references Indian organic inputs, urban constraints, and measurable KPIs."
    )

    response = client.chat.completions.create(
        model=SUPPORTED_MODEL,
        temperature=0.35,
        max_completion_tokens=3500,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "urban_playbooks", "schema": schema},
        },
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )

    raw = response.choices[0].message.content
    if not raw:
        raise GroqPlaybookError("Groq response missing content")
    return json.loads(raw)


__all__ = ["fetch_playbooks", "PlaybookRequest", "GroqPlaybookError", "SUPPORTED_MODEL"]
