# Ultralytics AGPL-3.0 License - https://ultralytics.com/license

"""Consumer contract registry and multi-output producer specifications for YOLOMM structured pruning.

This module defines the channel semantics and input-output contract rules for all
multi-input consumer modules and multi-output producer modules used in YOLOMM
models.  These contracts guide the pruning strategy by specifying:

- How many inputs a consumer expects (and whether they can be independently pruned).
- How the output channel count relates to the input channel counts.
- Which output slot(s) a downstream consumer is allowed to read from a multi-output
  producer (FCM, MultiHeadCrossAttention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ultralytics.nn.mm.pruning.graph import PruneNode

# ------------------------------------------------------------------
# Type aliases
# ------------------------------------------------------------------

ContractMode = Literal[
    "concat_like",
    "detect_head",
    "equal_width_left_output",
    "left_output",
    "declared_output",
]

OutputLayout = Literal[
    "single",
    "tuple_same_width",
    "tuple_declared_width",
]

# ------------------------------------------------------------------
# Exception types
# ------------------------------------------------------------------


class UnsupportedMultiInputConsumerError(RuntimeError):
    """Raised when a multi-input consumer module has no registered contract."""

    pass


class UnsupportedMultiOutputProducerError(RuntimeError):
    """Raised when a multi-output producer module has no registered output spec."""

    pass


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumerContract:
    """Contract specification for a multi-input consumer module.

    Attributes:
        module_name: Canonical class name of the module (e.g. "Concat", "FCM").
        min_inputs: Minimum number of input edges the module requires.
        mode: How the module computes its output channel count from its inputs.
        allow_independent_inputs: Whether each input edge can be pruned independently
            (True) or whether the inputs must be co-pruned to keep equal width
            (False).  Set to True for concat-like modules where each branch is
            independent, and for detect heads where each feature level is independent.
            Set to False for fusion modules that require both inputs to have the same
            width (e.g. FeatureFusion, FCM, SEFN, etc.).
    """

    module_name: str
    min_inputs: int
    mode: ContractMode
    allow_independent_inputs: bool


@dataclass(frozen=True)
class ProducerOutputSpec:
    """Specification for a module that produces multiple named outputs.

    Attributes:
        module_name: Canonical class name of the producer module.
        output_slots: Total number of output slots (e.g. 2 for FCM which returns
            a (main, aux) tuple).
        layout: How the output slots are laid out.  "single" means only one
            meaningful slot; "tuple_same_width" means all slots have the same
            channel count (used by FCM and MultiHeadCrossAttention).
    """

    module_name: str
    output_slots: int
    layout: OutputLayout


# ------------------------------------------------------------------
# Consumer contract registry
#
# Covers all multi-input module types that actually appear in
# ultralytics/cfg/models/mm/**/*.yaml configurations.
#
# ContractMode semantics:
#   concat_like          - output channels = sum of all input channels
#                          (e.g. Concat: c2 = sum(ch[x] for x in f))
#   detect_head          - multi-level feature pyramid head (Detect, v8Detect, ...)
#                          outputs are not channel-prunable
#   equal_width_left_output - both inputs must have equal channel count;
#                             output channels follow the LEFT (f[0]) input
#                             (e.g. FeatureFusion, FCM, SEFN, MSC, ...)
#   left_output          - same as equal_width_left_output (kept for clarity)
#   declared_output      - output channels are explicitly declared in YAML args
#                          (not derived from input channels)
# ------------------------------------------------------------------

CONSUMER_CONTRACTS: dict[str, ConsumerContract] = {
    # ----- Routing / concatenation -----
    "Concat": ConsumerContract(
        module_name="Concat",
        min_inputs=2,
        mode="concat_like",
        allow_independent_inputs=True,
    ),
    # ----- Detection heads -----
    "Detect": ConsumerContract(
        module_name="Detect",
        min_inputs=1,
        mode="detect_head",
        allow_independent_inputs=True,
    ),
    # ----- Equal-width two-input fusion modules (left output) -----
    # parse_model sets c2 = c_left for all of these.
    "FeatureFusion": ConsumerContract(
        module_name="FeatureFusion",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "FCM": ConsumerContract(
        module_name="FCM",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "FCMFeatureFusion": ConsumerContract(
        module_name="FCMFeatureFusion",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "ConvMixFusion": ConsumerContract(
        module_name="ConvMixFusion",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "ChannelGate": ConsumerContract(
        module_name="ChannelGate",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "CAM": ConsumerContract(
        module_name="CAM",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "SEFN": ConsumerContract(
        module_name="SEFN",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "FusionConvMSAA": ConsumerContract(
        module_name="FusionConvMSAA",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "MSC": ConsumerContract(
        module_name="MSC",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "SpatialDependencyPerception": ConsumerContract(
        module_name="SpatialDependencyPerception",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "FDFEF": ConsumerContract(
        module_name="FDFEF",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "DEA": ConsumerContract(
        module_name="DEA",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "MJRNet": ConsumerContract(
        module_name="MJRNet",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "MSIA": ConsumerContract(
        module_name="MSIA",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    "RFF": ConsumerContract(
        module_name="RFF",
        min_inputs=2,
        mode="equal_width_left_output",
        allow_independent_inputs=False,
    ),
    # ----- Multi-input fusion with gating / concatenation output -----
    "MCFGatedFusion": ConsumerContract(
        module_name="MCFGatedFusion",
        min_inputs=2,
        mode="declared_output",
        allow_independent_inputs=True,
    ),
    "CrossTransformerFusion": ConsumerContract(
        module_name="CrossTransformerFusion",
        min_inputs=2,
        mode="declared_output",
        allow_independent_inputs=False,
    ),
    "MultiHeadCrossAttention": ConsumerContract(
        module_name="MultiHeadCrossAttention",
        min_inputs=2,
        mode="declared_output",
        allow_independent_inputs=False,
    ),
}


# ------------------------------------------------------------------
# Multi-output producer specifications
#
# Modules that produce multiple named outputs (tuple) instead of a
# single tensor.  Each output slot is referenced by a consumer edge
# via the EdgeRef.output_slot field.
#
# Note: Index is a routing operator that selects a specific slot from
# a multi-output producer.  It is NOT a consumer in the sense of
# CONSUMER_CONTRACTS because it does not independently decide channel
# counts -- it just exposes one of the producer's slots.
# ------------------------------------------------------------------

PRODUCER_OUTPUT_SPECS: dict[str, ProducerOutputSpec] = {
    # FCM returns (main_features, aux_features) where both have equal channel count.
    "FCM": ProducerOutputSpec(
        module_name="FCM",
        output_slots=2,
        layout="tuple_same_width",
    ),
    # MultiHeadCrossAttention returns (self-attn_output, cross-attn_output)
    # with equal channel count per slot.
    "MultiHeadCrossAttention": ProducerOutputSpec(
        module_name="MultiHeadCrossAttention",
        output_slots=2,
        layout="tuple_same_width",
    ),
    "ChannelTransformer": ProducerOutputSpec(
        module_name="ChannelTransformer",
        output_slots=0,
        layout="tuple_declared_width",
    ),
}


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


def require_contract(node: PruneNode) -> ConsumerContract:
    """Return the consumer contract for a node.

    Args:
        node: A PruneNode representing a multi-input layer.

    Returns:
        The registered ConsumerContract for the node's type.

    Raises:
        UnsupportedMultiInputConsumerError: If no contract is registered for
            the node's type_name.
    """
    contract = CONSUMER_CONTRACTS.get(node.type_name)
    if contract is None:
        raise UnsupportedMultiInputConsumerError(
            f"No consumer contract registered for multi-input module '{node.type_name}' "
            f"(layer {node.idx}).  Add a CONSUMER_CONTRACTS entry or mark the module "
            "as a single-input layer."
        )
    return contract


def require_output_spec(node: PruneNode) -> ProducerOutputSpec:
    """Return the output specification for a multi-output producer node.

    Args:
        node: A PruneNode representing a multi-output producer layer.

    Returns:
        The registered ProducerOutputSpec for the node's type.

    Raises:
        UnsupportedMultiOutputProducerError: If no spec is registered for
            the node's type_name.
    """
    spec = PRODUCER_OUTPUT_SPECS.get(node.type_name)
    if spec is None:
        raise UnsupportedMultiOutputProducerError(
            f"No output spec registered for multi-output producer '{node.type_name}' "
            f"(layer {node.idx}).  Add a PRODUCER_OUTPUT_SPECS entry."
        )
    return spec
