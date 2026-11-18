"""Token cost estimates. Prices in USD per million tokens (Anthropic API list prices)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_read: float
    cache_write: float  # 5-minute cache writes


PRICES: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(4.00, 20.00, 0.20, 5.00),
    "claude-opus-5": ModelPrice(5.00, 25.00, 0.50, 6.25),
    "claude-opus-4-8": ModelPrice(5.00, 25.00, 0.50, 6.25),
    "claude-sonnet-5-5": ModelPrice(2.00, 10.00, 0.10, 2.50),
    "claude-sonnet-5": ModelPrice(2.00, 10.00, 0.20, 2.50),
    "claude-haiku-5-5": ModelPrice(0.10, 0.50, 0.01, 0.125),
    "extractive-v1": ModelPrice(0.0, 0.0, 0.0, 0.0),  # offline, no API
}


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_api(cls, usage: object) -> "Usage":
        return cls(
            **{
                field: int(getattr(usage, field, 0) or 0)
                for field in cls.__dataclass_fields__  # type: ignore[attr-defined]
            }
        )


def estimate_cost_usd(model: str, usage: Usage) -> float | None:
    """Estimated cost of one request, or None for a model without a known price."""
    price = PRICES.get(model)
    if price is None:
        return None
    total = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_read_input_tokens * price.cache_read
        + usage.cache_creation_input_tokens * price.cache_write
    )
    return round(total / 1_000_000, 6)
