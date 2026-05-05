"""Custom exception types for TrajPatch."""


class TrajPatchError(Exception):
    """Base exception for all TrajPatch failures."""


class ParserValidationError(TrajPatchError):
    """Raised when an LLM text response cannot be parsed or validated."""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        section: str | None = None,
        field: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.section = section
        self.field = field
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, object]:
        return {
            "message": str(self),
            "code": self.code,
            "section": self.section,
            "field": self.field,
            "details": dict(self.details),
        }


class ProviderConfigurationError(TrajPatchError):
    """Raised when a provider cannot be initialized from the config."""


class DatasetFormatError(TrajPatchError):
    """Raised when a dataset file cannot be parsed into the normalized format."""


class RetrievalError(TrajPatchError):
    """Raised when retrieval cannot be executed safely."""


class StructuredOutputError(TrajPatchError):
    """Raised when a provider-side structured output request cannot be completed safely."""

    def __init__(
        self,
        message: str,
        *,
        vendor: str | None = None,
        strategy: str | None = None,
        refusal: str | None = None,
        raw: object | None = None,
    ) -> None:
        super().__init__(message)
        self.vendor = vendor
        self.strategy = strategy
        self.refusal = refusal
        self.raw = raw
