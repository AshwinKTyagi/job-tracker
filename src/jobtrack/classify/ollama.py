"""M7 — local-LLM classifier backed by Ollama.

Implements the reproducibility contract in CONTRACTS.md §10. The rules engine is excellent
at sorting mail into event types and poor at pulling a company or a title out of prose; this
backend is the other way round, so it runs as the *primary* and leaves rules as the
low-confidence fallback::

    CompositeClassifier(OllamaClassifier(model), RulesClassifier(), min_confidence=0.60)

``classify`` never raises. A stopped daemon, a timeout, or a schema-invalid response all
return a confidence-0.0 sentinel, which the composite replaces with the rules answer — so
losing Ollama degrades quality without breaking ``sync``.

Determinism (I2) rests on four things: ``temperature=0`` with a fixed seed, grammar-
constrained decoding via Ollama's ``format`` parameter, a model pinned by digest and
quantization rather than tag, and an on-disk response cache. The cache is what makes it hold
in practice — a cached message is byte-identical forever, regardless of what the daemon does.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final

from jobtrack.classify.normalize import normalize_company
from jobtrack.classify.rules import detect_ats
from jobtrack.errors import ClassificationError
from jobtrack.models import Classification, EventType, RawMessage

logger = logging.getLogger(__name__)

#: The versioned prompt. Its SHA-256 is part of ``classifier_version``, so editing this file
#: is a version bump and old rows stay attributable to the prompt that produced them.
PROMPT_PATH: Final[Path] = Path(__file__).parent / "prompts" / "classify_v1.txt"

DEFAULT_HOST: Final[str] = "http://localhost:11434"
DEFAULT_SEED: Final[int] = 42
DEFAULT_TIMEOUT_SECONDS: Final[float] = 180.0
DEFAULT_NUM_PREDICT: Final[int] = 256

#: Body text beyond this is dropped. ATS mail puts the verdict up top and pads the rest with
#: boilerplate, and a shorter prompt is a faster and more reliable one.
MAX_BODY_CHARS: Final[int] = 2500

#: Confidence rubric. Fixed and additive, mirroring classify/confidence.py in spirit: the
#: model does not emit a calibrated probability, so the score reflects how much it actually
#: committed to. A result below the composite's threshold hands the message to the rules.
CONFIDENCE_TYPED: Final[float] = 0.70
"""Base score for a schema-valid, non-UNKNOWN verdict on a real job email."""
CONFIDENCE_COMPANY_BONUS: Final[float] = 0.15
CONFIDENCE_ROLE_BONUS: Final[float] = 0.10
CONFIDENCE_UNKNOWN: Final[float] = 0.30
"""An explicit UNKNOWN is deliberately low so the rules get a chance to disagree."""
CONFIDENCE_FAILED: Final[float] = 0.0
"""The sentinel: the daemon or the response failed, so the composite must prefer rules."""

#: The JSON schema handed to Ollama as ``format``. Grammar-constrained decoding makes schema
#: compliance a property of the sampler rather than a hope about the prompt, and the enum is
#: what stops the model inventing an EventType.
RESPONSE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "is_job_application_email": {"type": "boolean"},
        "event_type": {"type": "string", "enum": [member.value for member in EventType]},
        "company": {"type": ["string", "null"]},
        "role": {"type": ["string", "null"]},
        "location": {"type": ["string", "null"]},
    },
    "required": ["is_job_application_email", "event_type", "company", "role", "location"],
}

#: Values a model emits when it means "nothing here". Ollama's JSON mode will happily return
#: the *string* "null" for a nullable field, which would otherwise become a company name.
_NULLISH: Final[frozenset[str]] = frozenset({"", "null", "none", "n/a", "na", "unknown", "-"})

#: A transport takes (url, payload) and returns the decoded JSON body. Injected so tests run
#: without a daemon — the same seam ``GmailSource`` uses for the Gmail API.
Transport = Callable[[str, dict[str, Any]], dict[str, Any]]


def _http_transport(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    """Call Ollama and decode the reply.

    An empty payload means GET, which is what ``/api/tags`` requires — POSTing to it
    returns 405 and the model digest silently fails to resolve. A non-empty payload is
    POSTed as JSON.

    Args:
        url: Full endpoint URL.
        payload: Request body, JSON-encoded by this function. Empty means GET.
        timeout: Socket timeout in seconds.

    Returns:
        The decoded response body.

    Raises:
        ClassificationError: the daemon was unreachable, timed out, or sent invalid JSON.
    """
    data = json.dumps(payload).encode("utf-8") if payload else None
    headers = {"Content-Type": "application/json"} if payload else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except (urllib.error.URLError, OSError) as exc:
        raise ClassificationError(f"ollama at {url} is unreachable: {exc}") from exc
    try:
        decoded: dict[str, Any] = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ClassificationError(f"ollama at {url} returned invalid JSON: {exc}") from exc
    return decoded


def _clean(value: object) -> str | None:
    """Normalize a model-supplied string, folding placeholder text to None.

    Trailing periods are deliberately kept: "Acme Robotics, Inc." is a legal suffix, not
    sentence punctuation, and the display form stays verbatim (I8) because matching uses
    ``normalize_company`` rather than this string.
    """
    if not isinstance(value, str):
        return None
    text = " ".join(value.split()).strip(" ,;:-")
    return None if text.casefold().rstrip(".") in _NULLISH else text


def _prompt_sha() -> str:
    """SHA-256 of the prompt template on disk.

    Returns:
        The hex digest.

    Raises:
        ClassificationError: the template is missing.
    """
    try:
        return hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
    except OSError as exc:
        raise ClassificationError(f"prompt template missing at {PROMPT_PATH}: {exc}") from exc


class OllamaClassifier:
    """Local-LLM classifier for job-application mail.

    See the module docstring for the reproducibility contract. Construction resolves the
    model tag to a digest and quantization level, both of which go into ``version`` — so a
    model upgrade, an Ollama upgrade, or a prompt edit all invalidate old attributions
    rather than silently changing what stored rows mean.
    """

    name = "ollama"
    version: str
    """SHA-256 prefix of the prompt template, the resolved model digest, and the
    quantization level. Fixed at construction — a plain attribute rather than the property
    CONTRACTS.md §10 sketched, because the ``Classifier`` Protocol declares it as a settable
    variable and the value never changes after ``__init__`` anyway."""

    def __init__(
        self,
        model: str,
        *,
        host: str = DEFAULT_HOST,
        seed: int = DEFAULT_SEED,
        think: bool = False,
        cache: Path | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        transport: Transport | None = None,
    ) -> None:
        """Wire a classifier to a local Ollama daemon.

        Args:
            model: Ollama tag, resolved to a digest at construction and recorded in
                ``version``. No default — config.toml or the environment names it.
            host: Base URL of the daemon.
            seed: Sampling seed. Fixed for reproducibility.
            think: Must stay False. Exposed only so an eval harness can measure the cost of
                reasoning traces before rejecting them; they destroy determinism.
            cache: Directory for cached responses, or None to disable caching.
            timeout: Socket timeout in seconds. A cold model load can take a minute.
            transport: Injected request function. Tests pass a fake here — this parameter is
                the reason M7 is testable without a daemon.

        Raises:
            ClassificationError: the prompt template is missing.
        """
        self._model = model
        self._host = host.rstrip("/")
        self._seed = seed
        self._think = think
        self._cache_dir = cache
        self._timeout = timeout
        self._transport: Transport = transport or (
            lambda url, payload: _http_transport(url, payload, timeout=timeout)
        )
        self._system = PROMPT_PATH.read_text(encoding="utf-8")
        self._prompt_sha = _prompt_sha()
        self._fingerprint = self._resolve_model()
        self.version = f"{self._prompt_sha[:12]}+{self._fingerprint}"

    def _resolve_model(self) -> str:
        """Resolve the model tag to ``digest+quantization``, or a marker when unavailable.

        A daemon that cannot be reached at construction is not fatal — every ``classify``
        call degrades to the sentinel anyway — but the version string records that the
        pinning is unverified rather than pretending otherwise.
        """
        try:
            listing = self._transport(f"{self._host}/api/tags", {})
        except ClassificationError:
            logger.warning("could not reach ollama at %s to pin %s", self._host, self._model)
            return "unresolved"

        for entry in listing.get("models", []):
            if entry.get("name") != self._model:
                continue
            digest = str(entry.get("digest", ""))[:12]
            quant = str(entry.get("details", {}).get("quantization_level", "?"))
            return f"{digest}+{quant}"
        logger.warning("model %s is not present on the daemon at %s", self._model, self._host)
        return "unresolved"

    def _cache_path(self, message_id: str) -> Path | None:
        """Location of the cached response for a message, or None when caching is off.

        Keyed on (prompt_sha, model fingerprint, message_id) per CONTRACTS §10, so editing
        the prompt or changing the model misses the cache instead of returning a stale
        answer attributed to the wrong version.
        """
        if self._cache_dir is None:
            return None
        key = f"{self._prompt_sha}|{self._fingerprint}|{message_id}"
        return self._cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"

    def _read_cache(self, message_id: str) -> dict[str, Any] | None:
        """Return the cached raw response for a message, or None on any miss."""
        path = self._cache_path(message_id)
        if path is None or not path.is_file():
            return None
        try:
            cached: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("ignoring unreadable cache entry %s: %s", path, exc)
            return None
        return cached

    def _write_cache(self, message_id: str, payload: dict[str, Any]) -> None:
        """Persist a raw response. A cache that cannot be written is logged, never fatal."""
        path = self._cache_path(message_id)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            logger.debug("could not write cache entry %s: %s", path, exc)

    def _ask(self, message: RawMessage) -> dict[str, Any]:
        """Send one message to the daemon and return the parsed model output.

        Raises:
            ClassificationError: the daemon failed, or the reply was not schema-shaped.
        """
        user = (
            f"From: {message.from_email}\n"
            f"Subject: {message.subject}\n\n"
            f"{message.body_text[:MAX_BODY_CHARS]}"
        )
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "format": RESPONSE_SCHEMA,
            "think": self._think,
            "options": {
                "temperature": 0,
                "top_p": 1,
                "seed": self._seed,
                "num_predict": DEFAULT_NUM_PREDICT,
            },
        }
        response = self._transport(f"{self._host}/api/chat", payload)
        content = response.get("message", {}).get("content")
        if not isinstance(content, str):
            raise ClassificationError("ollama response had no message content")
        try:
            parsed: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ClassificationError(f"ollama returned non-JSON content: {exc}") from exc
        if not isinstance(parsed.get("event_type"), str):
            raise ClassificationError("ollama response is missing event_type")
        return parsed

    def _sentinel(self, message: RawMessage, reason: str) -> Classification:
        """A zero-confidence result, so the composite prefers the rules answer."""
        ats, _ = detect_ats(message)
        return Classification(
            message_id=message.message_id,
            event_type=EventType.UNKNOWN,
            ats=ats,
            confidence=CONFIDENCE_FAILED,
            needs_review=True,
            evidence=[f"ollama.unavailable.{reason}"],
            classifier_name=self.name,
            classifier_version=self.version,
        )

    def _build(self, message: RawMessage, parsed: dict[str, Any]) -> Classification:
        """Turn a validated model response into a Classification.

        Raises:
            ClassificationError: the model named an event type outside the enum.
        """
        raw_type = str(parsed["event_type"]).strip().lower()
        try:
            event_type = EventType(raw_type)
        except ValueError as exc:
            raise ClassificationError(f"ollama invented event type {raw_type!r}") from exc

        # A false triage gate means "not job mail" — drop the fields even if the model
        # filled them in anyway, which it does for university and banking notices.
        is_job_mail = bool(parsed.get("is_job_application_email", True))
        if not is_job_mail:
            event_type = EventType.UNKNOWN

        company = _clean(parsed.get("company")) if is_job_mail else None
        role = _clean(parsed.get("role")) if is_job_mail else None
        location = _clean(parsed.get("location")) if is_job_mail else None

        # ATS detection stays with the rules: it reads the sender and headers, which is
        # deterministic and not something the model can see more clearly than a regex.
        ats, ats_evidence = detect_ats(message)

        if event_type is EventType.UNKNOWN:
            confidence = CONFIDENCE_UNKNOWN
        else:
            confidence = CONFIDENCE_TYPED
            if company is not None:
                confidence += CONFIDENCE_COMPANY_BONUS
            if role is not None:
                confidence += CONFIDENCE_ROLE_BONUS

        evidence = [f"ollama.type.{event_type.value}"]
        if not is_job_mail:
            evidence.append("ollama.triage.not_job_mail")
        if company is not None:
            evidence.append("ollama.field.company")
        if role is not None:
            evidence.append("ollama.field.role")
        evidence.extend(ats_evidence)

        return Classification(
            message_id=message.message_id,
            event_type=event_type,
            company=company,
            company_key=normalize_company(company),
            role=role,
            location=location,
            ats=ats,
            confidence=min(confidence, 1.0),
            needs_review=event_type is EventType.UNKNOWN,
            evidence=evidence,
            classifier_name=self.name,
            classifier_version=self.version,
        )

    def classify(self, message: RawMessage) -> Classification:
        """Classify one message. Never raises.

        A cached response is returned verbatim, which is what makes ``reclassify`` free and
        byte-identical. Any failure — daemon down, timeout, malformed or schema-invalid
        response, an invented event type — degrades to a zero-confidence sentinel so the
        composite falls through to the rules engine.

        Args:
            message: The normalized email to classify.

        Returns:
            The classification, or the sentinel when the model could not produce one.
        """
        cached = self._read_cache(message.message_id)
        if cached is not None:
            try:
                return self._build(message, cached)
            except ClassificationError as exc:
                logger.debug("discarding unusable cache entry for %s: %s", message.message_id, exc)

        try:
            parsed = self._ask(message)
        except ClassificationError as exc:
            logger.warning("ollama could not classify %s: %s", message.message_id, exc)
            return self._sentinel(message, "transport")

        try:
            result = self._build(message, parsed)
        except ClassificationError as exc:
            logger.warning(
                "ollama returned an unusable verdict for %s: %s", message.message_id, exc
            )
            return self._sentinel(message, "schema")

        self._write_cache(message.message_id, parsed)
        return result

    def classify_batch(self, messages: Sequence[RawMessage]) -> list[Classification]:
        """Classify a sequence, preserving input order.

        Args:
            messages: Messages to classify.

        Returns:
            One Classification per input message, in the same order.
        """
        return [self.classify(message) for message in messages]


__all__ = [
    "DEFAULT_HOST",
    "PROMPT_PATH",
    "RESPONSE_SCHEMA",
    "OllamaClassifier",
    "Transport",
]
