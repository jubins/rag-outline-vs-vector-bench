"""Prometheus metrics and OpenTelemetry spans for each pipeline step."""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from opentelemetry import trace
from prometheus_client import CollectorRegistry, Histogram

REQUEST_BUCKETS = (0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 30)
STEP_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5)
ROUND_BUCKETS = (1, 2, 3, 4, 5, 8)


class Telemetry:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.request_seconds = Histogram(
            "rag_request_seconds",
            "End-to-end request time",
            ["design"],
            buckets=REQUEST_BUCKETS,
            registry=self.registry,
        )
        self.step_seconds = Histogram(
            "rag_step_seconds",
            "Time spent in one pipeline step",
            ["design", "step"],
            buckets=STEP_BUCKETS,
            registry=self.registry,
        )
        self.retrieval_rounds = Histogram(
            "rag_retrieval_rounds",
            "Retrieval attempts per request",
            ["design"],
            buckets=ROUND_BUCKETS,
            registry=self.registry,
        )
        self.tracer = trace.get_tracer("rag-bench")

    @contextmanager
    def step(self, design: str, name: str) -> Iterator[None]:
        start = time.perf_counter()
        with self.tracer.start_as_current_span(name):
            try:
                yield
            finally:
                self.step_seconds.labels(design, name).observe(time.perf_counter() - start)


def setup_tracing(service_name: str) -> None:
    """Send spans to an OTLP endpoint when OTEL_EXPORTER_OTLP_ENDPOINT is set."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
    )
    trace.set_tracer_provider(provider)
