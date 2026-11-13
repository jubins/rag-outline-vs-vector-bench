"""Design B: read a prebuilt outline, let the model pick sections, fetch them from storage."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Protocol

from bench.corpus import Section
from bench.llm import LLMClient
from bench.models import OutlineNode, Passage
from bench.telemetry import Telemetry

BUCKET = "rag-bench-trees"
TABLE = "rag-bench-outline-index"


class OutlineStore(Protocol):
    async def put_document(self, doc_id: str, sections: list[Section]) -> None: ...

    async def get_outline(self, doc_id: str) -> list[OutlineNode]: ...

    async def get_sections(self, doc_id: str, node_ids: list[str]) -> list[Passage]: ...

    async def count(self) -> int: ...


class InMemoryOutlineStore:
    def __init__(self) -> None:
        self._outlines: dict[str, list[OutlineNode]] = {}
        self._sections: dict[str, dict[str, Passage]] = {}

    async def put_document(self, doc_id: str, sections: list[Section]) -> None:
        self._outlines[doc_id] = [OutlineNode(s.id, s.title, s.summary) for s in sections]
        self._sections[doc_id] = {s.id: Passage(s.id, s.doc_id, s.text) for s in sections}

    async def get_outline(self, doc_id: str) -> list[OutlineNode]:
        return self._outlines[doc_id]

    async def get_sections(self, doc_id: str, node_ids: list[str]) -> list[Passage]:
        return [self._sections[doc_id][n] for n in node_ids if n in self._sections[doc_id]]

    async def count(self) -> int:
        return len(self._outlines)


class AwsOutlineStore:
    """Tree files in S3 and the section index in DynamoDB (LocalStack in the Compose setup)."""

    def __init__(self, endpoint_url: str, region: str = "us-east-1") -> None:
        import boto3

        kwargs = {
            "endpoint_url": endpoint_url,
            "region_name": region,
            "aws_access_key_id": "test",
            "aws_secret_access_key": "test",
        }
        self._s3 = boto3.client("s3", **kwargs)
        self._ddb = boto3.client("dynamodb", **kwargs)

    def ensure_resources(self) -> None:
        existing = {b["Name"] for b in self._s3.list_buckets()["Buckets"]}
        if BUCKET not in existing:
            self._s3.create_bucket(Bucket=BUCKET)
        if TABLE not in self._ddb.list_tables()["TableNames"]:
            self._ddb.create_table(
                TableName=TABLE,
                KeySchema=[{"AttributeName": "doc_id", "KeyType": "HASH"}],
                AttributeDefinitions=[{"AttributeName": "doc_id", "AttributeType": "S"}],
                BillingMode="PAY_PER_REQUEST",
            )
            self._ddb.get_waiter("table_exists").wait(TableName=TABLE)

    def _put_sync(self, doc_id: str, sections: list[Section]) -> None:
        tree = {s.id: s.text for s in sections}
        self._s3.put_object(
            Bucket=BUCKET, Key=f"trees/{doc_id}.json", Body=json.dumps(tree).encode()
        )
        outline = [{"id": s.id, "title": s.title, "summary": s.summary} for s in sections]
        self._ddb.put_item(
            TableName=TABLE,
            Item={"doc_id": {"S": doc_id}, "outline": {"S": json.dumps(outline)}},
        )

    def _outline_sync(self, doc_id: str) -> list[OutlineNode]:
        item = self._ddb.get_item(TableName=TABLE, Key={"doc_id": {"S": doc_id}})["Item"]
        return [OutlineNode(**n) for n in json.loads(item["outline"]["S"])]

    def _sections_sync(self, doc_id: str, node_ids: list[str]) -> list[Passage]:
        body = self._s3.get_object(Bucket=BUCKET, Key=f"trees/{doc_id}.json")["Body"].read()
        tree = json.loads(body)
        return [Passage(n, doc_id, tree[n]) for n in node_ids if n in tree]

    def _count_sync(self) -> int:
        if TABLE not in self._ddb.list_tables()["TableNames"]:
            return 0
        return int(self._ddb.scan(TableName=TABLE, Select="COUNT")["Count"])

    async def put_document(self, doc_id: str, sections: list[Section]) -> None:
        await asyncio.to_thread(self._put_sync, doc_id, sections)

    async def get_outline(self, doc_id: str) -> list[OutlineNode]:
        return await asyncio.to_thread(self._outline_sync, doc_id)

    async def get_sections(self, doc_id: str, node_ids: list[str]) -> list[Passage]:
        return await asyncio.to_thread(self._sections_sync, doc_id, node_ids)

    async def count(self) -> int:
        return await asyncio.to_thread(self._count_sync)


class OutlineCache:
    """A small time-to-live cache. It stands in for DAX, which LocalStack does not provide.

    The expiry matters in practice. When a document changes, a stale cache entry would serve
    sections from the old tree until the entry expires.
    """

    def __init__(self, store: OutlineStore, ttl_seconds: float = 60.0) -> None:
        self._store = store
        self._ttl = ttl_seconds
        self._entries: dict[str, tuple[float, list[OutlineNode]]] = {}

    async def get_outline(self, doc_id: str) -> list[OutlineNode]:
        now = time.monotonic()
        cached = self._entries.get(doc_id)
        if cached and now - cached[0] < self._ttl:
            return cached[1]
        outline = await self._store.get_outline(doc_id)
        self._entries[doc_id] = (now, outline)
        return outline

    def invalidate(self, doc_id: str) -> None:
        self._entries.pop(doc_id, None)


class OutlineRetriever:
    design = "outline"

    def __init__(
        self,
        store: OutlineStore,
        llm: LLMClient,
        telemetry: Telemetry,
        cache_ttl_seconds: float = 60.0,
    ) -> None:
        self._store = store
        self._cache = OutlineCache(store, cache_ttl_seconds)
        self._llm = llm
        self._telemetry = telemetry

    async def retrieve(self, question: str, doc_id: str) -> list[Passage]:
        with self._telemetry.step(self.design, "outline_lookup"):
            outline = await self._cache.get_outline(doc_id)
        with self._telemetry.step(self.design, "select"):
            node_ids = await self._llm.select_nodes(question, outline)
        with self._telemetry.step(self.design, "fetch"):
            return await self._store.get_sections(doc_id, node_ids)
