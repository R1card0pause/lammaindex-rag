from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import re
import shutil
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
import requests
from pypdf import PdfReader, PdfWriter
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from pydantic import PrivateAttr


DEFAULT_INPUT_DIR = Path("./pdfs")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
MINERU_API_BASE_URL = "https://mineru.net/api/v4"


class SiliconFlowEmbedding(BaseEmbedding):
    model: str
    api_key: str
    batch_size: int = 64
    timeout: int = 120
    _session: requests.Session = PrivateAttr(default_factory=requests.Session)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        response = self._session.post(
            f"{SILICONFLOW_BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={"model": self.model, "input": texts},
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed_batch([query])[0]

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            embeddings.extend(self._embed_batch(texts[i : i + self.batch_size]))
        return embeddings


@dataclass
class ParseResult:
    pdf: str
    output_dir: str
    markdown_files: list[str]
    seconds: float
    ok: bool
    error: str = ""


@dataclass(frozen=True)
class PdfPart:
    path: Path
    original_pdf: Path
    start_page_index: int


def now() -> float:
    return time.perf_counter()


def expand_env(value: str) -> str:
    pattern = re.compile(r"\$\{([^}]+)\}")
    return pattern.sub(lambda m: os.environ.get(m.group(1), ""), value)


def load_model_config(path: Path, embed_key: str) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    embed_cfg = raw["embeddings"][embed_key]
    api_key = expand_env(str(embed_cfg["api_key"]))
    if not api_key:
        raise RuntimeError(
            f"Missing API key for {embed_key}. Set LAZYLLM_SILICONFLOW_API_KEY first."
        )
    return {
        "model": embed_cfg["model"],
        "api_key": api_key,
        "num_worker": int(embed_cfg.get("num_worker", 5)),
    }


def safe_stem(path: Path) -> str:
    return re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", path.stem)[:120]


def mineru_data_id(path: Path) -> str:
    return f"pdfpart-{uuid4().hex}"


def parse_page_start(split_name: str) -> int:
    match = re.search(r"_part_(\d+)-\d+\.pdf$", split_name)
    return int(match.group(1)) if match else 0


def split_large_pdf(
    pdf_path: Path,
    max_size_mb: int = 200,
    splits_root: Path | None = None,
) -> list[PdfPart]:
    """Split a PDF into page ranges until each part is below MinerU API's size cap."""
    max_size_bytes = max_size_mb * 1024 * 1024
    original_size = pdf_path.stat().st_size
    if original_size <= max_size_bytes:
        return [PdfPart(pdf_path, pdf_path, 0)]

    if splits_root:
        splits_dir = splits_root / f"{safe_stem(pdf_path)}.splits"
    else:
        splits_dir = Path(f"{pdf_path}.splits")
    if splits_dir.exists():
        pdf_files = sorted(splits_dir.glob("*.pdf"), key=lambda p: parse_page_start(p.name))
        if pdf_files and all(p.stat().st_size <= max_size_bytes for p in pdf_files):
            return [PdfPart(p, pdf_path, parse_page_start(p.name)) for p in pdf_files]

    if splits_dir.exists():
        shutil.rmtree(splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Some planning PDFs contain large image streams. pypdf's default guard is
    # lower than MinerU's 200 MB file limit, so raise it for splitting only.
    import pypdf.filters as pypdf_filters

    pypdf_filters.MAX_DECLARED_STREAM_LENGTH = max(
        pypdf_filters.MAX_DECLARED_STREAM_LENGTH,
        2 * 1024 * 1024 * 1024,
        max_size_bytes,
    )
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    basename = pdf_path.stem
    num_parts = original_size // max_size_bytes + 1
    pages_per_part = max(1, total_pages // num_parts)

    chunks: list[tuple[list[int], int]] = []
    start_page = 0
    while start_page < total_pages:
        end_page = min(start_page + pages_per_part, total_pages)
        chunks.append((list(range(start_page, end_page)), start_page))
        start_page = end_page

    final_result: list[PdfPart] = []
    while chunks:
        page_indices, offset = chunks.pop(0)
        writer = PdfWriter()
        for page_index in page_indices:
            writer.add_page(reader.pages[page_index])

        part_name = f"{basename}_part_{offset}-{offset + len(page_indices)}.pdf"
        part_path = splits_dir / part_name
        with part_path.open("wb") as f:
            writer.write(f)

        if part_path.stat().st_size <= max_size_bytes or len(page_indices) == 1:
            final_result.append(PdfPart(part_path, pdf_path, offset))
        else:
            mid = len(page_indices) // 2
            chunks.insert(0, (page_indices[mid:], offset + mid))
            chunks.insert(0, (page_indices[:mid], offset))
            part_path.unlink()

    return sorted(final_result, key=lambda item: item.start_page_index)


def split_pdf_inputs(pdfs: list[Path], max_size_mb: int, splits_root: Path | None) -> list[PdfPart]:
    parts: list[PdfPart] = []
    for pdf in pdfs:
        parts.extend(split_large_pdf(pdf, max_size_mb=max_size_mb, splits_root=splits_root))
    return parts


def read_pdfs_with_pypdf(pdfs: list[Path]) -> tuple[list[Document], list[ParseResult]]:
    docs: list[Document] = []
    results: list[ParseResult] = []
    for pdf in pdfs:
        started = now()
        try:
            reader = PdfReader(str(pdf))
            parts = []
            for page_index, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    parts.append(f"\n\n[page {page_index + 1}]\n{text}")
            text = "".join(parts).strip()
            ok = bool(text)
            if ok:
                docs.append(
                    Document(
                        text=text,
                        metadata={
                            "source_pdf": str(pdf),
                            "reader": "pypdf",
                            "page_count": len(reader.pages),
                        },
                    )
                )
            results.append(
                ParseResult(
                    pdf=str(pdf),
                    output_dir="",
                    markdown_files=[],
                    seconds=now() - started,
                    ok=ok,
                    error="" if ok else "no extractable text",
                )
            )
        except Exception as exc:
            results.append(
                ParseResult(
                    pdf=str(pdf),
                    output_dir="",
                    markdown_files=[],
                    seconds=now() - started,
                    ok=False,
                    error=str(exc),
                )
            )
    return docs, results


def find_markdown_in_zip(zip_path: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)
    md_files = sorted(extract_dir.rglob("*.md"))
    return [p for p in md_files if p.is_file() and p.stat().st_size > 0]


def read_pdfs_with_mineru_api(
    pdf_parts: list[PdfPart],
    parse_root: Path,
    token: str,
    model_version: str,
    is_ocr: bool,
    enable_formula: bool,
    enable_table: bool,
    poll_interval: float,
    poll_timeout: float,
    batch_size: int,
    upload_workers: int,
) -> tuple[list[Document], list[ParseResult]]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
    }
    parse_root.mkdir(parents=True, exist_ok=True)
    documents: list[Document] = []
    parse_results: list[ParseResult] = []

    for start in range(0, len(pdf_parts), batch_size):
        batch_parts = pdf_parts[start : start + batch_size]
        files_payload = []
        for part in batch_parts:
            files_payload.append(
                {
                    "name": part.path.name,
                    "data_id": mineru_data_id(part.path),
                    "is_ocr": is_ocr,
                }
            )

        request_started = now()
        response = requests.post(
            f"{MINERU_API_BASE_URL}/file-urls/batch",
            headers=headers,
            json={
                "files": files_payload,
                "model_version": model_version,
                "enable_formula": enable_formula,
                "enable_table": enable_table,
                "language": "ch",
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != 0:
            raise RuntimeError(f"MinerU upload-url request failed: {payload}")

        batch_id = payload["data"]["batch_id"]
        upload_urls = payload["data"]["file_urls"]
        if len(upload_urls) != len(batch_parts):
            raise RuntimeError("MinerU returned a different number of upload URLs.")

        def upload_one(part_and_url: tuple[PdfPart, str]) -> None:
            part, url = part_and_url
            with part.path.open("rb") as f:
                upload_response = requests.put(url, data=f, timeout=600)
            upload_response.raise_for_status()

        with futures.ThreadPoolExecutor(max_workers=max(1, upload_workers)) as pool:
            list(pool.map(upload_one, zip(batch_parts, upload_urls)))

        deadline = time.time() + poll_timeout
        latest_result: dict[str, Any] | None = None
        while time.time() < deadline:
            poll_response = requests.get(
                f"{MINERU_API_BASE_URL}/extract-results/batch/{batch_id}",
                headers=headers,
                timeout=60,
            )
            poll_response.raise_for_status()
            latest_result = poll_response.json()
            if latest_result.get("code") != 0:
                raise RuntimeError(f"MinerU poll failed: {latest_result}")
            results = latest_result.get("data", {}).get("extract_result", [])
            states = [item.get("state") for item in results]
            if results and all(state in {"done", "failed"} for state in states):
                break
            time.sleep(poll_interval)
        else:
            raise TimeoutError(f"MinerU batch {batch_id} did not finish within {poll_timeout}s")

        result_by_name = {
            item.get("file_name"): item
            for item in latest_result.get("data", {}).get("extract_result", [])
        }
        for part in batch_parts:
            pdf = part.path
            item = result_by_name.get(pdf.name, {})
            out_dir = parse_root / safe_stem(pdf)
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "mineru_api_result.json").write_text(
                json.dumps(item, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if item.get("state") != "done":
                parse_results.append(
                    ParseResult(
                        pdf=str(part.original_pdf),
                        output_dir=str(out_dir),
                        markdown_files=[],
                        seconds=now() - request_started,
                        ok=False,
                        error=item.get("err_msg") or f"state={item.get('state')}",
                    )
                )
                continue

            zip_url = item.get("full_zip_url")
            zip_path = out_dir / "mineru_result.zip"
            zip_response = requests.get(zip_url, timeout=600)
            zip_response.raise_for_status()
            zip_path.write_bytes(zip_response.content)
            md_files = find_markdown_in_zip(zip_path, out_dir / "unzipped")
            for md_path in md_files:
                text = md_path.read_text(encoding="utf-8", errors="ignore").strip()
                if text:
                    documents.append(
                        Document(
                            text=text,
                            metadata={
                                "source_pdf": str(part.original_pdf),
                                "source_part_pdf": str(pdf),
                                "start_page_index": part.start_page_index,
                                "markdown_file": str(md_path),
                                "reader": "mineru-api",
                                "batch_id": batch_id,
                                "model_version": model_version,
                            },
                        )
                    )
            parse_results.append(
                ParseResult(
                    pdf=str(part.original_pdf),
                    output_dir=str(out_dir),
                    markdown_files=[str(p) for p in md_files],
                    seconds=now() - request_started,
                    ok=bool(md_files),
                    error="" if md_files else "MinerU result zip contained no markdown",
                )
            )

    return documents, parse_results


def write_event(log_path: Path, event: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def select_pdfs(input_dir: Path, max_files: int | None, smallest_first: bool) -> list[Path]:
    pdfs = sorted(input_dir.glob("*.pdf"))
    if smallest_first:
        pdfs = sorted(pdfs, key=lambda p: p.stat().st_size)
    return pdfs[:max_files] if max_files else pdfs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--work-dir", type=Path, default=Path("./runs/latest"))
    parser.add_argument("--config", type=Path, default=Path("./models.yaml"))
    parser.add_argument("--embed-key", default="embed_1")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--smallest-first", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--pdf-reader", choices=["mineru-api", "pypdf"], default="mineru-api")
    parser.add_argument("--mineru-api-model", default="vlm")
    parser.add_argument("--mineru-api-ocr", action="store_true")
    parser.add_argument("--mineru-api-disable-formula", action="store_true")
    parser.add_argument("--mineru-api-disable-table", action="store_true")
    parser.add_argument("--mineru-api-poll-interval", type=float, default=10.0)
    parser.add_argument("--mineru-api-timeout", type=float, default=3600.0)
    parser.add_argument("--mineru-api-batch-size", type=int, default=50)
    parser.add_argument("--mineru-api-upload-workers", type=int, default=8)
    parser.add_argument("--mineru-api-max-size-mb", type=int, default=200)
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    parse_dir = work_dir / "mineru"
    persist_dir = work_dir / "storage"
    log_path = work_dir / "events.jsonl"
    work_dir.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        log_path.unlink()

    pdfs = select_pdfs(args.input_dir, args.max_files, args.smallest_first)
    if not pdfs:
        raise RuntimeError(f"No PDF files found in {args.input_dir}")

    pdf_parts = [PdfPart(pdf, pdf, 0) for pdf in pdfs]
    if args.pdf_reader == "mineru-api":
        pdf_parts = split_pdf_inputs(
            pdfs,
            max_size_mb=args.mineru_api_max_size_mb,
            splits_root=work_dir / "pdf_splits",
        )

    summary: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "work_dir": str(work_dir),
        "pdf_count": len(pdfs),
        "pdf_part_count": len(pdf_parts),
        "embedding_model": None if args.parse_only else args.embed_key,
        "mineru_api_batch_size": args.mineru_api_batch_size if args.pdf_reader == "mineru-api" else None,
        "mineru_api_upload_workers": args.mineru_api_upload_workers if args.pdf_reader == "mineru-api" else None,
        "mineru_api_max_size_mb": args.mineru_api_max_size_mb if args.pdf_reader == "mineru-api" else None,
        "started_at_unix": time.time(),
    }
    write_event(log_path, {"event": "start", **summary})

    total_started = now()
    parse_started = now()
    if args.pdf_reader == "pypdf":
        documents, parse_results = read_pdfs_with_pypdf(pdfs)
        for result in parse_results:
            write_event(log_path, {"event": "parse_file", **asdict(result)})
            status = "OK" if result.ok else "FAIL"
            print(f"[parse:pypdf] {status} {Path(result.pdf).name} {result.seconds:.2f}s", flush=True)
    else:
        token = os.environ.get("MINERU_API_TOKEN")
        if not token:
            raise RuntimeError("Missing MINERU_API_TOKEN")
        documents, parse_results = read_pdfs_with_mineru_api(
            pdf_parts=pdf_parts,
            parse_root=parse_dir,
            token=token,
            model_version=args.mineru_api_model,
            is_ocr=args.mineru_api_ocr,
            enable_formula=not args.mineru_api_disable_formula,
            enable_table=not args.mineru_api_disable_table,
            poll_interval=args.mineru_api_poll_interval,
            poll_timeout=args.mineru_api_timeout,
            batch_size=args.mineru_api_batch_size,
            upload_workers=args.mineru_api_upload_workers,
        )
        for result in parse_results:
            write_event(log_path, {"event": "parse_file", **asdict(result)})
            status = "OK" if result.ok else "FAIL"
            print(f"[parse:mineru-api] {status} {Path(result.pdf).name} {result.seconds:.2f}s", flush=True)

    parse_seconds = now() - parse_started
    failed = [r for r in parse_results if not r.ok]
    if failed:
        write_event(log_path, {"event": "failed", "failed": [asdict(r) for r in failed]})
        raise RuntimeError(f"{len(failed)} MinerU parse jobs failed; see {log_path}")

    if not documents:
        raise RuntimeError("MinerU produced no readable markdown documents.")

    if args.parse_only:
        total_seconds = now() - total_started
        final = {
            "event": "finish_parse_only",
            "pdf_count": len(pdfs),
            "pdf_part_count": len(pdf_parts),
            "document_count": len(documents),
            "parse_seconds": parse_seconds,
            "total_seconds": total_seconds,
            "finished_at_unix": time.time(),
        }
        write_event(log_path, final)
        print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
        return 0

    model_cfg = load_model_config(args.config, args.embed_key)

    Settings.embed_model = SiliconFlowEmbedding(
        model=model_cfg["model"],
        api_key=model_cfg["api_key"],
        batch_size=64,
        timeout=120,
    )
    Settings.llm = None
    Settings.node_parser = SentenceSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    index_started = now()
    nodes = Settings.node_parser.get_nodes_from_documents(documents)
    write_event(log_path, {"event": "nodes_created", "node_count": len(nodes)})
    storage_context = StorageContext.from_defaults()
    index = VectorStoreIndex(nodes, storage_context=storage_context, use_async=True, insert_batch_size=256)
    persist_dir.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(persist_dir))
    index_seconds = now() - index_started

    total_seconds = now() - total_started
    final = {
        "event": "finish",
        "pdf_count": len(pdfs),
        "pdf_part_count": len(pdf_parts),
        "document_count": len(documents),
        "node_count": len(nodes),
        "parse_seconds": parse_seconds,
        "index_seconds": index_seconds,
        "total_seconds": total_seconds,
        "persist_dir": str(persist_dir),
        "finished_at_unix": time.time(),
    }
    write_event(log_path, final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
