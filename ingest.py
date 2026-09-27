"""
ingest.py — Document Ingestion Pipeline for the Hierarchical Academic AI Mentor.

Usage:
    python ingest.py                 # incremental: only new/changed files are processed
    python ingest.py --rebuild       # wipe the DB and re-ingest everything
    python ingest.py --docs my_pdfs --db data/other_db

Design goals (and WHY):
  * Domain-agnostic: the first sub-folder under /docs becomes a `domain` tag
    (docs/spring_boot/*.pdf -> domain="spring_boot"). The Router agent can later
    filter retrieval by domain, so mixing AI theory and Web Services docs in one
    DB does not pollute results.
  * Incremental: files are fingerprinted with SHA-256. Re-running the script after
    dropping one new PDF costs seconds, not minutes.
  * Idempotent: chunk IDs are deterministic, so re-ingesting can never create duplicates.
  * Fault-tolerant: one corrupted PDF is logged and skipped; it does not kill the run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF; `import fitz` alone is deprecated upstream
import pytesseract
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from PIL import Image
from tqdm import tqdm

# Maps a file extension to LangChain's Language enum, which supplies separators
# tuned to that language's syntax (split on class/method boundaries first,
# instead of blank lines). Extensions not listed here fall back to the
# default prose splitter, which still works, just less precisely.
_CODE_LANGUAGES: dict[str, Language] = {
    ".java": Language.JAVA,
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".cs": Language.CSHARP,
}

log = logging.getLogger("ingest")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class IngestConfig:
    """Single source of truth for tunables. Frozen => can't be mutated by accident."""

    docs_dir: Path = Path("docs")
    persist_dir: Path = Path("data/chroma_db")
    manifest_path: Path = Path("data/ingest_manifest.json")
    collection_name: str = "academic_mentor"

    # all-MiniLM-L6-v2: 384-dim, ~80MB, runs fine on CPU. Its max input is ~256
    # word-pieces, so very large chunks get silently truncated when embedded.
    # ~1000 chars (~200-250 tokens) is the sweet spot for this model.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 150  # overlap preserves context that straddles a boundary

    min_page_chars: int = 30   # pages with less text are likely images/scans/blank
    min_chunk_chars: int = 50  # tiny chunks (page numbers, footers) are noise
    batch_size: int = 128      # embed + insert in batches to bound RAM usage
    # Prose docs (assignment sheets, slides) AND source code (existing skeleton
    # projects). Code is essential context: the Drafter cannot correctly extend
    # a class/GUI it has never actually seen the real fields/methods of.
    extensions: tuple[str, ...] = (
        ".pdf", ".md", ".txt",
        ".java", ".py", ".js", ".ts", ".cpp", ".c", ".h", ".cs", ".sql",
    )
    # Code chunks are larger than prose chunks: splitting a method in half is
    # worse than a slightly bigger chunk. Chosen per-extension below.
    code_chunk_size: int = 1500
    code_chunk_overlap: int = 200

    # --- Image handling (PDFs only) ---
    extract_images: bool = True
    # Skip tiny embedded images (bullets, logos, icons) — real diagrams/screenshots
    # are almost always bigger than this on both axes.
    image_min_dim_px: int = 100
    # Below this many OCR'd characters, an image is treated as "not text-bearing"
    # (a diagram/photo rather than a text screenshot) and is a candidate for
    # captioning instead, if enabled.
    image_ocr_min_chars: int = 15
    # Tesseract language codes. Greek is included because course slides use Greek
    # UI labels (e.g. 'Πλήθος Γευμάτων'). Requires the matching traineddata
    # installed (see README); falls back to "eng" alone if unavailable.
    ocr_lang: str = "eng+ell"
    # Explicit path to tesseract.exe, for when it's not on PATH (common on
    # Windows if the installer's "add to PATH" step was skipped). Empty string
    # means "assume it's on PATH". ImageProcessor also tries the default
    # Windows install location automatically if this is empty and PATH fails.
    tesseract_cmd: str = ""
    # Off by default: needs a local Ollama vision model pulled and adds latency.
    # Enable with `python ingest.py --caption-images`.
    caption_images: bool = False
    vision_model: str = "moondream"  # small (~1.7GB), fast; use --vision-model llava for higher quality
    ollama_base_url: str = "http://localhost:11434"


# --------------------------------------------------------------------------- #
# Manifest: remembers which file versions are already in the DB
# --------------------------------------------------------------------------- #
class Manifest:
    """Maps relative file path -> SHA-256 of its content. Persisted as JSON."""

    def __init__(self, path: Path):
        self.path = path
        self._data: dict[str, str] = {}
        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("Manifest is corrupted; starting fresh.")

    def get(self, key: str) -> str | None:
        return self._data.get(key)

    def set(self, key: str, file_hash: str) -> None:
        self._data[key] = file_hash
        self._save()  # saved per file so a crash mid-run doesn't lose progress

    def remove(self, key: str) -> None:
        self._data.pop(key, None)
        self._save()

    def keys(self) -> set[str]:
        return set(self._data)

    def clear(self) -> None:
        self._data = {}
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Loading + cleaning
# --------------------------------------------------------------------------- #
class DocumentLoader:
    """Turns a file on disk into a list of page-level `Document`s with clean text."""

    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg

    def load(self, path: Path) -> list[Document]:
        suffix = path.suffix.lower()
        is_code = suffix in _CODE_LANGUAGES or suffix in {".sql", ".h"}

        if suffix == ".pdf":
            # PyMuPDF: fast, handles odd encodings/fonts, keeps 1 Document per page,
            # which gives us page numbers for citations ("see slide 14").
            raw = PyMuPDFLoader(str(path)).load()
        elif suffix in {".md", ".txt"} or is_code:
            raw = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

        cleaned: list[Document] = []
        for doc in raw:
            # WHY separate paths: the prose cleaner collapses runs of spaces and
            # rejoins hyphenated line-breaks, both of which would corrupt Java/
            # Python indentation and string literals. Code only gets minimal,
            # non-destructive cleanup.
            text = self._clean_code(doc.page_content) if is_code else self._clean(doc.page_content)
            if len(text) < self.cfg.min_page_chars:
                continue  # skip blank / image-only pages (would need OCR)
            doc.page_content = text
            cleaned.append(doc)

        if suffix == ".pdf" and not cleaned:
            log.warning("%s has no extractable text (scanned PDF?). OCR is needed.", path.name)
        return cleaned

    @staticmethod
    def _clean(text: str) -> str:
        """Fix common PDF extraction artifacts that hurt embedding quality."""
        text = text.replace("\x00", "")                # null bytes crash some DBs
        text = re.sub(r"-\n(?=[a-z])", "", text)       # re-join hyphenated line breaks
        text = re.sub(r"[ \t]+", " ", text)            # collapse runs of spaces
        text = re.sub(r"\n{3,}", "\n\n", text)         # collapse excess blank lines
        return text.strip()

    @staticmethod
    def _clean_code(text: str) -> str:
        """Strip only what's safe: null bytes and \\r. Indentation/spacing must survive intact."""
        text = text.replace("\x00", "").replace("\r\n", "\n").replace("\r", "\n")
        return text.strip("\n")


# --------------------------------------------------------------------------- #
# Image extraction: turns embedded PDF images into searchable text Documents
# --------------------------------------------------------------------------- #
class ImageProcessor:
    """
    Extracts embedded images from PDF pages and turns them into text via OCR
    (always, if enabled) and optionally a local vision model (only when OCR
    finds little/no text, e.g. an actual diagram rather than a text screenshot).

    WHY a separate class: image handling has its own failure modes (missing
    Tesseract binary, missing language pack, no Ollama running) that must never
    take down ingestion of the surrounding text — every failure here is caught
    and logged, not raised.
    """

    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg
        self._ocr_lang = cfg.ocr_lang       # may be downgraded to "eng" on first failure
        self._ocr_lang_checked = False
        self._warned_no_ollama = False
        self._configure_tesseract_path()

    def _configure_tesseract_path(self) -> None:
        """
        Point pytesseract at the tesseract binary explicitly, instead of relying
        on PATH. WHY: on Windows, forgetting to check "add to PATH" during the
        installer (or PATH changes needing a fresh terminal) is the single most
        common setup failure here, and it fails silently — OCR just returns
        empty strings, with no obvious error pointing at the cause.
        """
        if self.cfg.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.cfg.tesseract_cmd
            log.info("Using explicit Tesseract path: %s", self.cfg.tesseract_cmd)
            return

        # Nothing explicit given: if it's already on PATH, pytesseract will find
        # it on its own and we're done. Only probe further if that fails.
        try:
            pytesseract.get_tesseract_version()
            return
        except Exception:  # noqa: BLE001 - not on PATH; keep looking
            pass

        # Common default install locations, checked as a convenience fallback.
        candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            "/usr/local/bin/tesseract",
            "/opt/homebrew/bin/tesseract",
            "/usr/bin/tesseract",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                pytesseract.pytesseract.tesseract_cmd = candidate
                log.info("Auto-detected Tesseract at: %s", candidate)
                return

        log.warning(
            "Tesseract binary not found on PATH or in common install locations. "
            "OCR will be skipped for all images. Fix by setting IngestConfig.tesseract_cmd "
            "(or --tesseract-cmd) to the full path of tesseract.exe, e.g. "
            r'"C:\Program Files\Tesseract-OCR\tesseract.exe".'
        )

    def extract(self, pdf_path: Path, file_hash: str, domain: str) -> list[Document]:
        if not self.cfg.extract_images:
            return []
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not open %s for image extraction: %s", pdf_path.name, exc)
            return []

        results: list[Document] = []
        seen_hashes: set[str] = set()  # skip repeated logos/decorations within one file
        for page_index in range(len(doc)):
            page = doc[page_index]
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    extracted = doc.extract_image(xref)
                except Exception as exc:  # noqa: BLE001 - a single bad image must not stop the file
                    log.debug("Skipping unreadable image xref=%s on %s p.%d: %s",
                              xref, pdf_path.name, page_index + 1, exc)
                    continue

                raw_bytes = extracted["image"]
                img_hash = hashlib.sha256(raw_bytes).hexdigest()
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)

                try:
                    pil_img = Image.open(io.BytesIO(raw_bytes))
                    pil_img.load()
                except Exception:  # noqa: BLE001 - corrupt/unsupported image data
                    continue
                if pil_img.width < self.cfg.image_min_dim_px or pil_img.height < self.cfg.image_min_dim_px:
                    continue  # too small to be a meaningful diagram/screenshot (icon, bullet, rule line)

                text, source = self._describe(pil_img)
                if not text:
                    continue

                results.append(Document(
                    page_content=text,
                    metadata={
                        "source": self._rel_source(pdf_path),
                        "filename": pdf_path.name,
                        "domain": domain,
                        "is_code": False,
                        "is_image": True,
                        "image_source": source,  # "ocr" or "caption", so the Drafter can weigh confidence
                        "page": page_index + 1,
                        "file_hash": file_hash,
                        "img_index": img_idx,
                    },
                ))
        doc.close()
        return results

    def _describe(self, img: Image.Image) -> tuple[str, str]:
        """Returns (text, source) where source is 'ocr' or 'caption'; ('', '') if nothing useful found."""
        ocr_text = self._ocr(img)
        if len(ocr_text) >= self.cfg.image_ocr_min_chars:
            return ocr_text, "ocr"
        if self.cfg.caption_images:
            caption = self._caption(img)
            if caption:
                # Keep any short OCR fragment too (e.g. a chart's axis label) alongside the caption.
                combined = f"{caption}\n(visible text: {ocr_text})" if ocr_text else caption
                return combined, "caption"
        return ocr_text, "ocr"  # short/empty OCR text; still returned, filtered by caller if empty

    def _ocr(self, img: Image.Image) -> str:
        try:
            text = pytesseract.image_to_string(img, lang=self._ocr_lang)
        except pytesseract.TesseractError as exc:
            if not self._ocr_lang_checked and self._ocr_lang != "eng":
                log.warning(
                    "OCR language pack '%s' unavailable (%s); falling back to English only. "
                    "Install Greek traineddata for full accuracy on Greek course material.",
                    self._ocr_lang, exc,
                )
                self._ocr_lang = "eng"
                self._ocr_lang_checked = True
                return self._ocr(img)
            log.debug("OCR failed: %s", exc)
            return ""
        except Exception as exc:  # noqa: BLE001 - e.g. Tesseract binary not installed at all
            log.debug("OCR unavailable: %s", exc)
            return ""
        return re.sub(r"\n{2,}", "\n", text).strip()

    def _caption(self, img: Image.Image) -> str:
        """Local vision-model captioning via Ollama's REST API directly (kept import-free of src/,
        which cannot be imported here without creating a circular import with src/config.py)."""
        try:
            import requests  # local import: only needed when --caption-images is used
        except ImportError:
            return ""
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        try:
            resp = requests.post(
                f"{self.cfg.ollama_base_url}/api/generate",
                json={
                    "model": self.cfg.vision_model,
                    "prompt": "Describe this image factually in 1-2 sentences: what kind of diagram, "
                              "screenshot, or figure is it, and what does it show? Do not guess at "
                              "text you cannot clearly read.",
                    "images": [b64],
                    "stream": False,
                },
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except Exception as exc:  # noqa: BLE001 - Ollama not running, model not pulled, etc.
            if not self._warned_no_ollama:
                log.warning("Image captioning unavailable (%s). Continuing with OCR only.", exc)
                self._warned_no_ollama = True
            return ""

    def _rel_source(self, pdf_path: Path) -> str:
        return pdf_path.relative_to(self.cfg.docs_dir).as_posix()


# --------------------------------------------------------------------------- #
# The pipeline
# --------------------------------------------------------------------------- #
class IngestionPipeline:
    def __init__(self, cfg: IngestConfig):
        self.cfg = cfg
        self.loader = DocumentLoader(cfg)
        self.image_processor = ImageProcessor(cfg)
        self.manifest = Manifest(cfg.manifest_path)
        self.prose_splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            # Try to split on the most semantic boundary first: paragraph, then
            # line, then sentence, then word.
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        # One splitter per language, built lazily and cached: each uses that
        # language's own separators (e.g. split before a Java method/class
        # rather than mid-body). Unmapped code extensions (.sql, .h) fall back
        # to the prose splitter, which still works, just less precisely.
        self._code_splitters: dict[Language, RecursiveCharacterTextSplitter] = {
            lang: RecursiveCharacterTextSplitter.from_language(
                language=lang, chunk_size=cfg.code_chunk_size, chunk_overlap=cfg.code_chunk_overlap
            )
            for lang in set(_CODE_LANGUAGES.values())
        }
        log.info("Loading embedding model '%s' (first run downloads it)...", cfg.embedding_model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=cfg.embedding_model,
            encode_kwargs={"normalize_embeddings": True},  # required for cosine similarity
        )
        self.store = Chroma(
            collection_name=cfg.collection_name,
            embedding_function=self.embeddings,
            persist_directory=str(cfg.persist_dir),  # persistent, on local disk
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ---- public API -------------------------------------------------------- #
    def run(self, rebuild: bool = False) -> None:
        if not self.cfg.docs_dir.exists():
            self.cfg.docs_dir.mkdir(parents=True)
            log.error("Created '%s'. Put your PDFs there and re-run.", self.cfg.docs_dir)
            return

        if rebuild:
            log.info("--rebuild: wiping collection and manifest.")
            self.store.delete_collection()
            self.manifest.clear()
            self.store = Chroma(
                collection_name=self.cfg.collection_name,
                embedding_function=self.embeddings,
                persist_directory=str(self.cfg.persist_dir),
                collection_metadata={"hnsw:space": "cosine"},
            )

        files = self._discover_files()
        self._purge_deleted(files)

        stats = {"added": 0, "updated": 0, "skipped": 0, "failed": 0, "chunks": 0}
        for path in tqdm(files, desc="Ingesting", unit="file"):
            rel = self._rel(path)
            try:
                file_hash = self._hash_file(path)
                known = self.manifest.get(rel)
                if known == file_hash:
                    stats["skipped"] += 1
                    continue

                if known is not None:  # file changed -> drop its stale chunks first
                    self._delete_source(rel)
                    stats["updated"] += 1
                else:
                    stats["added"] += 1

                n = self._ingest_file(path, rel, file_hash)
                stats["chunks"] += n
                # Only mark as done AFTER success, so failures are retried next run.
                self.manifest.set(rel, file_hash)
            except Exception as exc:  # noqa: BLE001 - one bad file must not kill the batch
                stats["failed"] += 1
                log.exception("Failed on %s: %s", rel, exc)

        total = self.store._collection.count()  # noqa: SLF001
        log.info(
            "Done. added=%d updated=%d skipped=%d failed=%d new_chunks=%d | DB total=%d chunks",
            stats["added"], stats["updated"], stats["skipped"],
            stats["failed"], stats["chunks"], total,
        )

    # ---- internals --------------------------------------------------------- #
    def _ingest_file(self, path: Path, rel: str, file_hash: str) -> int:
        pages = self.loader.load(path)
        if not pages:
            return 0

        domain = self._domain_of(path)
        is_pdf = path.suffix.lower() == ".pdf"
        for page in pages:
            # Chroma metadata must be str/int/float/bool. Keep it flat and simple.
            page.metadata = {
                "source": rel,
                "filename": path.name,
                "domain": domain,
                "is_code": path.suffix.lower() in _CODE_LANGUAGES,
                # PyMuPDF pages are 0-indexed; humans count from 1. Non-PDF files
                # (code, markdown) have no real pagination, so use 1 as a placeholder
                # rather than a misleading page number.
                "page": (int(page.metadata.get("page", 0)) + 1) if is_pdf else 1,
                "file_hash": file_hash,
            }

        splitter = self._code_splitters.get(_CODE_LANGUAGES.get(path.suffix.lower()), self.prose_splitter)
        chunks = [
            c for c in splitter.split_documents(pages)
            if len(c.page_content) >= self.cfg.min_chunk_chars
        ]

        # Deterministic IDs: same file content + same position => same ID.
        ids = [f"{file_hash[:16]}-{i:05d}" for i in range(len(chunks))]
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

        # Images are extracted whole (not further split — an OCR/caption block is
        # already a single cohesive unit) and get their own ID namespace so they
        # can never collide with text-chunk IDs.
        if is_pdf:
            image_docs = self.image_processor.extract(path, file_hash, domain)
            for i, d in enumerate(image_docs):
                d.metadata["chunk_index"] = i
            image_ids = [f"{file_hash[:16]}-img-{i:05d}" for i in range(len(image_docs))]
            chunks += image_docs
            ids += image_ids

        # Batched insert keeps peak memory low even for 500-page textbooks.
        bs = self.cfg.batch_size
        for start in range(0, len(chunks), bs):
            self.store.add_documents(chunks[start:start + bs], ids=ids[start:start + bs])
        return len(chunks)

    def _delete_source(self, rel: str) -> None:
        existing = self.store.get(where={"source": rel}).get("ids", [])
        if existing:
            self.store.delete(ids=existing)

    def _purge_deleted(self, current_files: list[Path]) -> None:
        """If a PDF was removed from /docs, remove its chunks from the DB too."""
        current = {self._rel(p) for p in current_files}
        for gone in self.manifest.keys() - current:
            log.info("Removing chunks of deleted file: %s", gone)
            self._delete_source(gone)
            self.manifest.remove(gone)

    def _discover_files(self) -> list[Path]:
        files = [
            p for p in sorted(self.cfg.docs_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in self.cfg.extensions
        ]
        log.info("Found %d supported file(s) in '%s'.", len(files), self.cfg.docs_dir)
        return files

    def _rel(self, path: Path) -> str:
        return path.relative_to(self.cfg.docs_dir).as_posix()

    def _domain_of(self, path: Path) -> str:
        """docs/<domain>/file.pdf -> '<domain>'; files directly in docs/ -> 'general'."""
        parts = path.relative_to(self.cfg.docs_dir).parts
        return parts[0].lower() if len(parts) > 1 else "general"

    @staticmethod
    def _hash_file(path: Path) -> str:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):  # 1 MB blocks
                h.update(block)
        return h.hexdigest()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--docs", type=Path, default=IngestConfig.docs_dir)
    parser.add_argument("--db", type=Path, default=IngestConfig.persist_dir)
    parser.add_argument("--rebuild", action="store_true", help="Wipe DB and re-ingest all.")
    parser.add_argument("--no-images", action="store_true", help="Disable image extraction entirely.")
    parser.add_argument("--caption-images", action="store_true",
                         help="Also caption diagrams/photos with a local Ollama vision model "
                              "(OCR alone only captures embedded TEXT, not diagram content). "
                              "Requires: ollama pull moondream  (or llava for higher quality).")
    parser.add_argument("--vision-model", default=IngestConfig.vision_model,
                         help="Ollama vision model tag to use with --caption-images.")
    parser.add_argument("--ocr-lang", default=IngestConfig.ocr_lang,
                         help="Tesseract language code(s), e.g. 'eng' or 'eng+ell'.")
    parser.add_argument("--tesseract-cmd", default=IngestConfig.tesseract_cmd,
                         help=r'Full path to tesseract.exe, e.g. "C:\Program Files\Tesseract-OCR\tesseract.exe". '
                              "Use this instead of adding Tesseract to PATH.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
    # Keep the manifest next to the DB so different DBs never share state.
    cfg = IngestConfig(
        docs_dir=args.docs,
        persist_dir=args.db,
        manifest_path=args.db.parent / f"{args.db.name}_manifest.json",
        extract_images=not args.no_images,
        caption_images=args.caption_images,
        vision_model=args.vision_model,
        ocr_lang=args.ocr_lang,
        tesseract_cmd=args.tesseract_cmd,
    )
    IngestionPipeline(cfg).run(rebuild=args.rebuild)
    return 0


if __name__ == "__main__":
    sys.exit(main())
