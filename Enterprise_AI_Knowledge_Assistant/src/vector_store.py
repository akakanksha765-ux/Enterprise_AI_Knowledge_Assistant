import json
from datetime import datetime

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# from src.huggingface import login_huggingface

from src.config import (
    settings,
    DOCS_DIR,
    INDEX_DIR,
    INFO_FILE,
    VECTOR_STORE_DIR,
)
from src.logger import logger

# **********************************
# EMBDDING MODEL
# **********************************
_embedding_model = None

def get_embedding_model():
    """Return embedding model."""

    global _embedding_model

    if _embedding_model is None:
        # login_huggingface()
        _embedding_model = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
        )

    return _embedding_model

# **********************************
# DOCUMENT LOADING, SPLITTING, AND INDEXING
# **********************************
def load_documents():
    """Load PDF documents."""

    loader = PyPDFDirectoryLoader(DOCS_DIR)
    return loader.load()

def split_documents(documents):
    """Split documents into chunks."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return splitter.split_documents(documents)

# **********************************
# INDEX METADATA
# **********************************
def save_index_info(chunks):
    """Save index information."""

    files = []

    for pdf in DOCS_DIR.glob("*.pdf"):
        stat = pdf.stat()

        files.append(
            {
                "name": pdf.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        )

    info = {
        "created_at": datetime.now().isoformat(),
        "embedding_model": settings.embedding_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "document_count": len(files),
        "chunk_count": len(chunks),
        "files": files,
    }

    with open(INFO_FILE, "w") as file:
        json.dump(info, file, indent=4)

# -------------- Functionality to sense any changes in the documents and rebuild the index if needed --------------

def load_index_info() -> dict | None:
    """Load index metadata."""

    if not INFO_FILE.exists():
        return None

    with open(INFO_FILE, "r") as file:
        return json.load(file)


def get_document_info() -> list:
    """Return current document metadata."""

    files = []

    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        stat = pdf.stat()

        files.append(
            {
                "name": pdf.name,
                "size": stat.st_size,
                "modified": stat.st_mtime,
            }
        )

    return files

# **********************************
# CHANGE DETECTION
# **********************************

def documents_changed() -> bool:
    """Check whether documents changed."""

    info = load_index_info()

    if info is None:
        return True

    return info["files"] != get_document_info()

def embedding_changed() -> bool:
    """Check whether embedding model changed."""

    info = load_index_info()

    if info is None:
        return True

    return info["embedding_model"] != settings.embedding_model

def chunking_changed() -> bool:
    """Check chunk settings."""

    info = load_index_info()

    if info is None:
        return True

    return (
        info["chunk_size"] != settings.chunk_size
        or info["chunk_overlap"] != settings.chunk_overlap
    )

# **********************************
# FAISS Indexing and Retrieval
# **********************************

def build_vector_store():
    """Build FAISS index."""

    # login_huggingface()
    documents = load_documents()
    chunks = split_documents(documents)

    vector_store = FAISS.from_documents(
        chunks,
        get_embedding_model(),
    )

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(VECTOR_STORE_DIR))
    save_index_info(chunks)

    return vector_store


def rebuild_vector_store():
    """Force rebuild of the vector index."""

    return build_vector_store()



def load_index():
    """Load FAISS index."""

    return FAISS.load_local(
        str(VECTOR_STORE_DIR),
        get_embedding_model(),
        allow_dangerous_deserialization=True,
    )

def initialize_vector_store():
    """Build or load vector store."""

    if not (VECTOR_STORE_DIR / "index.faiss").exists():
        logger.info("Building vector index...")
        return build_vector_store()

    if documents_changed():
        logger.info("Documents changed. Rebuilding index...")
        return build_vector_store()

    if embedding_changed():
        logger.info("Embedding model changed. Rebuilding index...")
        return build_vector_store()

    if chunking_changed():
        logger.info("Chunk settings changed. Rebuilding index...")
        return build_vector_store()

    logger.info("Loading existing index...")
    return load_index()


