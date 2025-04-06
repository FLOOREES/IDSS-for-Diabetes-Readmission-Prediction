import os
import logging
import pandas as pd
from icd_mapping import ICD9Encoder # type: ignore[import]

# =============================
# Configuration
# =============================

DATA_PATH = "data/diabetic_data_no_na_unknown.csv"
OUTPUT_PATH = "data/diabetic_data_embed_8.csv"
ICD9_CHAPTERS_PATH = "data/icd9Chapters.json"
ICD9_HIERARCHY_PATH = "data/icd9Hierarchy.json"
CACHE_DIR = "cache"

# =============================
# Logger Setup
# =============================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================
# Main Execution
# =============================

def main():
    logger.info("=== ICD9 EMBEDDING GENERATION STARTED ===")

    # Step 1: Load input data
    if not os.path.exists(DATA_PATH):
        logger.error(f"Input data file not found: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)
    logger.info(f"Loaded dataset with shape: {df.shape}")

    # Step 2: Initialize encoder
    encoder = ICD9Encoder(
        icd9_chapters_path=ICD9_CHAPTERS_PATH,
        icd9_hierarchy_path=ICD9_HIERARCHY_PATH,
        spacy_model="en_core_sci_md",
        cache_dir=CACHE_DIR
    )

    # Step 3: Add or reuse diagnosis embeddings (with t-SNE reduction)
    df = encoder.add_diag_embeddings(df, reduce=True)

    # Step 4: Save cached embeddings for future reuse
    encoder.save_cache()

    # Step 5: Save output dataset with reduced embeddings
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logger.info(f"Saved embedded dataset to: {OUTPUT_PATH}")

    logger.info("=== ICD9 EMBEDDING GENERATION COMPLETE ===")

if __name__ == "__main__":
    main()

import pandas as pd
import json
import numpy as np
import spacy
import logging
import os
from typing import Union, Dict, Any, Optional
from functools import lru_cache
from icdmappings import Mapper
from sklearn.manifold import TSNE, trustworthiness

class ICD9Encoder:
    """
    Encodes ICD-9 diagnosis codes into semantically meaningful embeddings using SpaCy
    and dimensionality reduction (e.g., t-SNE).
    """

    def __init__(
        self,
        icd9_chapters_path: str,
        icd9_hierarchy_path: str,
        spacy_model: str = "en_core_sci_md",
        general_weight: float = 0.3,
        specific_weight: float = 0.7,
        cache_dir: str = "cache"
    ):
        """
        Initialize the encoder with paths to ICD-9 metadata and SpaCy model.

        Args:
            icd9_chapters_path (str): Path to icd9Chapters.json file.
            icd9_hierarchy_path (str): Path to icd9Hierarchy.json file.
            spacy_model (str): Name of the SpaCy model to load.
            general_weight (float): Weight for general description in embedding.
            specific_weight (float): Weight for specific description in embedding.
            cache_dir (str): Directory path to save/load cached mappings.
        """
        with open(icd9_chapters_path, "r", encoding="utf-8") as f:
            self.icd9_chapters = json.load(f)
        with open(icd9_hierarchy_path, "r", encoding="utf-8") as f:
            self.icd9_hierarchy = json.load(f)

        self.nlp = spacy.load(spacy_model)
        self.mapper = Mapper()
        self.general_weight = general_weight
        self.specific_weight = specific_weight

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.embedding_cache_path = os.path.join(self.cache_dir, "embedding_cache.npz")
        self.reduction_cache_path = os.path.join(self.cache_dir, "reduction_map.npz")

        self._embedding_map = {}  # str -> np.ndarray
        self._reduction_map = {}  # tuple -> np.ndarray

        self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.embedding_cache_path):
            data = np.load(self.embedding_cache_path, allow_pickle=True)
            self._embedding_map = {k: v for k, v in data.items()}
            self.logger.info("Loaded embedding cache with %d entries.", len(self._embedding_map))
        if os.path.exists(self.reduction_cache_path):
            data = np.load(self.reduction_cache_path, allow_pickle=True)
            self._reduction_map = {eval(k): v for k, v in data.items()}
            self.logger.info("Loaded reduction cache with %d entries.", len(self._reduction_map))

    def save_cache(self):
        np.savez_compressed(self.embedding_cache_path, **self._embedding_map)
        red_serialized = {str(k): v for k, v in self._reduction_map.items()}
        np.savez_compressed(self.reduction_cache_path, **red_serialized)
        self.logger.info("Saved embedding and reduction caches to disk.")

    def find_diag(self, code: Any) -> Union[Dict[str, str], str]:
        if pd.isna(code):
            return 'No diag'
        code = str(code).replace('.', '').zfill(3)
        for d in self.icd9_hierarchy:
            if d.get('icd9') == code or d.get('icd9') == code + '0':
                return {
                    'specific': d.get('descLong', d.get('major', 'Unknown')),
                    'general': d.get('subchapter', '') + '. ' + d.get('chapter', '')
                }
            if d.get('threedigit') == code:
                return {
                    'specific': d.get('major', 'Unknown'),
                    'general': d.get('subchapter', '') + '. ' + d.get('chapter', '')
                }
        return 'Not found'

    def _embed_text(self, text: str) -> np.ndarray:
        if text in self._embedding_map:
            return self._embedding_map[text]
        vec = self.nlp(text).vector
        self._embedding_map[text] = vec
        return vec

    def _create_embedding(self, desc: Union[Dict[str, str], str]) -> np.ndarray:
        if isinstance(desc, dict):
            emb_specific = self._embed_text(desc['specific'])
            emb_general = self._embed_text(desc['general'])
            return self.general_weight * emb_general + self.specific_weight * emb_specific
        else:
            return self._embed_text(desc)

    def _reduce_embeddings(self, df: pd.DataFrame, n_components: int = 8) -> None:
        self.logger.info("Reducing embeddings with t-SNE to %d dimensions...", n_components)

        all_emb = np.vstack([
            np.vstack(df['emb_diag_1'].values),
            np.vstack(df['emb_diag_2'].values),
            np.vstack(df['emb_diag_3'].values)
        ])
        unique_emb = np.unique(all_emb, axis=0)

        to_reduce = [tuple(vec) for vec in unique_emb if tuple(vec) not in self._reduction_map]
        if not to_reduce:
            self.logger.info("All embeddings already reduced. Skipping t-SNE.")
            return

        tsne = TSNE(n_components=n_components, method='exact')
        reduced = tsne.fit_transform(np.array(to_reduce))

        for i, vec in enumerate(to_reduce):
            self._reduction_map[vec] = reduced[i]

        trust = trustworthiness(np.array(to_reduce), reduced, n_neighbors=5)
        self.logger.info("Trustworthiness of t-SNE embedding (k=5): %.3f", trust)

    def add_diag_embeddings(self, df: pd.DataFrame, reduce: bool = True) -> pd.DataFrame:
        required_cols = ['diag_1', 'diag_2', 'diag_3']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required diagnosis columns: {required_cols}")

        for i in range(1, 4):
            diag_col = f'diag_{i}'
            desc_col = f'desc_diag_{i}'
            if desc_col not in df.columns:
                df[desc_col] = df[diag_col].apply(lambda x: self.find_diag(x))

        for i in range(1, 4):
            desc_col = f'desc_diag_{i}'
            emb_col = f'emb_diag_{i}'
            df[emb_col] = df[desc_col].apply(lambda x: self._create_embedding(x))

        if reduce:
            self._reduce_embeddings(df)
            for i in range(1, 4):
                emb_col = f'emb_diag_{i}'
                red_col = f'red_emb_diag_{i}'
                df[red_col] = df[emb_col].apply(lambda x: tuple(x)).map(self._reduction_map)
                df.drop(columns=[emb_col], inplace=True)

        return df
