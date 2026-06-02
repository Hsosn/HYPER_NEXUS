"""
NLP Workbench — Comprehensive Natural Language Processing Toolkit
==================================================================
Provides advanced NLP capabilities: text preprocessing, embeddings,
sentiment analysis, NER simulation, text generation scaffolding,
language detection, topic modeling, and evaluation metrics.
"""

from __future__ import annotations

import collections
import json
import math
import random
import re
import string
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Counter, Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class TextPreprocessing(Enum):
    LOWERCASE = "lowercase"
    STRIP_PUNCTUATION = "strip_punctuation"
    STRIP_WHITESPACE = "strip_whitespace"
    REMOVE_STOPWORDS = "remove_stopwords"
    STEMMING = "stemming"
    LEMMATIZATION = "lemmatization"
    REMOVE_URLS = "remove_urls"
    REMOVE_EMAILS = "remove_emails"
    REMOVE_NUMBERS = "remove_numbers"
    EXPAND_CONTRACTIONS = "expand_contractions"
    NORMALIZE_UNICODE = "normalize_unicode"


class EmbeddingModel(Enum):
    WORD2VEC = "word2vec"
    GLOVE = "glove"
    FASTTEXT = "fasttext"
    BERT = "bert"
    SBERT = "sbert"
    GPT_EMBED = "gpt_embedding"
    CUSTOM = "custom"


STOP_WORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "shall", "should", "may", "might", "i", "you", "he",
    "she", "it", "we", "they", "this", "that", "these", "those", "am",
    "not", "no", "nor", "so", "as", "if", "then", "than", "just", "about",
    "above", "after", "again", "all", "also", "any", "because", "before",
    "between", "both", "each", "few", "more", "most", "other", "some",
    "such", "only", "own", "same", "too", "very", "into", "over", "under",
    "up", "out", "off", "down",
}


# ---------------------------------------------------------------------------
# Text Preprocessor
# ---------------------------------------------------------------------------

class TextPreprocessor:
    """Configurable text preprocessing pipeline."""

    DEFAULT_PIPELINE = [
        TextPreprocessing.NORMALIZE_UNICODE,
        TextPreprocessing.LOWERCASE,
        TextPreprocessing.STRIP_WHITESPACE,
        TextPreprocessing.REMOVE_URLS,
        TextPreprocessing.STRIP_PUNCTUATION,
    ]

    @staticmethod
    def normalize_unicode(text: str) -> str:
        try:
            import unicodedata
            return unicodedata.normalize("NFKC", text)
        except ImportError:
            return text

    @staticmethod
    def strip_punctuation(text: str) -> str:
        return text.translate(str.maketrans("", "", string.punctuation))

    @staticmethod
    def remove_stopwords(tokens: List[str]) -> List[str]:
        return [t for t in tokens if t.lower() not in STOP_WORDS]

    @staticmethod
    def remove_urls(text: str) -> str:
        return re.sub(r"https?://\S+|www\.\S+", "", text)

    @staticmethod
    def remove_emails(text: str) -> str:
        return re.sub(r"\S+@\S+", "", text)

    @staticmethod
    def remove_numbers(text: str) -> str:
        return re.sub(r"\d+", "", text)

    @staticmethod
    def stem(tokens: List[str]) -> List[str]:
        """Simple Porter-inspired stemmer simulation."""
        stems = []
        for token in tokens:
            word = token.lower()
            if word.endswith("ing"):
                word = word[:-3]
            elif word.endswith("ed"):
                word = word[:-2]
            elif word.endswith("ly"):
                word = word[:-2]
            elif word.endswith("tion"):
                word = word[:-4] + "te"
            elif word.endswith("ness"):
                word = word[:-4]
            stems.append(word)
        return stems

    @classmethod
    def process(cls, text: str,
                steps: Optional[List[TextPreprocessing]] = None) -> Dict[str, Any]:
        """Apply preprocessing pipeline and return results."""
        if steps is None:
            steps = cls.DEFAULT_PIPELINE

        original = text
        result = text
        applied = []

        for step in steps:
            if step == TextPreprocessing.LOWERCASE:
                result = result.lower()
                applied.append("lowercase")
            elif step == TextPreprocessing.STRIP_PUNCTUATION:
                result = cls.strip_punctuation(result)
                applied.append("strip_punctuation")
            elif step == TextPreprocessing.STRIP_WHITESPACE:
                result = re.sub(r"\s+", " ", result).strip()
                applied.append("strip_whitespace")
            elif step == TextPreprocessing.REMOVE_URLS:
                result = cls.remove_urls(result)
                applied.append("remove_urls")
            elif step == TextPreprocessing.REMOVE_EMAILS:
                result = cls.remove_emails(result)
                applied.append("remove_emails")
            elif step == TextPreprocessing.REMOVE_NUMBERS:
                result = cls.remove_numbers(result)
                applied.append("remove_numbers")
            elif step == TextPreprocessing.NORMALIZE_UNICODE:
                result = cls.normalize_unicode(result)
                applied.append("normalize_unicode")

        tokens = result.split()
        if TextPreprocessing.REMOVE_STOPWORDS in steps:
            tokens = cls.remove_stopwords(tokens)
            applied.append("remove_stopwords")
        if TextPreprocessing.STEMMING in steps:
            tokens = cls.stem(tokens)
            applied.append("stemming")

        return {
            "original": original,
            "processed": result,
            "tokens": tokens,
            "num_tokens": len(tokens),
            "steps_applied": applied,
            "compression_ratio": len(original) / max(len(result), 1),
        }


# ---------------------------------------------------------------------------
# Embedding Generator (Simulated)
# ---------------------------------------------------------------------------

class EmbeddingGenerator:
    """Generate and manipulate text embeddings."""

    @staticmethod
    def generate(text: str, model: str = "bert",
                 dimensions: int = 768) -> Dict[str, Any]:
        """Generate an embedding vector for text (simulated)."""
        rng = random.Random(hash(text) & 0xFFFFFFFF)
        vector = [rng.gauss(0, 0.1) for _ in range(dimensions)]
        # Normalize
        norm = math.sqrt(sum(v * v for v in vector))
        vector = [v / norm for v in vector]

        return {
            "model": model,
            "dimensions": dimensions,
            "vector": vector[:32],  # sample
            "norm": norm,
            "text_length": len(text),
        }

    @staticmethod
    def cosine_similarity(vec_a: List[float],
                          vec_b: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a, dtype=np.float64)
        b = np.array(vec_b, dtype=np.float64)
        dot = float(np.dot(a, b))
        norm = float(np.linalg.norm(a) * np.linalg.norm(b))
        return dot / norm if norm > 0 else 0.0

    @staticmethod
    def batch_embed(texts: List[str], model: str = "bert",
                    dimensions: int = 768) -> Dict[str, Any]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        for text in texts:
            emb = EmbeddingGenerator.generate(text, model, dimensions)
            embeddings.append(emb["vector"])
        return {
            "model": model,
            "dimensions": dimensions,
            "num_texts": len(texts),
            "embeddings_sample": embeddings[:5],
            "shape": [len(texts), dimensions],
        }


# ---------------------------------------------------------------------------
# Sentiment Analyzer
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """Rule-based sentiment analysis with lexicon support."""

    POSITIVE_LEXICON: Set[str] = {
        "good", "great", "excellent", "amazing", "wonderful", "fantastic",
        "brilliant", "outstanding", "superb", "awesome", "love", "beautiful",
        "happy", "joyful", "positive", "perfect", "best", "impressive",
        "pleasant", "delightful", "splendid", "fabulous", "terrific",
        "marvelous", "magnificent", "glorious", "lovely", "nice",
    }

    NEGATIVE_LEXICON: Set[str] = {
        "bad", "terrible", "awful", "horrible", "poor", "dreadful",
        "hate", "ugly", "disgusting", "repulsive", "nasty", "disappointing",
        "worst", "inferior", "mediocre", "lousy", "appalling", "atrocious",
        "abysmal", "horrendous", "displeasing", "unsatisfactory",
        "annoying", "frustrating", "irritating",
    }

    INTENSIFIERS: Set[str] = {
        "very", "extremely", "incredibly", "absolutely", "totally",
        "completely", "utterly", "highly", "remarkably", "exceptionally",
    }

    @classmethod
    def analyze(cls, text: str) -> Dict[str, Any]:
        """Analyze sentiment of text."""
        tokens = text.lower().split()
        positive_score = 0.0
        negative_score = 0.0
        intensifier = 1.0

        for token in tokens:
            if token in cls.INTENSIFIERS:
                intensifier = 1.5
                continue
            if token in cls.POSITIVE_LEXICON:
                positive_score += intensifier
            elif token in cls.NEGATIVE_LEXICON:
                negative_score += intensifier
            intensifier = 1.0

        total = positive_score + negative_score
        if total == 0:
            compound = 0.0
        else:
            compound = (positive_score - negative_score) / total

        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"

        return {
            "label": label,
            "score": compound,
            "positive": positive_score,
            "negative": negative_score,
            "confidence": min(abs(compound) * 2, 1.0),
            "num_tokens": len(tokens),
        }

    @classmethod
    def analyze_batch(cls, texts: List[str]) -> List[Dict[str, Any]]:
        return [cls.analyze(t) for t in texts]


# ---------------------------------------------------------------------------
# Topic Modeling (Simulated)
# ---------------------------------------------------------------------------

class TopicModeler:
    """Simulated topic modeling with LDA-like extraction."""

    @staticmethod
    def extract_topics(documents: List[str],
                       num_topics: int = 5,
                       num_words: int = 10,
                       seed: int = 0) -> Dict[str, Any]:
        """Extract topics from documents (simulated)."""
        rng = random.Random(seed)
        all_tokens = []
        for doc in documents:
            all_tokens.extend(re.findall(r"\w+", doc.lower()))

        # Build co-occurrence clusters (simulated)
        vocab = list(set(all_tokens))
        if len(vocab) < num_topics * num_words:
            num_topics = max(1, len(vocab) // num_words)

        topics = []
        for t in range(num_topics):
            words = rng.sample(vocab, min(num_words, len(vocab)))
            topics.append({
                "topic_id": t,
                "words": words,
                "weights": [rng.random() for _ in words],
            })

        # Document-topic distribution
        doc_topics = []
        for doc in documents:
            dist = [rng.random() for _ in range(num_topics)]
            total = sum(dist)
            dist = [d / total for d in dist]
            doc_topics.append(dist)

        return {
            "num_topics": num_topics,
            "num_documents": len(documents),
            "vocab_size": len(vocab),
            "topics": topics,
            "document_topics_sample": doc_topics[:5],
        }


# ---------------------------------------------------------------------------
# Text Generation (Scaffolding)
# ---------------------------------------------------------------------------

class TextGenerator:
    """N-gram based text generation with Markov chains."""

    def __init__(self, n: int = 3) -> None:
        self.n = n
        self.ngrams: Dict[Tuple[str, ...], List[str]] = {}
        self.trained: bool = False

    def train(self, texts: List[str]) -> None:
        """Train the n-gram model on texts."""
        self.ngrams.clear()
        for text in texts:
            tokens = re.findall(r"\w+", text.lower())
            for i in range(len(tokens) - self.n + 1):
                key = tuple(tokens[i:i + self.n - 1])
                self.ngrams.setdefault(key, []).append(tokens[i + self.n - 1])
        self.trained = True

    def generate(self, seed_text: str = "",
                 max_length: int = 20) -> Dict[str, Any]:
        """Generate text from the trained model."""
        if not self.trained:
            return {"text": "", "error": "Model not trained"}

        tokens = re.findall(r"\w+", seed_text.lower()) if seed_text else []
        if len(tokens) < self.n - 1:
            # Pick random start
            if self.ngrams:
                key = random.choice(list(self.ngrams.keys()))
                tokens = list(key)
            else:
                return {"text": "", "error": "No training data"}

        output = list(tokens)
        for _ in range(max_length):
            key = tuple(tokens[-(self.n - 1):]) if len(tokens) >= self.n - 1 else tuple(tokens)
            if key in self.ngrams and self.ngrams[key]:
                next_word = random.choice(self.ngrams[key])
                output.append(next_word)
                tokens.append(next_word)
            else:
                break

        return {
            "text": " ".join(output),
            "num_tokens": len(output),
            "seed": seed_text,
            "model": f"{self.n}-gram",
        }


# ---------------------------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------------------------

class NLPMetrics:
    """NLP evaluation metrics."""

    @staticmethod
    def bleu(reference: str, candidate: str, max_n: int = 4) -> Dict[str, Any]:
        """Compute BLEU score (simplified)."""
        ref_tokens = reference.lower().split()
        cand_tokens = candidate.lower().split()
        precisions = []

        for n in range(1, max_n + 1):
            ref_ngrams = collections.Counter(
                tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
            )
            cand_ngrams = collections.Counter(
                tuple(cand_tokens[i:i + n]) for i in range(len(cand_tokens) - n + 1)
            )
            matches = sum((cand_ngrams & ref_ngrams).values())
            total = max(sum(cand_ngrams.values()), 1)
            precisions.append(matches / total)

        if len(cand_tokens) < len(ref_tokens):
            bp = math.exp(1 - len(ref_tokens) / max(len(cand_tokens), 1))
        else:
            bp = 1.0

        if min(precisions) == 0:
            bleu_score = 0.0
        else:
            avg_log = sum(math.log(p) for p in precisions) / len(precisions)
            bleu_score = bp * math.exp(avg_log)

        return {
            "bleu": bleu_score,
            "precisions": precisions,
            "brevity_penalty": bp,
            "reference_length": len(ref_tokens),
            "candidate_length": len(cand_tokens),
        }

    @staticmethod
    def perplexity(probabilities: List[float]) -> float:
        """Compute perplexity from token probabilities."""
        if not probabilities:
            return float("inf")
        log_prob = sum(math.log(max(p, 1e-10)) for p in probabilities)
        return math.exp(-log_prob / len(probabilities))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def preprocess_text(text: str,
                           steps: Optional[List[str]] = None) -> Dict[str, Any]:
    """Preprocess text with configurable pipeline."""
    pp = TextPreprocessor()
    if steps:
        step_enums = []
        for s in steps:
            try:
                step_enums.append(TextPreprocessing(s))
            except ValueError:
                pass
    else:
        step_enums = None
    return pp.process(text, step_enums)


async def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze sentiment of text."""
    return SentimentAnalyzer.analyze(text)


async def analyze_sentiment_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Analyze sentiment for multiple texts."""
    return SentimentAnalyzer.analyze_batch(texts)


async def generate_embeddings(text: str,
                               model: str = "bert",
                               dimensions: int = 768) -> Dict[str, Any]:
    """Generate embeddings for text."""
    return EmbeddingGenerator.generate(text, model, dimensions)


async def compute_similarity(vec_a: List[float],
                              vec_b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    return EmbeddingGenerator.cosine_similarity(vec_a, vec_b)


async def extract_topics(documents: List[str],
                          num_topics: int = 5) -> Dict[str, Any]:
    """Extract topics from documents."""
    return TopicModeler.extract_topics(documents, num_topics)


async def generate_text(seed_text: str = "",
                         ngram: int = 3,
                         max_length: int = 20,
                         training_texts: Optional[List[str]] = None) -> Dict[str, Any]:
    """Generate text using n-gram model."""
    gen = TextGenerator(n=ngram)
    if training_texts:
        gen.train(training_texts)
    else:
        gen.train([
            "the quick brown fox jumps over the lazy dog",
            "machine learning is transforming the world of technology",
            "natural language processing enables computers to understand text",
        ])
    return gen.generate(seed_text, max_length)


async def compute_bleu(reference: str, candidate: str) -> Dict[str, Any]:
    """Compute BLEU score."""
    return NLPMetrics.bleu(reference, candidate)
