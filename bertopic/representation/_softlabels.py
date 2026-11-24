import numpy as np
import torch

from scipy.sparse import csr_matrix
from bertopic.representation._base import BaseRepresentation
from bertopic._topics import Keywords
from bertopic._corpus import Corpus

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bertopic import BERTopic

DEFAULT_PROMPT = "Generate a label that captures the overall theme of the document below.\n\nDocument: [DOCUMENT]\n\nTopic:"


class SoftLabels(BaseRepresentation):
    def __init__(
        self,
        model: str = "baidu/ERNIE-4.5-0.3B-PT",
        top_n_words: int = 10,
        nr_samples: int = 100,
        temperature: float = 3.0,
        prompt: str | None = None,
        batch_size: int = 16,
        max_length: int = 512,
        device: str | None = None,
    ):
        """Describe topics with the words a small language model expects as their label.

        c-TF-IDF can only rank words that occur in a topic's documents. This model
        instead asks a causal language model, for each sampled document, to label
        its theme, and reads the next-token distribution at that point rather than
        the generated text. That distribution, restricted to the vectorizer's
        vocabulary, is a soft label: it gives weight to theme words such as
        "hockey" even in documents that never use them (Li et al., 2026).

        The soft labels of a topic's documents are averaged. A word is then scored by
        its weight in the topic times the log-ratio to its average weight over all
        topics, so that words that are likely for every topic are pushed down.

        Only alphabetic vocabulary words that the language model represents with a
        single token are scored. No text is generated, so one forward pass per document
        is all that is needed and a model with a few hundred million parameters works well.

        Arguments:
            model: A causal language model on the Hugging Face Hub, or a tuple of
                   an already loaded `(model, tokenizer)`.
            top_n_words: The top n words to extract per topic.
            nr_samples: The number of documents to sample per topic.
            temperature: The softmax temperature of the soft labels. Values above 1
                         spread weight over more words.
            prompt: The prompt, in which `[DOCUMENT]` is replaced by the document.
                    It should end where the label would start.
            batch_size: The number of documents per forward pass.
            max_length: Prompts with more tokens have their document shortened.
            device: The device to run the language model on. Defaults to CUDA when available.

        Usage:

        ```python
        from bertopic.representation import SoftLabels
        from bertopic import BERTopic

        # Create your representation model
        representation_model = SoftLabels("baidu/ERNIE-4.5-0.3B-PT")

        # Use the representation model in BERTopic on top of the default pipeline
        topic_model = BERTopic(representation_model=representation_model)
        ```
        """
        self.model = model
        self.top_n_words = top_n_words
        self.nr_samples = nr_samples
        self.temperature = temperature
        self.prompt = prompt if prompt is not None else DEFAULT_PROMPT
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self._lm = None
        self._tokenizer = None

    def extract_topics(
        self,
        topic_model: "BERTopic",
        corpus: Corpus,
        topic_representations: dict[int, Keywords],
        c_tf_idf: csr_matrix,
        embeddings: np.ndarray | None = None,
    ) -> dict[int, Keywords]:
        """Extract topics.

        Arguments:
            topic_model: A BERTopic model
            corpus: Documents and their metadata.
            topic_representations: A dictionary mapping topic IDs to their
                                   Keywords representations as calculated by c-TF-IDF.
            c_tf_idf: The topic c-TF-IDF representation
            embeddings: Pre-trained document embeddings (not used)

        Returns:
            updated_topics: Updated topic representations
        """
        self._load()
        words, token_ids = self._single_token_vocabulary(topic_model.vectorizer_model.get_feature_names_out())
        topic_ids = list(topic_representations.keys())

        # Average the soft labels of the sampled documents of each topic
        mass = np.zeros((len(topic_ids), len(words)))
        for row, topic_id in enumerate(topic_ids):
            documents = corpus.get_topic(topic_id, nr_samples=self.nr_samples, with_text=True).documents
            if documents:
                mass[row] = self._soft_labels(documents, token_ids).mean(axis=0)

        # Keep what sets a topic apart: weight each word by how much more likely it is
        # for this topic than on average over the topics
        average = np.maximum(mass[mass.sum(axis=1) > 0].mean(axis=0), 1e-12)
        scores = mass * np.log(np.maximum(mass, 1e-12) / average)

        updated_topics = {}
        for row, topic_id in enumerate(topic_ids):
            if mass[row].sum() == 0:
                updated_topics[topic_id] = Keywords(topic_representations[topic_id].top_n(self.top_n_words))
                continue
            top = np.argsort(-scores[row])[: self.top_n_words]
            updated_topics[topic_id] = Keywords([(words[index], float(scores[row, index])) for index in top])
        return updated_topics

    def _load(self):
        """Load the language model on first use."""
        if self._lm is not None:
            return
        if isinstance(self.model, str):
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            self._tokenizer = AutoTokenizer.from_pretrained(self.model)
            self._lm = AutoModelForCausalLM.from_pretrained(self.model, dtype=dtype).eval().to(device)
        else:
            self._lm, self._tokenizer = self.model
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # Left padding keeps the last real token at position -1 for every row
        self._tokenizer.padding_side = "left"

    def _single_token_vocabulary(self, vocabulary) -> tuple[list[str], torch.Tensor]:
        """Keep the alphabetic vocabulary words that are one token when they follow a space."""
        words, token_ids = [], []
        for word in vocabulary:
            pieces = self._tokenizer.encode(f" {word}", add_special_tokens=False)
            if len(pieces) == 1 and word.isalpha() and len(word) > 2:
                words.append(word)
                token_ids.append(pieces[0])
        if not words:
            raise ValueError("No word in the vocabulary is a single token for this language model.")
        return words, torch.as_tensor(token_ids, device=self._lm.device)

    def _create_prompt(self, document: str) -> str:
        """Fill in the prompt, shortening the document if the prompt is too long."""
        prompt = self.prompt.replace("[DOCUMENT]", document).rstrip()
        if len(self._tokenizer.encode(prompt)) > self.max_length:
            overhead = len(self._tokenizer.encode(self.prompt.replace("[DOCUMENT]", "").rstrip()))
            tokens = self._tokenizer.encode(document, add_special_tokens=False)
            tokens = tokens[: max(self.max_length - overhead - 2, 1)]
            document = self._tokenizer.decode(tokens, skip_special_tokens=True)
            prompt = self.prompt.replace("[DOCUMENT]", document).rstrip()
        return prompt

    @torch.no_grad()
    def _soft_labels(self, documents: list[str], token_ids: torch.Tensor) -> np.ndarray:
        """The next-token distribution over the vocabulary after each document's prompt."""
        prompts = [self._create_prompt(document) for document in documents]
        labels = []
        for start in range(0, len(prompts), self.batch_size):
            batch = self._tokenizer(
                prompts[start : start + self.batch_size], return_tensors="pt", padding=True
            )
            logits = self._lm(
                input_ids=batch["input_ids"].to(self._lm.device),
                attention_mask=batch["attention_mask"].to(self._lm.device),
                use_cache=False,
            ).logits[:, -1, :]
            logits = logits.index_select(-1, token_ids).float() / self.temperature
            labels.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(labels, axis=0)
