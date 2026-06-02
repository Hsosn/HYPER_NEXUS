# NLP Workbench — Advanced Level

**Level:** Advanced  
**Category:** AI/ML

## Overview
Natural language processing tools with real text processing, tokenization, and embedding generation capabilities.

## Capabilities
- **Text Classification**: RNN, BiLSTM, CNN-text, Transformer architectures
- **Sentiment Analysis**: Real text scoring pipeline
- **Tokenization**: Pure Python word-level tokenizer (compatible with nn.Embedding)
- **Embeddings**: HuggingFace, sentence-transformers, word2vec patterns
- **NER**: Named Entity Recognition with HuggingFace pipeline integration

## Usage
```bash
python nlp_workbench.py classify --model transformer
python nlp_workbench.py sentiment --texts "This is great!" "This is terrible"
python nlp_workbench.py tokenize --texts "Hello world" --vocab-size 30000
python nlp_workbench.py embed --texts "ML is amazing" --method sentence_transformers
python nlp_workbench.py ner --text "Apple Inc. is based in Cupertino"
```

## Key Classes
| Class | Description |
|-------|-------------|
| `SimpleTokenizer` | Pure Python tokenizer |
| `pad_sequences` | Sequence padding |
