# Research Versions Archive

This directory serves as a chronological ledger of the experimental phases and models tested in the NewsLies project. The active, deployment-ready code is maintained at the repository root and `docs/` folder.

## v1_lstm_baseline
The initial implementation phase. Features an LSTM model built on a small, custom-stemmed Arabic vocabulary.
- Discovered catastrophic source-leakage (the model was essentially learning to map publisher names to credibility, ignoring semantics).
- Contains the legacy `docs/` files including the 75MB PyTorch model converted to `model.onnx`.

## v2_arabert_hf
The second iteration upgrading to `aubmindlab/bert-base-arabertv02`.
- Validated the source-leakage hypothesis by achieving 87% F1 on random splits but plummeting to 38% F1 on disjoint (unseen source) splits.
- Modernized the web deployment to use Hugging Face's official `@xenova/transformers` library running native WordPiece tokenization in the browser.
- Deployed a highly optimized ONNX version of the AraBERT model directly from the Hugging Face Hub, shrinking the GitHub repository size.
