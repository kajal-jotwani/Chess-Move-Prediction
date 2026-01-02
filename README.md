# Chess Move Prediction

A small, hand-built notebook that learns to predict the next chess move from a FEN. I kept it simple on purpose — the notebook parses a games CSV, converts positions to 8×8×12 binary planes, trains a Keras ConvNet and shows quick inference examples.

What's in this repo
- chessPaa.ipynb — the working notebook (data parsing, featurization, model, evaluation, inference).
- games.csv — the game export used by the notebook.

Quick start
1. Install the basics:
   pip install numpy pandas python-chess tensorflow tqdm
2. Place your games CSV at the repo root as games.csv (or edit CSV_PATH in the notebook).
3. Open chessPaa.ipynb in Jupyter or Colab and run the cells.

What I did in the notebook
- Encoding: 12 binary planes (piece × color) from FEN (see function fen_to_planes).
- Parsing: supports SAN or UCI move strings (san_moves_to_uci_list).
- Training setup used: TOP_N = 256 (vocab), MAX_EXAMPLES cap, EPOCHS = 12, BATCH_SIZE = 64.
- Model: small Conv2D → Dense → softmax (Keras).
- Example results from the run: Top-1 ≈ 14.6%, Top-3 ≈ 28.3%.
- Quick helper: predict_next_moves_from_sequence(move_sequence, top_k=3)

Files to look at first
- chessPaa.ipynb — follow the code cells from data load → preprocess → model → predict.
- games.csv — the raw input the notebook expects.

References / reading I used
- Predicting Moves in Chess using Convolutional Networks — https://www.semanticscholar.org/paper/Predicting-Moves-in-Chess-using-Convolutional-Oshri/28a9fff7208256de548c273e96487d750137c31d?p2df
- Predicting professional players' chess moves with deep learning — https://medium.com/data-science/predicting-professional-players-chess-moves-with-deep-learning-9de6e305109e

Contributing
- If you want to improve this, please fork and open a PR.
- Small, focused changes are easiest (scripts to replace notebook cells, CLI/train/predict scripts, tests, a small sample CSV).
- Add tests for parsing and fen_to_planes if you change preprocessing.
- Open an issue for bigger changes or to discuss experiments before implementing.
- License: choose one (MIT recommended) and add a LICENSE file.

Acknowledgements
- python-chess for board utilities
- public game datasets (e.g., Lichess) that make experiments like this possible
