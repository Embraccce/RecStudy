------
# RecStudy: A Comprehensive Study of Recommender Systems

Welcome to **RecStudy**, an introductory journey for beginners in recommender systems. This project covers various techniques, from basic collaborative filtering to deep learning algorithms and advanced graph-based and knowledge graph methods.

The primary reference for this project is Professor Yu Fangren's book, [*"Hands-on Learning of Recommender Systems: Algorithm Implementation Based on PyTorch"*](https://book.douban.com/subject/36160038/). A significant portion of the code is inspired by the content of the book and the  [**author's code repository**](https://github.com/rexrex9/). Special thanks to Professor Yu Fangren for his invaluable contributions!

To facilitate learning for beginners, all code is thoroughly commented, well-organized, and ready to run without any adjustments. You can directly execute the code and see the final results. If you encounter any issues, feel free to raise them in the Issues section!

Below is the directory structure and a detailed description of each chapter.

------

## Project Structure

```plaintext
RecStudy/
├─ dataset/                        # Datasets used in the project
│  ├─ ml-100k                      # MovieLens 100K dataset
│  ├─ ml-100k-original             # Original MovieLens 100K dataset
│  └─ ml-latest-small              # MovieLens Latest Small dataset
├─ Part1.BasicRecAlgorithms/       # Basic recommender system algorithms
├─ Part2.AdvancedRecAlgorithms/    # Advanced deep learning-based algorithms
├─ Part3.GraphRecAlgorithms/       # Graph-based recommender algorithms
├─ Part4.KGRecAlgorithms/          # Knowledge graph recommender algorithms
├─ Part5.ConstructRecSys/          # Recommender system construction techniques
├─ Part6.EvaRecSys/                # Evaluation methods for recommender systems
└─ utils/                          # Utility scripts
```

------

## Chapters Overview

### **Part 1: Basic Recommender Systems**

This chapter introduces foundational algorithms and techniques for recommender systems, including matrix factorization, collaborative filtering, and similarity-based methods.

| File                                                         | Description                                                  |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| [`ALS_PyTorch.py`](Part1.BasicRecAlgorithms/ALS_PyTorch.py)         | Alternating Least Squares (ALS) implemented in PyTorch.      |
| [`ALS_tradition.py`](Part1.BasicRecAlgorithms/ALS_tradition.py)     | ALS implemented in NumPy.                                    |
| [`basicSim.py`](Part1.BasicRecAlgorithms/basicSim.py)               | Basic similarity computation for user-item pairs.            |
| [`dataloader.py`](Part1.BasicRecAlgorithms/dataloader.py)           | General data loader for recommender systems.                 |
| [`dataloader4ml100kIndexs.py`](Part1.BasicRecAlgorithms/dataloader4ml100kIndexs.py) | Data loader for MovieLens 100K with indexed features.        |
| [`dataloader4ml100kOneHot.py`](Part1.BasicRecAlgorithms/dataloader4ml100kOneHot.py) | Data loader for MovieLens 100K with one-hot encoding.        |
| [`evaluate.py`](Part1.BasicRecAlgorithms/evaluate.py)               | Evaluation metrics for recommender systems.                  |
| [`FM.py`](Part1.BasicRecAlgorithms/FM.py)                           | Factorization Machines (FM) implementation.                  |
| [`FM_embbeding_style.py`](Part1.BasicRecAlgorithms/FM_embbeding_style.py) | FM with embedding style implementation.                      |
| [`furtherSim.py`](Part1.BasicRecAlgorithms/furtherSim.py)           | Extensions of similarity computation.                        |
| [`itemCF_01label.py`](Part1.BasicRecAlgorithms/itemCF_01label.py)   | Item-based collaborative filtering with binary labels.       |
| [`LR.py`](Part1.BasicRecAlgorithms/LR.py)                           | Logistic Regression-based recommendation.                    |
| [`POLY2.py`](Part1.BasicRecAlgorithms/POLY2.py)                     | Polynomial-based recommendation model.                       |
| [`SVD.ipynb`](Part1.BasicRecAlgorithms/SVD.ipynb)                   | Singular Value Decomposition (SVD) for recommendations.      |
| [`userCF_01label.py`](Part1.BasicRecAlgorithms/userCF_01label.py)   | User-based collaborative filtering with binary labels.       |
| [`userItemCF_15label.ipynb`](Part1.BasicRecAlgorithms/userItemCF_15label.ipynb) | Hybrid user-item collaborative filtering with detailed labels. |


------

### **Part 2: Advanced Recommender Systems**

This chapter explores deep learning-based approaches, such as neural collaborative filtering, embedding methods, and sequence-based models.

| File                  | Description                                           |
| --------------------- | ----------------------------------------------------- |
| [`ALS_MLP.ipynb`](Part2.AdvancedRecAlgorithms/ALS_MLP.ipynb)       | Neural Collaborative Filtering (NCF) with MLP.        |
| [`Embedding_CNN.ipynb`](Part2.AdvancedRecAlgorithms/Embedding_CNN.ipynb)  | Embedding-based recommendation using CNNs.            |
| [`Embedding_MLP.ipynb`](Part2.AdvancedRecAlgorithms/Embedding_MLP.ipynb)  | Embedding-based recommendation using MLP.             |
| [`FNN_plus.ipynb`](Part2.AdvancedRecAlgorithms/FNN_plus.ipynb)       | Feedforward Neural Networks with additional features. |
| [`RNN_rec.ipynb`](Part2.AdvancedRecAlgorithms/RNN_rec.ipynb)        | Sequential recommendation using RNNs.                 |
| [`RNN_rec_ALS.ipynb`](Part2.AdvancedRecAlgorithms/RNN_rec_ALS.ipynb)    | Combining RNNs with ALS for hybrid recommendations.   |

------

### **Part 3: Graph-Based Recommender Systems**

In this chapter, we introduce graph neural networks (GNNs) and other graph-based techniques for recommendations.
 *(Files in this directory to be updated.)*
| File                  | Description                                           |
| --------------------- | ----------------------------------------------------- |
------

### **Part 4: Knowledge Graph Recommender Systems**

This chapter focuses on leveraging knowledge graphs to enhance recommendation quality.
 *(Files in this directory to be updated.)*
| File                  | Description                                           |
| --------------------- | ----------------------------------------------------- |
------

### **Part 5: Constructing Recommender Systems**

This section delves into building end-to-end recommender systems, including system design and implementation.
 *(Files in this directory to be updated.)*
| File                  | Description                                           |
| --------------------- | ----------------------------------------------------- |
------

### **Part 6: Evaluating Recommender Systems**

This chapter discusses evaluation techniques and metrics for recommender systems, including accuracy and diversity measures.
 *(Files in this directory to be updated.)*
| File                  | Description                                           |
| --------------------- | ----------------------------------------------------- |
------

## Dataset

The project uses several datasets, primarily the [**MovieLens**](https://files.grouplens.org/datasets/movielens/) dataset, with different versions organized as follows:

- `ml-100k`: Preprocessed MovieLens 100K dataset.
- `ml-100k-original`: Original MovieLens 100K dataset.
- `ml-latest-small`: Latest small version of MovieLens.

------

## Utilities

Additional scripts and helper functions are stored in the `utils/` directory. These include data preprocessing and utility scripts to facilitate the project.
| File                  | Description                                           |
| --------------------- | ----------------------------------------------------- |
------

## Getting Started

To start exploring the code and running experiments:

1. Clone the repository:

   ```bash
   git clone https://github.com/Embraccce/RecStudy.git
   cd RecStudy
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run examples or experiment scripts from respective directories.

------

## Contributing

We welcome contributions to improve and expand this repository. Feel free to submit issues or pull requests.

------

## License

This project is licensed under the [MIT License](https://opensource.org/license/MIT).

------

