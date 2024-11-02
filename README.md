# Paper Implementations

This repository contains implementations of various research papers using PyTorch. The goal is to familiarize with PyTorch by replicating results from different papers.

## Repository Structure

- `src/`: Shared source code for training, utilities, etc.
- `papers/`: Each paper has its own directory containing specific implementations, notebooks, and results.

## Getting Started

### Prerequisites

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/BertilBraun/Paper-Implementations.git
    cd Paper-Implementations
    ```

2. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

### Running the Code

1. Navigate to the specific paper directory:

    ```sh
    cd papers/paper1
    ```

2. Follow the instructions in the paper's `README.md` to run the implementation.

### Adding a New Paper

To add a new paper implementation to this repository, run the `create_paper.py` script:

```sh
python create_paper.py
```

It will prompt you to enter the paper's title and URL. The script will create a new directory in the `papers/` folder with the necessary structure.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
