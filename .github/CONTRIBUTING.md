# Contributing to Tensorlink

🚀 **Welcome to Tensorlink**  
We’re building a peer-to-peer framework for running large-scale neural networks using PyTorch and Hugging Face. 
Tensorlink aims to democratize access to powerful models through compute-sharing, modular APIs, and privacy-preserving distributed training and inference.

Thanks for your interest, we welcome more contributors!

---

## 🧰 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/tensorlink.git
cd tensorlink
````

### 2. Set Up the Environment

We recommend using a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .[dev]
```

### 3. Run Tests

Before submitting a PR, ensure all tests pass:

```bash
pytest tests
```

```bash
pre-commit install
pre-commit run -a
```

---

## ✅ Contribution Guidelines

* **Use consistent code style.** Follow [PEP8](https://peps.python.org/pep-0008/) and run `black` and `flake8`.
* **Write meaningful commits.** Describe *why* the change is made.
* **Keep PRs focused.** Small, scoped changes are easier to review.
* **Document clearly.** Use docstrings and comments for clarity.
* **Test your code.** Add or update tests to cover your changes.

Working on compute nodes or model offloading?

* Use the `examples/` directory to simulate training and inference.
  * `distributed_example.py` demonstrates all node components working together.

---

## 📥 Submitting a Pull Request

1. Fork the repo
2. Create a branch
3. Push your changes
4. Open a PR on GitHub

Include:

* A brief summary of your change
* Related issue numbers (e.g., `Closes #42`)
* Any relevant logs, test output, or screenshots
