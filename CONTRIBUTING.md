# Contributing to AbAg Binding Affinity Prediction

First off, thank you for considering contributing to this project! 🎉

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues. When you create a bug report, include as many details as possible:

- **Use a clear and descriptive title**
- **Describe the exact steps to reproduce the problem**
- **Provide specific examples** (code snippets, data samples)
- **Describe the behavior you observed** and what you expected
- **Include your environment details** (OS, Python version, PyTorch version)

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:

- **Use a clear and descriptive title**
- **Provide a step-by-step description** of the suggested enhancement
- **Explain why this enhancement would be useful**
- **List some examples** of how it would be used

### Pull Requests

1. Fork the repo and create your branch from `main`
2. If you've added code, add tests
3. If you've changed APIs, update the documentation
4. Ensure the test suite passes
5. Make sure your code follows the existing style
6. Issue the pull request!

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/AbAg_binding_prediction.git
cd AbAg_binding_prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements.txt

# Install development dependencies
pip install pytest black flake8 mypy
```

## Code Style

- Follow PEP 8 style guidelines
- Use `black` for code formatting: `black .`
- Use `flake8` for linting: `flake8 .`
- Add type hints where possible
- Write docstrings for all public functions/classes

## Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_installation.py

# Run with coverage
pytest --cov=abag_affinity tests/
```

## Documentation

- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update relevant documentation in docs/
- Add examples for new features

## Commit Messages

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters or less
- Reference issues and pull requests liberally after the first line

Examples:
```
Add GELU activation option to model
Fix memory leak in embedding generation
Update documentation for v2 improvements
```

## Areas We Need Help

- 📊 **Dataset Integration** - Adding more antibody-antigen databases
- 🏗️ **Model Improvements** - New architectures, loss functions
- 📝 **Documentation** - Tutorials, examples, guides
- 🧪 **Testing** - Unit tests, integration tests
- 🐛 **Bug Fixes** - Fixing reported issues
- ⚡ **Performance** - Optimization and speedups

## Questions?

Feel free to open an issue with the "question" label or reach out to the maintainers.

Thank you for contributing! 🙌
