# Changelog

All notable changes to the AbAg Binding Prediction project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-10-31

### Added
- Initial release of AbAg binding affinity prediction model
- Deep learning model with Spearman ρ = 0.85 performance
- Support for antibody-antigen binding affinity prediction
- Pre-trained model (agab_phase2_model.pth)
- Comprehensive documentation in English and Korean
- Example usage scripts
- Installation and quick start guides
- MIT License

### Features
- AffinityPredictor class for easy model usage
- Support for both single predictions and batch processing
- Model performance metrics and validation results
- Python package setup for pip installation

---

## How to Version Future Releases

When making changes, follow this pattern:

### Version Numbering (Semantic Versioning)
- **MAJOR version (X.0.0)**: Breaking changes, incompatible API changes
- **MINOR version (1.X.0)**: New features, backward-compatible
- **PATCH version (1.0.X)**: Bug fixes, backward-compatible

### Steps for New Releases

1. **Update version numbers:**
   - `setup.py`: Update `version="X.X.X"`
   - `abag_affinity/__init__.py`: Update `__version__ = "X.X.X"`

2. **Update this CHANGELOG.md:**
   - Add new section with version and date
   - Document all changes under Added/Changed/Fixed/Removed

3. **Commit changes:**
   ```bash
   git add setup.py abag_affinity/__init__.py CHANGELOG.md
   git commit -m "Bump version to X.X.X"
   ```

4. **Create git tag:**
   ```bash
   git tag -a vX.X.X -m "Release vX.X.X: Brief description"
   ```

5. **Push to GitHub:**
   ```bash
   git push origin main
   git push origin vX.X.X
   ```

### Example Entry for Future Version

```markdown
## [1.1.0] - YYYY-MM-DD

### Added
- New feature A
- New feature B

### Changed
- Improved performance of X
- Updated dependencies

### Fixed
- Bug fix for issue #123
- Corrected error in Y

### Removed
- Deprecated function Z
```
