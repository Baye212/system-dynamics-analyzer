# System Dynamics Analyzer — Project Guidelines

## Code Style

- **Type hints**: Comprehensive throughout. Use `Tuple`, `Dict`, `Union`, `Callable` from `typing`. NumPy arrays typed as `np.ndarray`.
- **LaTeX formatting**: Heavy LaTeX integration for mathematical expressions. Format eigenvalues/parameters as `f"$\\lambda_1 = {:.3f}$"` for Streamlit rendering.
- **Emoji indicators**: Use 🟢 (stable), 🔴 (unstable/node), 🟡 (center) for classification; ⭐ (exemplary file), ❌ (incomplete), ⚠️ (gotcha).
- **Language**: Primary docs in French (see [README.md](README.md)); code comments bilingual where helpful for English-only readers.
- **Method documentation**: Include docstrings with numpy-style parameter/return types; many methods populate `self.report` dict for state.

## Architecture

**Monolithic Core Pattern**:

- `src/dynamics_analyzer.py`: Main `DynamicsAnalyzer` class handles 6-step pedagogical pipeline:
  1. Eigenvalue analysis & classification
  2. Eigenspace computation (stable/unstable/center)
  3. Isocline plotting (∂x/∂t=0, ∂y/∂t=0)
  4. Quadrant flux analysis
  5. Phase portrait with trajectories
  6. PDF report generation

**Dual System Support** (Linear + Nonlinear):

- Constructor accepts: `np.ndarray` (2×2 matrix for `ẋ=Ax`) OR `Tuple[Callable, Callable]` (functions for nonlinear).
- Methods degrade gracefully (nonlinear raises `ValueError` for eigenvalue analysis).

**UI Layer Separation**:

- `app.py`: Streamlit web interface (primary; formatted results with LaTeX).
- `main_interface.py`: CLI alternative (step-by-step prompts).
- `system_input.py`: Input parsing utilities.

**⚠️ Critical**: Placeholder modules (`linear_analyzer.py`, `nonlinear_analyzer.py`, `stability.py`, `phase_plotter.py`, `utils.py`) are empty scaffolds—do not assume they're active.

See [methodology.md](docs/methodology.md) for mathematical foundations.

## Build and Test

**Start Local**:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Logs: `http://localhost:8501`

**Deploy** (Streamlit Cloud):

1. Push to GitHub repo: `AZ473/system-dynamics-analyzer`
2. Go to `share.streamlit.io` → Deploy → select this repo, main file `app.py`

**Testing**:

- Test files exist but are **empty** (`tests/test_linear_analyzer.py`, etc.) — no CI/CD configured.
- Run manually: `pytest tests/` (once tests are written).
- **Priority**: Add tests for eigenvalue classification and trajectory integration before refactoring.

## Conventions

### 1. **Attribute State Management**

Methods populate `self.report` dict with results (e.g., `self.report['eigen']`, `self.report['eigenspaces']`, `self.report['isoclines']`). This enables multi-step analysis without tight coupling. Always return results _and_ update `self.report` for consistency.

### 2. **Numerical Operations**

- Use `np.linalg.eig(A)` for eigendecomposition (returns eigenvalues, eigenvectors).
- Use `scipy.integrate.solve_ivp(..., method='RK45')` for trajectory integration.
- Eigenvalue precision: Uses `np.isclose(val.imag, 0, atol=1e-8)` to classify real vs. complex. **Be cautious with borderline cases** (multiplicity, near-zero imaginary parts).

### 3. **Matplotlib Figure Isolation**

- Streamlit can reuse figure state across runs. Use `plt.close()` after `plt.show()` in non-Streamlit contexts.
- When `plot=False` is passed, ensure figures are created but not displayed (don't assume matplotlib state is clean).

### 4. **Stability Classification**

Current node types: nœud stable, nœud instable, selle, centre, spirale stable, spirale instable.  
**Known gap**: "Foyer-selle" (saddle-focus) documented but not classified in code. Watch eigenvalue trace/determinant edge cases.

### 5. **Input Validation**

- Assume 2D systems (2×2 matrix, (x, y) pairs) — **no validation** for higher dimensions.
- ⚠️ **Security**: `system_input.py` uses `eval()` for expression parsing. Streamlit UI is safe (server-side), but CLI interface is vulnerable to code injection. Do not expose CLI to untrusted input.

### 6. **Report PDF Generation**

- Uses matplotlib backend to render phase portraits → PDF via reportlab or similar.
- Ensure all plots call `plt.close()` after rendering to PDF to avoid figure bleed-through in subsequent steps.

### 7. **Code Duplication Flag**

- ⚠️ Two `dynamics_analyzer.py` files exist:
  - Root-level (simplified/mockup version)
  - `src/dynamics_analyzer.py` (full implementation)
  - **Use `src/` version exclusively**. Root-level is deprecated; delete or rename after audit.

## Common Pitfalls

1. **Editing root-level `dynamics_analyzer.py` instead of `src/`** → Changes lost; app still uses src version.
2. **Matplotlib figure state** → Running tests followed by Streamlit may show stale plots. Always `plt.close()`.
3. **Nonlinear eigenvalue access** → Will raise `ValueError`. Check system type first or wrap with try-except.
4. **PDF generation on Windows** → Ensure reportlab + system fonts are compatible; test locally before deployment.
5. **Missing test coverage** → No regression protection when refactoring. Prioritize tests for `src/dynamics_analyzer.py` methods.

## Key Files

- [src/dynamics_analyzer.py](src/dynamics_analyzer.py) ⭐ — Core implementation; exemplary 6-step pipeline and report dict pattern.
- [app.py](app.py) ⭐ — Streamlit UI; shows formatting helpers (`format_eigenvalues_display`) and LaTeX rendering.
- [README.md](README.md) — Project overview, deployment guide.
- [docs/methodology.md](docs/methodology.md) — Mathematical background (link only; don't embed here).

## Getting Started (New Contributors)

1. **Setup**: Clone, `pip install -r requirements.txt`, `streamlit run app.py`.
2. **Read**: [methodology.md](docs/methodology.md) for math; [src/dynamics_analyzer.py](src/dynamics_analyzer.py) for code structure.
3. **Run examples**: Try [examples/linear_systems.py](examples/linear_systems.py) or predefined examples in Streamlit UI.
4. **Avoid**: Editing root `dynamics_analyzer.py` or CLI (`main_interface.py`) without understanding Streamlit priority.
5. **Test**: Before adding features, add a test case to `tests/` (currently empty; use pytest format).
