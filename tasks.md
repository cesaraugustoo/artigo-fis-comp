### **Phase 1: Project Foundation & Core Physics Correction**

This initial phase focuses on critical setup, correcting a major physics bug, and establishing the testing framework. These tasks are foundational and must be completed before any significant refactoring or feature development.

1.  **[High] Create `pyproject.toml` for dependency management.** (Effort: Low)
    * **Description**: Establish a formal project structure with a `pyproject.toml` file to manage all dependencies.
    * **Depends on**: None.
    * **Rationale**: This enables a consistent and reproducible development environment for all subsequent tasks.

2.  **[High] Correct the H+ Hamiltonian Implementation.** (Effort: Medium)
    * **Description**: The current implementation of the `H+` Hamiltonian incorrectly swaps the Rotating Wave Approximation (RWA) and counter-rotating terms. This must be corrected to ensure physical accuracy.
    * **Depends on**: None.
    * **Rationale**: This is a critical physics bug that invalidates simulation results. It must be fixed before any further validation or testing.

3.  **[High] Set up the `pytest` testing framework.** (Effort: Low)
    * **Description**: Configure `pytest` for the project and create the initial `tests/` directory structure.
    * **Depends on**: Task 1.
    * **Rationale**: Establishes the foundation for all future testing, ensuring that corrections and new features can be validated.

4.  **[High] Add Unit Tests for `HamiltonianBuilder` and Core Operators.** (Effort: Medium)
    * **Description**: Create unit tests that specifically verify the corrected `H+` and existing `H-` Hamiltonian terms (RWA and non-RWA). Also, verify the dimensions and hermiticity of all basic and composite operators.
    * **Depends on**: Task 2, Task 3.
    * **Rationale**: Locks in the physics correction with automated tests, preventing regressions and validating the core mathematical constructs.

---

### **Phase 2: Architecture Hardening & Standardization**

With the core physics corrected and a testing foundation in place, this phase focuses on improving the software architecture to reduce code duplication and enhance maintainability.

5.  **[High] Refactor `HamiltonianBuilder` to use Strategy Pattern.** (Effort: High)
    * **Description**: Decouple the Hamiltonian construction logic by refactoring the `HamiltonianBuilder` into a factory that uses `HPlusStrategy` and `HMinusStrategy` classes.
    * **Depends on**: Task 4.
    * **Rationale**: Improves code organization and makes it easier to add new Hamiltonians in the future without modifying the core builder.

6.  **[High] Refactor Simulation Solver to use Strategy Pattern.** (Effort: Medium)
    * **Description**: Decouple the `QuantumSimulator` from the specific `qutip.mesolve` implementation. Create a `QuTiPSolverStrategy` that wraps the `mesolve` call, allowing the simulator to accept different solver strategies.
    * **Depends on**: Task 3.
    * **Rationale**: Increases modularity and allows for future extensions, such as different numerical solvers or GPU-accelerated backends.

7.  **[Medium] Standardize Output File Naming Convention.** (Effort: Medium)
    * **Description**: Implement a single, unambiguous naming convention for all simulation output files (states, times, metrics).
    * **Depends on**: None.
    * **Rationale**: Simplifies data loading and removes the need for complex, fragile file-finding logic in downstream scripts. This is a prerequisite for centralizing data access.

8.  **[Medium] Refactor Data Loading Logic into a Central Utility.** (Effort: High)
    * **Description**: Create a centralized `SimulationDataLoader` utility in `utils.py` that handles finding and loading all data for a given simulation run. Refactor all CLI scripts (`plot_*.py`, `compare_results.py`, etc.) to use this new utility.
    * **Depends on**: Task 7.
    * **Rationale**: Drastically reduces code duplication, simplifies maintenance, and provides a single, reliable way to access simulation results.

---

### **Phase 3: Comprehensive Testing & Continuous Integration**

This phase expands the test suite to cover all major components and automates the testing process to ensure long-term code quality and stability.

9.  **[High] Add End-to-End Integration Test.** (Effort: Medium)
    * **Description**: Create an integration test that runs a complete, low-dimensional simulation, from configuration to output file generation, asserting that files are created and valid.
    * **Depends on**: Task 5, Task 6, Task 8.
    * **Rationale**: Verifies that the entire simulation pipeline works as expected after the major architectural refactoring.

10. **[Medium] Add Unit Tests for `StateGenerator`.** (Effort: Medium)
    * **Description**: Add tests to verify that all generated quantum states have a trace of 1 and that pure states have a purity of 1.
    * **Depends on**: Task 3.
    * **Rationale**: Ensures the physical validity of the initial conditions for all simulations.

11. **[Medium] Add Unit Tests for Metric Calculators.** (Effort: Medium)
    * **Description**: Add baseline tests for key metrics: Coherence for a pure state should be near zero, the R-parameter for a Fock state should be 1, and Wigner negativity should always be non-negative.
    * **Depends on**: Task 3.
    * **Rationale**: Validates the correctness of the physics analysis modules.

12. **[Medium] Set up a CI workflow using GitHub Actions.** (Effort: Medium)
    * **Description**: Create a CI pipeline that automatically installs dependencies, runs linters (`black`, `flake8`), and executes the full `pytest` suite on every push to the main branch.
    * **Depends on**: Task 1, Task 9.
    * **Rationale**: Automates quality checks and prevents regressions, ensuring the project remains in a healthy state.

---

### **Phase 4: Numerical Accuracy & Physics Validation**

This phase focuses on enhancing the numerical robustness of the simulation and providing tools for physicists to validate the results.

13. **[High] Implement Post-Hoc Hilbert Space Truncation Warning.** (Effort: Medium)
    * **Description**: After a simulation run, log a warning if the population in the highest-energy Fock state of either mode exceeds a defined threshold (e.g., 1e-4), suggesting that the Hilbert space may be too small.
    * **Depends on**: None.
    * **Rationale**: A critical feature to ensure the physical validity of results by detecting inadequate Hilbert space truncation.

14. **[Medium] Improve Wigner Negativity Integration Accuracy.** (Effort: Medium)
    * **Description**: Benchmark the current Riemann sum for Wigner negativity against `scipy.integrate.simpson`. Based on the results, update the calculation to use the more accurate method and make it configurable (`fast` vs. `accurate`).
    * **Depends on**: None.
    * **Rationale**: Improves the accuracy of a key non-classicality indicator.

15. **[Medium] Add Reproducibility Seed.** (Effort: Low)
    * **Description**: Add a `--seed` CLI argument that sets the `numpy` random seed to ensure any stochastic processes in the simulation are reproducible.
    * **Depends on**: None.
    * **Rationale**: Essential for scientific reproducibility and debugging.

16. **[Low] Add Lamb-Dicke Parameter (`eta`) Validation.** (Effort: Low)
    * **Description**: Add a check in the `SimulationConfig` to log a warning if `eta` is set to a value (e.g., > 0.3) that might compromise the validity of the Lamb-Dicke approximation.
    * **Depends on**: None.
    * **Rationale**: Helps prevent users from running simulations outside the valid physical regime.

---

### **Phase 5: API, Documentation & Final Polish**

The final phase focuses on usability, documentation, and minor cleanup to prepare the project for wider use.

17. **[Medium] Implement a High-Level Programmatic API.** (Effort: Medium)
    * **Description**: Create a new `src/api.py` module with a `Simulation` class that abstracts the setup and execution details, allowing a full simulation to be configured and run in just a few lines of code.
    * **Depends on**: Task 5, Task 6.
    * **Rationale**: Greatly improves the usability of the simulator as a library, beyond its CLI tools.

18. **[Medium] Enhance Docstrings and README with Physics Formulas.** (Effort: Medium)
    * **Description**: Update all public docstrings to NumPy style. Critically, update docstrings in physics-related modules (`hamiltonians`, `metrics`, `dissipation`) and the `README.md` to include the specific LaTeX equations being implemented. Correct the qubit basis definition and add the missing dephasing term to the master equation in the `README`.
    * **Depends on**: Task 2.
    * **Rationale**: Improves clarity, maintainability, and ensures a clear link between the code and the underlying physics.

19. **[Low] Remove Commented-Out Legacy Code.** (Effort: Low)
    * **Description**: Remove the large commented-out code blocks from `wigner_negativity.py` and `coherence.py`.
    * **Depends on**: None.
    * **Rationale**: Improves code readability. Version control (Git) is the proper place for legacy code.

20. **[Low] Restructure Project to Group Physics Models.** (Effort: Low)
    * **Description**: Move `operators.py`, `states.py`, `hamiltonians.py`, and `dissipation.py` into a new `src/model/` sub-package to better organize the code.
    * **Depends on**: Task 9.
    * **Rationale**: A minor architectural improvement that clarifies the separation between the physics model and the simulation framework.

---

### **Phase 6: Advanced & Future Work**

These tasks provide significant enhancements but are complex or lower priority. They can be addressed after the core project is stable and well-documented.

21. **[High] Create Hilbert Space Convergence Validation Script.** (Effort: Medium)
    * **Description**: Create a tool that runs a simulation for a range of `N_a` and `N_b` values and plots a key observable vs. `N` to help users determine the necessary truncation cutoff for their specific parameters.
    * **Depends on**: Task 13.
    * **Rationale**: Provides a crucial tool for ensuring the numerical convergence and physical accuracy of simulations.

22. **[Low] Integrate GPU Acceleration Support.** (Effort: High)
    * **Description**: Add a configuration flag (`use_gpu: true`) to enable the use of the `qutip-cupy` backend for `mesolve`.
    * **Depends on**: Task 6.
    * **Rationale**: A major performance enhancement for users with compatible NVIDIA hardware.