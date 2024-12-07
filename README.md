
### Main Files

- **blended_measurement.py**: Functions for generating blended measurement operators.
- **circuit.py**: Functions for constructing quantum circuits.
- **classical_shadow.py**: Implementation of classical shadow techniques.
- **event_learning_fuc.py**: Functions for quantum event identification and finding.
- **generate_haar_random_povm.py**: Script to generate and save Haar random POVM sets.
- **generate_test_script.py**: Script to generate test configurations.
- **plot.py**: Functions for plotting results.
- **povm_generate_function.py**: Functions for generating POVM elements.
- **run_test_script.py**: Script to run experiments based on test configurations.
- **sample.py**: Functions for sampling POVM elements.
- **tools.py**: Utility functions used across the project.
- **vector_generating_function.py**: Functions for generating vectors used in POVM generation.

## Installation

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Create a virtual environment**:
    ```bash
    python -m venv env
    ```

3. **Activate the virtual environment**:
    ```bash
    source env/bin/activate  # For Linux/MacOS
    env\Scripts\activate     # For Windows
    ```

4. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Create necessary directories**:
    ```bash
    mkdir -p Haar_measurement_dir result_json result_plot
    ```

## Usage

### Generating Test Scripts

To generate test scripts, run:
```bash
python generate_test_script.py --dimension 8 --m_s 4 --gate_num_times 1 --method_s random blended --copies_s 1 2 --rank_s '{"8": [4]}' --file_path test_config --case_1_high 0.9 --case_1_low 0.1 --case_2_pro 0.01 --test_time 100 --average_time 20 --case 2

