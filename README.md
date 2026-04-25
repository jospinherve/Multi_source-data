# PPMI Subtypes & Normalizing Flow Modeling

This repository contains the official code implementation for an end-to-end workflow modeling patient progression and disease subscores in Parkinson's disease, leveraging data from the PPMI cohort. 

The complete methodology is divided into three consecutive phases:
1. **Clinical Subtype Discovery** (SuStaIn algorithm)
2. **External Imaging Validation** (Longitudinal characterization)
3. **Deep Normalizing Flow Generative Modeling** (Conditional RealNVP)

## Repository Structure

- `run_ppmi_extended_pipeline.py`: **Phase 1** - The main script orchestrating the discovery of optimal clinical subtypes and disease staging from baseline clinical data using the SuStaIn framework.
- `characterize_clinical_subtypes_via_imaging.py`: **Phase 2** - The external validation script that correlates the discovered clinical subtypes with longitudinal imaging biomarkers (DaTscan, R2*) at future visits (M12, M24, M36, M48).
- `train_best_normalizing_flow.py`: **Phase 3** - The generative modeling script integrating both feature selection (via mutual information) and the Conditional RealNVP Normalizing Flow training. Includes early stopping, test evaluation (Log-Likelihood and BPD), and automatic visualization of latent spaces & generated samples.
- `config.yaml`: The configuration file directing architecture logic, scenario selection, preprocessing schemes, and optimization constants for the generative models.
- `requirements.txt`: The python package dependencies needed to run the environment.

*(Note: Raw and processed PPMI dataset files are kept private according to data usage agreements and are therefore not distributed directly in this repository.)*

## Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <YOUR-REPO-URL>
   cd <REPO-FOLDER>
   ```

2. **Create a virtual environment and install dependencies**:
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   # On Linux/macOS
   # source venv/bin/activate
   
   pip install -r requirements.txt
   ```

## Usage: The Full Workflow

### 1. Subtype Discovery (SuStaIn)
Run the clinical timeline modeling to discover the optimal number of subtypes and patient stages:
```bash
python run_ppmi_extended_pipeline.py
```
*(Outputs and trajectory timelines are saved in `outputs/sustain_results/`)*

### 2. External Validation (Imaging Markers)
Validate the discovered subtypes against longitudinal DaTscan and R2* physiological markers:
```bash
python characterize_clinical_subtypes_via_imaging.py
```
*(Statistical reports and longitudinal boxplots are saved in `outputs/imaging_longitudinal_validation/`)*

### 3. Generative Normalizing Flows
Initialize the conditional latent space and generate synthetic samples modeled upon your discovered subtypes:
```bash
python train_best_normalizing_flow.py \
    --config config.yaml \
    --data-root "/path/to/your/PPMI_Data" \
    --scenario "subscores" \
    --preprocessing "standard" \
    --seed 42
```