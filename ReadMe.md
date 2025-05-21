# FabriCAD Dataset

## Overview

**FabriCAD** is a fully synthetic dataset designed to support research in AI-assisted manufacturing approaches. It pairs detailed manufacturing process plans with corresponding parametric CAD data to provide a rich resource for machine learning workflows in industrial manufacturing.

The dataset focuses on representing realistic manufacturing processes in metalworking by generating sequences derived from Markovchains extracted from +50,000 real-world metalworking work plans. It provides detailed, multi-step process plans and feature-level CAD information to facilitate granular analysis and model training.

This dataset is intended for:

- Research in AI-assisted manufacturing and process automation.
- Development and benchmarking of machine learning models for manufacturing process prediction and optimization.
- Educational and prototyping purposes within industrial engineering and manufacturing informatics.


---

## Key Features

- **Parametric CAD models (STEP format)**: The dataset includes 3D CAD files representing the manufactured parts.
- **Structured multi-step process plans**: Each sample contains a detailed workflow with steps like milling, drilling, welding, and grinding.
- **Human-readable and machine-readable data**: Process plans and metadata are provided in CSV format.
- **Feature-level details**: Detailed information is available on a per-feature basis, including tooling, cutting parameters, and process-specific annotations.
- **Synthetic yet realistic**: Based on Markov chains derived from 50,000 real work plans, ensuring the dataset reflects plausible manufacturing sequences.
- **Suitable for ML research**: The dataset supports machine learning approaches in manufacturing process prediction, optimization, and automation.

---

## Dataset Composition

Each sample primarily consists of:

- A **manufacturing process plan**, detailing:
  - Process steps
  - Workstations
  - Duration and cost estimates
  - Feature-level metadata (e.g., type of hole, thread details, tooling)
- A **corresponding parametric 3D CAD model** in STEP format.

Note: To save space, the main dataset stores only the final CAD models. If you require more granular CAD process data or feature-level changes, please reach out at us.

---

## Manufacturing Process Coverage

The generated parts and plans currently cover metalworking operations including:

- **Geometry-affecting steps:**
  - Milling
  - Drilling
  - Welding
  - Grinding

- **Additional process steps:**
  - Deburring
  - Marking
  - Delivery
  - Inspection
  - Testing

---


## Limitations & Disclaimer

- All data is **synthetic and simplified**.
- The dataset is **not suitable for productive or industrial use**.
- Parts are generated following rules enforcing manufacturability, realistic wall thicknesses, and plausible placements.
- Due to the large scale, some unrealistic configurations may occur.
- Use the dataset responsibly, and note that it is primarily designed for research and testing.

---

## How to Get Started

1. Download the preview data from the Repository or the io-page [GitHub repository](https://cimtt-kiel.github.io/FabriCAD/).
2. Explore the process plan CSV files and the STEP CAD-models. If you want you can use the streamlit app to visualize the data. For that u cad setup an environment via:
```
uv venv
uv pip install -e
```

Then run the Streamlit-App via (use autocomplete for the home emoji):
```
streamlit run Home.py
```
3. Contact us if you need more detailed CAD process data or additional features.

---

## Contact

For questions, collaborations, or requests for extended data, please reach out:

**Prof. Dr.-Ing. Daniel Böhnke**
Email: daniel.boehnke@fh-kiel.de

**Michel Kruse**  
Email: michel.kruse@fh-kiel.de

---

## License

This dataset is provided for research purposes under MIT License. Please cite this page or the publication if you use this dataset.

---

## Acknowledgements

We would like to thank the industrial partner who enabled the access to the real-world process plans used to calculate the Markovchains.

---

*FabriCAD Project — Created by Cimtt, FH Kiel — 2025*
