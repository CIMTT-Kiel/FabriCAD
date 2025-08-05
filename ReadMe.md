# FabriCAD

A fully synthetic dataset designed to support research in AI-assisted manufacturing approaches. FabriCAD pairs detailed manufacturing process plans with corresponding parametric CAD data to provide a rich resource for machine learning workflows in industrial manufacturing.
> **Note**: The FabriCAD dataset was created using a graph-based modeling language in **Design Cockpit 43 (DC43)**, developed by [IILS](https://www.iils.de/de/), who also supported the dataset development.

---

## 🎯 Overview

The dataset focuses on representing realistic manufacturing processes in metalworking by generating sequences derived from Markov chains extracted from 50,000+ real-world metalworking work plans. It provides detailed, multi-step process plans and feature-level CAD information to facilitate granular analysis and model training.

### Key Features

- **Parametric CAD models** (STEP format): 3D CAD files representing manufactured parts
- **Structured multi-step process plans**: Detailed workflows with steps like milling, drilling, welding, and grinding
- **Human-readable and machine-readable data**: Process plans and metadata in CSV format
- **Feature-level details**: Information on tooling, cutting parameters, and process-specific annotations
- **Synthetic yet realistic**: Based on Markov chains from 50,000+ real work plans
- **ML-ready**: Supports machine learning approaches in manufacturing process prediction and optimization

---

## 🔬 Intended Use Cases

- **Research in AI-assisted manufacturing** and process automation
- **Development and benchmarking** of machine learning models for manufacturing process prediction and optimization
- **Educational and prototyping purposes** within industrial engineering and manufacturing informatics
- **Manufacturing informatics** research and development

---

## 📦 Dataset Structure

Each sample primarily consists of:

### Manufacturing Process Plan
- **Process steps**: Milling, drilling, welding, grinding, etc.
- **Workstations**: Equipment and setup information
- **Duration and cost estimates**: Time and resource planning
- **Feature-level metadata**: Type of hole, thread details, tooling specifications

### 3D CAD Model
- **Parametric design**: STEP format (.stp)
- **Manufacturable geometry**: Realistic wall thicknesses and feature placements
---

## 🛠️ Supported Manufacturing Operations

### Geometry-Affecting Steps
- **Milling**: Material removal operations
- **Drilling**: Hole creation and modifications
- **Welding**: Joining operations
- **Grinding**: Surface finishing and precision operations

### Additional Process Steps
- **Deburring**: Edge finishing
- **Marking**: Part identification and labeling
- **Delivery**: Logistics and transportation
- **Inspection**: Quality control procedures
- **Testing**: Validation and verification processes

---

## 📥 Dataset Access

Complete datasets are available through our research data repository:

- **fabricad-10k**: 10,000 samples (compact dataset for initial experiments)
- **fabricad-50k**: 50,000 samples (medium-scale for comprehensive analyses)
- **fabricad-100k**: 100,000 samples (full dataset for intensive ML research)
- **fabricad-1k-detailed**: 1,000 samples with complete process visualization

**Access is provided upon request.** Contact us with your research affiliation, intended use case, and preferred dataset size.

### Preview Data
Download preview data from this repository or visit our [demo page](https://cimtt-kiel.github.io/FabriCAD/) to explore sample data and visualizations.

---

## 🚀 Quick Start

### 1. Explore Preview Data
```bash
git clone https://github.com/CIMTT-Kiel/FabriCAD.git
cd FabriCAD
```

### 2. Set Up Environment
```bash
uv venv
uv pip install -e .
```

### 3. Launch Visualization App
```bash
streamlit run Home.py
```

### 4. Explore the Data
- Browse process plan CSV files
- Examine STEP CAD models
- Use the Streamlit app for interactive visualization

---

## 📄 Data Format

### CSV Files (Process Plans)
- **Structured format**: Machine-readable process sequences
- **Comprehensive metadata**: Feature-level details and specifications
- **Human-readable**: Clear column headers and data organization

### CAD Files
- **Standard format**: STEP (.stp) files for broad compatibility
- **Parametric models**: Fully defined geometric features
- **Manufacturing-ready**: Realistic constraints and feature placement

### Detailed Dataset Features
The `fabricad-1k-detailed` dataset includes:
- **Process images**: Visual representation of each manufacturing step
- **Intermediate CAD states**: Geometry at each process stage

---

## 🎛️ Advanced Capabilities

**Contact us for custom requirements and extended data options.**

---

## ⚠️ Important Limitations

- **Synthetic data**: All data is artificially generated and simplified
- **Research purposes only**: Not suitable for productive or industrial use
- **Manufacturability rules**: Parts follow realistic constraints, but some unrealistic configurations may occur due to large scale
- **Quality variance**: Due to automated generation, manual review recommended for critical applications

---

## 🏗️ Technical Implementation

### Data Generation Pipeline
- **Markov chain modeling**: Process sequences based on real manufacturing data
- **Rule-based geometry**: CAD generation following manufacturability constraints
- **Feature correlation**: Realistic relationships between process steps and geometric features

### Graph-Based Design via DC43
- The synthetic parts and process plans were generated using DC43, a graph-based design tool developed by [IILS](https://www.iils.de/de/).
- The underlying design language enabled rule-based, modular construction of manufacturable geometries.
- The DC43 development team supported the project by providing access to the tool and assisting with its integration into our dataset generation pipeline.

### Quality Assurance
- **Automated validation**: Geometric and process consistency checks
- **Manufacturability constraints**: Wall thickness, feature placement, and accessibility rules
- **Process logic**: Realistic sequencing and resource requirements

---

## 📚 Research Applications

### Machine Learning Workflows
- **Process prediction**: Sequence modeling and next-step prediction
- **Cost estimation**: Resource and time prediction models
- **Quality optimization**: Process parameter optimization

### Academic Research
- **Benchmarking**: Standardized dataset for comparative studies
- **Algorithm development**: Training and testing new approaches
- **Educational use**: Teaching manufacturing informatics concepts

---

## 📧 Contact & Collaboration

For questions, collaborations, dataset access requests, or custom data generation:

**Prof. Dr.-Ing. Daniel Böhnke**  
📧 daniel.boehnke@fh-kiel.de

**Michel Kruse**  
📧 michel.kruse@fh-kiel.de

**Institution:** CIMTT, Fachhochschule Kiel

**For dataset access:** Please include your research affiliation, intended use case, and preferred dataset size in your inquiry.

---

## 📜 License & Citation

This dataset is provided under the **MIT License** for research purposes.

### Citation
If you use this dataset in your research, please cite:

```bibtex
@dataset{fabricad2025,
  title={FabriCAD: A Synthetic Dataset for AI-Assisted Manufacturing Research},
  author={Böhnke, Daniel and Fabian Heinze and Kruse, Michel},
  year={2025},
  publisher={CIMTT, FH Kiel},
  url={https://github.com/CIMTT-Kiel/FabriCAD}
}
```

---

## 🙏 Acknowledgments

We would like to thank our industrial partner who enabled access to the real-world process plans used to calculate the Markov chains that form the foundation of this synthetic dataset.

Special thanks go to the Design Cockpit 43 (DC43) development team at IILS, who provided access to the software and supported us during the development. Their graph-based design language was instrumental in generating structured CAD and process data.

---

**FabriCAD Project — Created by CIMTT, FH Kiel — 2025**

---

## 🔗 Related Resources
- ✈️ **Design Cockpit 43**: [https://www.iils.de/de/]
- 📊 **Demo Page**: [cimtt-kiel.github.io/FabriCAD](https://cimtt-kiel.github.io/FabriCAD/)
- 🏫 **CIMTT**: [Center for Industrial Manufacturing Technologies and Processes](https://www.fh-kiel.de/)
