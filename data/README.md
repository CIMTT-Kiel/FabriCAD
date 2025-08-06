# Dataset Structure

The preview data contains 25 manufacturing process samples with intermediate states. (Intermediate states are available for a subset of 10.000 samples, not for all files)

## Directory Structure

```
fabricad-interim-states-1k/
├── 00000000/                          # Sample ID (8-digit, zero-padded)
├── 00000001/                          # Samples follow identical structure
├── ...                                # Samples 00000002 to 00000998
└── 00000999/                          # Total: 1,000 samples
```

## Sample Structure

Each sample follows this structure (example: `00000001`):

```
00000001/
├── geometry_00000001.STEP             # CAD geometry for the final state
├── plan.csv                           # Manufacturing plan
├── plan_metadata.csv                  # Metadata for the process
├── interim/                           # Process intermediate states
│   ├── interim_schweißen_step2.png    # Format: interim_{step_type}_step{step_nr}.{png and Step} 
│   ├── interim_schweißen_step2.STEP
│   ├── interim_fräsen_step3.png
│   ├── interim_fräsen_step3.STEP      # 3D geometry per step
│   ├──...
│   └── substeps/                      # Detailed substeps on machining-feature level 
│       ├── features.csv               # Geometric features for all steps seperated by main and substep
│       ├── comp_cost_step_2.csv       # Cost calculations if available
│       ├── comp_time_step_2.csv       # Time calculations if available
│       ├── substep_2.1.png            # For substep visualizations
│       ├── substep_2.1.STEP           # Substep CAD format
│       ├── substep_2.2.png/.STEP      # Format: substep_{step}.{substep} - corresponding to features.csv
│       └── ...                        
└── negative/                          # Negative geometries (removed material)
    ├── negative_welding_step2.STEP    # Material removal per step (on main-step level not on machining-feature level)
    └── ...
```

## Naming Conventions

### Substep Files
- **Format**: `substep_{main_step}.{sub_step}.{extension}`
- **Examples**: 
  - `substep_2.1.png` / `substep_2.1.STEP` (Main step 2, substep 1)
  - `substep_5.6.png` / `substep_5.6.STEP` (Main step 5, substep 6)
- **Pairing**: Each substep has both PNG (only for visualization in viewer) and STEP (CAD) files

### Process Files
- **Interim**: `interim_{process_name}_step{step_nr}.{extension}`
- **Negative**: `negative_{process_name}_step{step_nr}.STEP`

## File Types

- **`.STEP`**: CAD geometries STEP (ISO-10303-21)
- **`.png`**: Rendered visualizations  (only for viewer)
- **`.csv`**: Process data, features, costs, timing

## Important Data Relationships

### Feature Mapping
- **`features.csv`**: Contains all machining features with IDs
- **Feature-to-File Mapping**: Feature IDs correspond to substep file numbers
- **Example**: Feature ID `2.3` maps to `substep_2.3.png` and `substep_2.3.STEP`

### File Completeness
- **Missing Final Step**: Last interim step CAD file is omitted as it is equal to final state `geometry_XXXXXXXX.STEP`
- **Negative Accumulation**: Each negative file contains all material removed up to that step