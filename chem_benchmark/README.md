## REL-C: Chemical Relational Reasoning

**Domain:** Chemistry  
**Entities:** Molecules (SMILES)  
**Goal:** Reason over sets of molecules to infer shared structure or missing elements.

REL-C contains three tasks with increasing relational complexity and operand difficulty.

### Tasks

- **REL-C1: Constitutional isomer set classification (RC = 2)**  
  Determine whether a set of molecules all share the same molecular formula but differ in connectivity.

- **REL-C2: Largest continuous common substructure (RC = 2)**  
  Identify the **maximum connected common substructure (MCS)** shared across all molecules in a set.

- **REL-C3: Missing isomer completion (RC ≥ number of isomers)**  
  Given a partial set of constitutional isomers, infer and enumerate the missing isomers implied by the shared molecular formula.  
  This task requires simultaneously binding:
  - The full isomer space  
  - The observed subset  
  - The shared molecular formula  