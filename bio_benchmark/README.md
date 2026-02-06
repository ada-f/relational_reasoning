## REL-B: Biological Relational Reasoning

**Domain:** Evolutionary biology  
**Entities:** Biological sequences and taxa  
**Goal:** Detect and localize **homoplasy**—independent emergence of the same sequence motif in evolutionarily distant lineages.

Each REL-B task provides:
- A **multiple sequence alignment (MSA)**
- A corresponding **phylogenetic tree**

The model must:
1. Decide whether structured homoplasy is present.
2. If present, identify which taxa participate.

### Task

- **Homoplasy detection and identification (RC = number of homoplastic taxa)**  
  Solving requires jointly binding:
  - Shared motifs in the alignment  
  - The evolutionary relationships encoded in the tree  