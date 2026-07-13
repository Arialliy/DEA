# Search notes and screening ledger

## Search contract

- Date: 2026-07-13
- Research question: which prior work already covers pixel-to-instance reformulation, seed/growth programs, connected/topological outputs, center/radius target atoms and low-FA IRSTD?
- Sources accepted: CVF Open Access, NeurIPS Proceedings, OpenReview venue records, arXiv abstracts, and official journal pages.
- Sources excluded from evidence: MDPI, blogs, Reddit, commercial summaries, ResearchGate mirrors and search snippets when a primary source was available.
- Privacy: queries used generic public concepts; no local method document or unpublished implementation text was pasted into search.

## Query families

1. `MSHNet infrared small target scale location sensitivity CVPR`
2. `IRSTD shape edge false alarm CVPR ICCV`
3. `2025 2026 infrared small target centroid radius point mask`
4. `mask classification per-pixel classification instance segmentation`
5. `seeded recurrent mask growth flood filling network`
6. `region growing CNN instance segmentation frontier`
7. `deep watershed component energy instance segmentation`
8. `topology preserving segmentation component graph`
9. `recurrent instance masks polygon structured prediction`

## Screened corpus

### Included in final map (14)

- MSHNet (CVPR 2024)
- ISNet (CVPR 2022)
- MDvsFA (ICCV 2019)
- Point-to-Mask (arXiv 2026)
- SPIRE (arXiv 2026)
- DISTA-Net (ICCV 2025)
- MaskFormer (NeurIPS 2021)
- Mask2Former (CVPR 2022)
- Flood-Filling Networks (Nature Methods 2018; arXiv 2016)
- Deep Watershed Transform (CVPR 2017)
- End-to-End Instance Segmentation with Recurrent Attention (CVPR 2017)
- Polygon-RNN (CVPR 2017)
- Topograph (ICLR 2025 Spotlight)
- Region Growing CNN (arXiv 2020)

### Screened as background/control candidates, not final cards (12)

- Text-IRSTD (ICCV 2025): cross-modal semantic text; changes the information source and is outside the image-only baseline contract.
- PAL for point-supervised IRSTD (ICCV 2025): training curriculum/pseudo-label framework, not the full-supervision decision-unit issue.
- QueryInst (ICCV 2021): strengthens the query-mask prior-art pressure already represented by MaskFormer/Mask2Former.
- Mask Transfiner (CVPR 2022): mask refinement; useful as a negative example because a refinement head would violate the non-stacking constraint.
- CenterMask (CVPR 2020): center-conditioned instance masks; redundant with the center/mask pressure captured by Point-to-Mask and MaskFormer.
- Pixel affinity plus graph merging (ECCV 2018): graph-merge instance segmentation; relevant control if a forest/affinity pivot is later proposed.
- Topology-Preserving Deep Image Segmentation (NeurIPS 2019): persistent-homology topology loss; Topograph provides the more recent strict graph-based pressure.
- Polygon-RNN++ (CVPR 2018): improves Polygon-RNN but does not change the main structured-sequence pressure.
- SpirDet (arXiv 2024): sparse fast/slow decoder; primarily efficiency, not component decision semantics.
- FDEP (arXiv 2025): foundation representation/distillation framework; a likely cross-backbone baseline but also an example of multi-part framework design to avoid copying.
- GenMask (CVPR 2026): direct generative mask modeling; broad generative segmentation, not yet the closest low-FA component comparator.
- Text-guided and foundation-model IRSTD variants: valuable for a later current-SOTA table, not for deciding this frozen-d0 mechanism.

## Search-induced changes to the method

1. Downgraded `ROOT–ADD–STOP` from “candidate core novelty” to “support grammar under high prior-art pressure.”
2. Rejected a query/mask-classification head as the primary design because it would be a direct MaskFormer transplant.
3. Rejected a center/radius renderer because Point-to-Mask and SPIRE occupy that output reformulation.
4. Rejected topology loss, component graph loss and watershed post-processing as headline mechanisms.
5. Added mandatory controls for compact mask classification and unconstrained region growth.
6. Added serial-latency, exposure-bias and score-order blocking to the required evidence package.
7. Kept only the possible combined gap: an atomic connected component support with one decision mark and exact whole-component threshold filtration. This is explicitly not yet novelty-cleared.

## Next literature action before a paper claim

- Perform backward/forward citation chasing from Flood-Filling Networks, Region Growing CNN, Deep Watershed and MaskFormer for any method that jointly guarantees connected support and a global component score.
- Search patent/medical-connectomics literature for `rooted tree mask`, `component program`, `connected mask autoregressive`, `atomic component threshold` and `instance filtration`.
- Search recent AAAI/IJCAI structured-output work for finite-set likelihoods over connected masks.
- If a near-identical formulation is found, pivot before implementing the neural head; do not rename the same idea.
