Segmentation model (to be added):
Typical choice: a U‑Net–style model or an encoder–decoder with EfficientNet (or similar) as encoder and deconvolution/upsampling decoder.
Output: 2‑channel mask (cup, disc), from which we compute CDR with compute_cdr_from_masks.