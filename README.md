### Summary

We implemented one of the stages of the WebP image encoding pipeline on NVIDIA GPUs, to take advantage of the parallelism offered by CUDA. To do so, we investigated the algorithms used by the reference implementation of WebP encoding and rewrote several versions of portions of the encoding pipeline to best take advantage of CUDA. We then compared the modified pipeline’s performance on both GHC Cluster machines and the Pittsburgh Supercomputing Cluster Bridges machines, versus the sequential C implementation.

### December 10: Final Project Writeup, Poster, and Presentation

Our final project writeup is located [here](final_report.pdf), and our poster is located [here](poster.pdf).
Preview our poster below:
![Image preview of poster](https://github.com/emmaloool/15418-Final-Project/tree/gh-pages/poster-preview.png)


### November 18: Project Checkpoint

Our checkpoint report is located [here](checkpoint.pdf).

### Octobr 30: Project Proposal

Our project proposal is located [here](proposal.pdf).
