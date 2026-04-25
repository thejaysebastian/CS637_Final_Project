# CS637_Final_Project
This project will use the EuroSat RGB dataset to experiment with the DenseNet architecture.

## Research Question

How does a memory-efficient PyTorch implementation of DenseNet perform on EuroSat RGB land-cover classification compared with the ResNet-50 and GoogLeNet baselines reported in the EuroSat paper?

## Project plan
1. Using the EuroSat RGB dataset, implement the DenseNet architecture in PyTorch as described in their original paper here: http://arxiv.org/abs/1608.06993. The original model and code are here: https://github.com/liuzhuang13/DenseNet. This original implemenation was in lua and not memory efficient, so we are implementing the author's "memory efficient" implementation in Python located here: https://github.com/gpleiss/efficient_densenet_pytorch, from the paper located here: https://arxiv.org/pdf/1707.06990.pdf.

1. Build, Train, and Test the DenseNet architecture with the EuroSat RGB dataset. The dataset is located here: https://huggingface.co/datasets/blanchon/EuroSAT_RGB.

1. Evaluate the performance of the DenseNet model against the ResNet-50 and GoogLeNet results in the original EuroSat paper. The paper is here: https://arxiv.org/abs/1709.00029 and the original EuroSat repo is here: https://github.com/phelber/EuroSAT.

1. Use the same train/test split and other methodological approaches from section IV.A of the EuroSat paper (e.g., pretrained on ILSVRC-2012). See Tables 2 and 3. Replicate these tables, adding DenseNet to the list.

1. Create slides for presentation (Shorter than paper presentation) per the requirements below.

1. Write individual papers describing contributions and results per the requirements below.

Citations:
@article{helber2017eurosat,
   title={EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification},
   author={Helber, et al.},
   journal={ArXiv preprint arXiv:1709.00029},
   year={2017}
}

@inproceedings{DenseNet2017,
  title={Densely connected convolutional networks},
  author={Huang, Gao and Liu, Zhuang and van der Maaten, Laurens and Weinberger, Kilian Q },
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2017}
}

@article{pleiss2017memory,
  title={Memory-Efficient Implementation of DenseNets},
  author={Pleiss, Geoff and Chen, Danlu and Huang, Gao and Li, Tongcheng and van der Maaten, Laurens and Weinberger, Kilian Q},
  journal={arXiv preprint arXiv:1707.06990},
  year={2017}
}

## High level requirements:
- Use the EuroSat RGB dataset
- Create a one-page proposal (Complete)
- Produce a 10-minute presentation
    - 12/15 slides
    - Based on what was in the proposal
    - 2-3 slides for project description, dataset description, data preprocessing (if any)
    - Remainder of slides should show method (architecture) applied or implemented and the results
    - Each member of the group must present a part
    - Must be in room 5 minutes prior to show time. Presentation time is 3:30-3:40.
- Write an individual term report
    - Include individual contributions, analysis, and reflections on project outcomes
    - Intro - introduce the problem being solved
    - Dataset - briefly discuss dataset and any preprocessing that may have been needed
    - Method - discuss method in detail
    - Results - report results with tables and figures as needed. Results should be the same for both team members, but reporting is independent
    - Conclusion - Your analysis about the project (such as why we chose the particular architecture, challenges faced while implementing, how I overcame issues, any further improvement possible.
    - Format - no specified format or page limit for the report.




