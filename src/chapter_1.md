# Neuromorphic Computing Tools, Libraries, and Frameworks

[Lava](https://github.com/lava-nc/lava) is an open-source software framework for developing neuro-inspired applications and mapping them to neuromorphic hardware. Lava provides developers with the tools and abstractions to develop applications that fully exploit the principles of neural computation. Constrained in this way, like the brain, Lava applications allow neuromorphic platforms to intelligently process, learn from, and respond to real-world data with great gains in energy efficiency and speed compared to conventional computer architectures.

[Lava DL](https://github.com/lava-nc/lava-dl) is an enhanced version of [SLAYER](https://github.com/bamsumit/slayerPytorch). Some enhancements include support for recurrent network structures, a wider variety of neuron models and synaptic connections (complete list of features [here](https://github.com/lava-nc/lava-dl/blob/main/lib/dl/slayer/README.md)). This version of SLAYER is built on top of the PyTorch deep learning framework, similar to its predecessor.

[Lava Dynamic Neural Fields (DNF)](https://github.com/lava-nc/lava-dnf) are neural attractor networks that generate stabilized activity patterns in recurrently connected populations of neurons. These activity patterns form the basis of neural representations, decision making, working memory, and learning. DNFs are the fundamental building block of [dynamic field theory](https://dynamicfieldtheory.org/), a mathematical and conceptual framework for modeling cognitive processes in a closed behavioral loop.

[Neuromorphic Constraint Optimization](https://github.com/lava-nc/lava-optimization) is a library of solvers that leverage neuromorphic hardware for constrained optimization. Constrained optimization searches for the values of input variables that minimize or maximize a given objective function, while the variables are subject to constraints. This kind of problem is ubiquitous throughout scientific domains and industries. Constrained optimization is a promising application for neuromorphic computing as it [naturally aligns with the dynamics of spiking neural networks](https://doi.org/10.1109/JPROC.2021.3067593).

[NeuroKit2](https://github.com/neuropsychology/NeuroKit) is a user-friendly Python package providing easy access to advanced biosignal processing routines. 

[Neuro Digital Signal Processing(NeuroDSP) Toolbox](https://github.com/neurodsp-tools/neurodsp) is a collection of approaches for applying digital signal processing to neural time series, including algorithms that have been proposed for the analysis of neural time series. It also includes simulation tools for generating plausible simulations of neural time series.

[Norse](https://norse.github.io/norse/) is a deep learning tool with spiking neural networks (SNNs) in PyTorch. It expands PyTorch with primitives for bio-inspired neural components, bringing you two advantages: a modern and proven infrastructure based on PyTorch and deep learning-compatible spiking neural network components.

[JAX](https://github.com/google/jax) is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), brought together for high-performance numerical computing and machine learning research. It provides composable transformations of Python+NumPy programs such as differentiate, vectorize, parallelize, Just-In-Time (JIT) compile to GPU/TPU, and more.

[XLA (Accelerated Linear Algebra)](https://www.tensorflow.org/xla) is a domain-specific compiler for linear algebra that can accelerate TensorFlow models with potentially no source code changes.

[MNE-Python](https://github.com/mne-tools/mne-python) is an open-source Python package for exploring, visualizing, and analyzing human neurophysiological data such as Magnetoencephalography (MEG) , Electroencephalography (EEG), sEEG, ECoG, and more. It includes modules for data input/output, preprocessing, visualization, source estimation, time-frequency analysis, connectivity analysis, machine learning, and statistics.

[Nengo](https://github.com/nengo/nengo) is a Python library for building and simulating large-scale neural models. It can create sophisticated spiking and non-spiking neural simulations with sensible defaults in a few lines of code. 

[Keras](https://keras.io) is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.It was developed with a focus on enabling fast experimentation. It is capable of running on top of TensorFlow, Microsoft Cognitive Toolkit, R, Theano, or PlaidML.

[ONNX Runtime](https://github.com/microsoft/onnxruntime) is a cross-platform, high performance ML inferencing and training accelerator. It supports models from deep learning frameworks such as PyTorch and TensorFlow/Keras as well as classical machine learning libraries such as scikit-learn, LightGBM, XGBoost, etc.

[TorchScript](https://pytorch.org/docs/stable/jit.html) is a way to create serializable and optimizable models from PyTorch code. This allows any TorchScript program to be saved from a Python process and loaded in a process where there is no Python dependency.

[TorchServe](https://pytorch.org/serve/) is a flexible and easy to use tool for serving PyTorch models.

[TensorFlow Model Optimization Toolkit](https://www.tensorflow.org/model_optimization) is a suite of tools that users, both novice and advanced, can use to optimize machine learning models for deployment and execution. It provides supported techniques that include quantization and pruning for sparse weights. Along with APIs built specifically for Keras.

[DeepSpars](https://github.com/neuralmagic/deepsparse) is an inference runtime offering GPU-class performance on CPUs and APIs to integrate ML into your application.

[Intel® Neural Compressor](https://github.com/intel/neural-compressor) is a Low Precision Optimization Tool, targeting to provide unified APIs for network compression technologies, such as low precision quantization, sparsity, pruning, knowledge distillation, across different deep learning frameworks to pursue optimal inference performance. 

[Kornia](https://kornia.github.io/) is a differentiable computer vision library that consists of a set of routines and differentiable modules to solve generic CV (Computer Vision) problems.

[PyTorch-NLP](https://pytorchnlp.readthedocs.io/en/latest/) is a library for Natural Language Processing (NLP) in Python. It’s built with the very latest research in mind, and was designed from day one to support rapid prototyping. PyTorch-NLP comes with pre-trained embeddings, samplers, dataset loaders, metrics, neural network modules and text encoders.

[Ignite](https://pytorch.org/ignite) is a high-level library to help with training and evaluating neural networks in PyTorch flexibly and transparently.

[Hummingbird](https://github.com/microsoft/hummingbird) is a library for compiling trained traditional ML models into tensor computations. It allows users to seamlessly leverage neural network frameworks (such as PyTorch) to accelerate traditional ML models.

[Deep Graph Library (DGL)](https://www.dgl.ai/) is a Python package built for easy implementation of graph neural network model family, on top of PyTorch and other frameworks.

[TensorLy](http://tensorly.org/stable/home.html) is a high level API for tensor methods and deep tensorized neural networks in Python that aims to make tensor learning simple.

[GPyTorch](https://cornellius-gp.github.io/) is a Gaussian process library implemented using PyTorch, designed for creating scalable, flexible Gaussian process models.

[Poutyne](https://poutyne.org/) is a Keras-like framework for PyTorch and handles much of the boilerplating code needed to train neural networks.

[Forte](https://github.com/asyml/forte/tree/master/docs) is a toolkit for building NLP pipelines featuring composable components, convenient data interfaces, and cross-task interaction.

[TorchMetrics](https://github.com/PyTorchLightning/metrics) is a Machine learning metrics for distributed, scalable PyTorch applications.

[Captum](https://captum.ai/) is an open source, extensible library for model interpretability built on PyTorch.

[Transformer](https://github.com/huggingface/transformers) is a State-of-the-art Natural Language Processing for Pytorch, TensorFlow, and JAX.

[Hydra](https://hydra.cc) is a framework for elegantly configuring complex applications.

[Accelerate](https://huggingface.co/docs/accelerate) is a simple way to train and use PyTorch models with multi-GPU, TPU, mixed-precision.

[Ray](https://github.com/ray-project/ray) is a fast and simple framework for building and running distributed applications.

[ParlAI](http://parl.ai/) is a unified platform for sharing, training, and evaluating dialog models across many tasks.

[PyTorchVideo](https://pytorchvideo.org/) is a deep learning library for video understanding research. Hosts various video-focused models, datasets, training pipelines and more.

[Opacus](https://opacus.ai/) is a library that enables training PyTorch models with Differential Privacy.

[PyTorch Lightning](https://github.com/williamFalcon/pytorch-lightning) is a Keras-like ML library for PyTorch. It leaves core training and validation logic to you and automates the rest.

[PyTorch Geometric Temporal](https://github.com/benedekrozemberczki/pytorch_geometric_temporal) is a temporal (dynamic) extension library for PyTorch Geometric.

[PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) is a library for deep learning on irregular input data such as graphs, point clouds, and manifolds.

[Raster Vision](https://docs.rastervision.io/) is an open source framework for deep learning on satellite and aerial imagery.

[CrypTen](https://github.com/facebookresearch/CrypTen) is a framework for Privacy Preserving ML. Its goal is to make secure computing techniques accessible to ML practitioners.

[Optuna](https://optuna.org/) is an open source hyperparameter optimization framework to automate hyperparameter search.

[Pyro](http://pyro.ai/) is a universal probabilistic programming language (PPL) written in Python and supported by PyTorch on the backend.

[Albumentations](https://github.com/albu/albumentations) is a fast and extensible image augmentation library for different CV tasks like classification, segmentation, object detection and pose estimation.

[Skorch](https://github.com/skorch-dev/skorch) is a high-level library for PyTorch that provides full scikit-learn compatibility.

[MMF](https://mmf.sh/) is a modular framework for vision & language multimodal research from Facebook AI Research (FAIR).

[AdaptDL](https://github.com/petuum/adaptdl) is a resource-adaptive deep learning training and scheduling framework.

[Polyaxon](https://github.com/polyaxon/polyaxon) is a platform for building, training, and monitoring large-scale deep learning applications.

[TextBrewer](http://textbrewer.hfl-rc.com/) is a PyTorch-based knowledge distillation toolkit for natural language processing

[AdverTorch](https://github.com/BorealisAI/advertorch) is a toolbox for adversarial robustness research. It contains modules for generating adversarial examples and defending against attacks.

[NeMo](https://github.com/NVIDIA/NeMo) is a a toolkit for conversational AI.

[ClinicaDL](https://clinicadl.readthedocs.io/) is a framework for reproducible classification of Alzheimer's Disease

[Stable Baselines3 (SB3)](https://github.com/DLR-RM/stable-baselines3) is a set of reliable implementations of reinforcement learning algorithms in PyTorch.

[TorchIO](https://github.com/fepegar/torchio) is a set of tools to efficiently read, preprocess, sample, augment, and write 3D medical images in deep learning applications written in PyTorch.

[PySyft](https://github.com/OpenMined/PySyft) is a Python library for encrypted, privacy preserving deep learning.

[Flair](https://github.com/flairNLP/flair) is a very simple framework for state-of-the-art natural language processing (NLP).

[Glow](https://github.com/pytorch/glow) is a ML compiler that accelerates the performance of deep learning frameworks on different hardware platforms.

[FairScale](https://github.com/facebookresearch/fairscale) is a PyTorch extension library for high performance and large scale training on one or multiple machines/nodes.

[MONAI](https://monai.io/) is a deep learning framework that provides domain-optimized foundational capabilities for developing healthcare imaging training workflows.

[PFRL](https://github.com/pfnet/pfrl) is a deep reinforcement learning library that implements various state-of-the-art deep reinforcement algorithms in Python using PyTorch.

[Einops](https://github.com/arogozhnikov/einops) is a flexible and powerful tensor operations for readable and reliable code.

[PyTorch3D](https://pytorch3d.org/) is a deep learning library that provides efficient, reusable components for 3D Computer Vision research with PyTorch.

[Ensemble Pytorch](https://ensemble-pytorch.readthedocs.io/) is a unified ensemble framework for PyTorch to improve the performance and robustness of your deep learning model.

[Lightly](https://github.com/lightly-ai/lightly) is a computer vision framework for self-supervised learning.

[Higher](https://github.com/facebookresearch/higher) is a library which facilitates the implementation of arbitrarily complex gradient-based meta-learning algorithms and nested optimisation loops with near-vanilla PyTorch.

[Horovod](http://horovod.ai/) is a distributed training library for deep learning frameworks. Horovod aims to make distributed DL fast and easy to use.

[PennyLane](https://pennylane.ai/) is a library for quantum ML, automatic differentiation, and optimization of hybrid quantum-classical computations.

[Detectron2](https://github.com/facebookresearch/detectron2) is FAIR's next-generation platform for object detection and segmentation.

[Fastai](https://docs.fast.ai/) is a library that simplifies training fast and accurate neural nets using modern best practices.