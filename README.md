# genAI-playground
This project provides implementation and integration through a CLI of state-of-the-art generative AI models, such as Denoising Diffusion Probabilistic Models (DDPMs), Cold Diffusion, and Latent Diffusion Models, as well as their predecessors, Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), and Energy Based Models (EBMs). This implementations were used to support our research in the fundamental properties of Diffusion Models and their evaluations as part of our Master Project @ ETHZ.

### Use Instructions

For simple use we provide a Command Line Interface were the user can select the desired model, parameters, and dataset to train the generative model. Example:
```
python imgGen_wrapper.py --model DDPM --img-size 28 --MNIST --noise-steps 100 500 750 1000 1500 2000 --batch-size 32 --epochs 22 --learning-rate 1e-4 3e-4 1e-3 --beta-start 5e-5 1e-4 5e-4 1e-3 --beta-end 0.01 0.02 0.05 --model-output output_models --eval FID --eval-steps 3 --eval-samples 750
```

As appreciated, it is possible to give a single parameter or a set of parameters, if a set of parameters is given, then all the possible combinations of the given parameters will be trained. The given example will result on training 36 different models (meshgrid)

### Evaluation

If evaluation is indicated, then the performance of the model will be calculated every eval-steps using given eval-samples based on the given measurement (currently only FID is supprted). [<cite>[Maximilian Seitzer][1]</cite>]

[1]: https://github.com/mseitzer/pytorch-fid

### Further Instructions
Please refer to
```
python imgGen_wrapper.py --help
```
