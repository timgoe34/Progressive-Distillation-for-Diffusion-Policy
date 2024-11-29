# Progressive Distillation for Diffusion Policy

The standard Diffusion Policy uses this training loop to train a diffusion model conditioned on a latent observation space to predict robot action sequences (adapted from https://colab.research.google.com/drive/18GIHeOQ5DyjMN8iIRZL2EKZ0745NLIpg?usp=sharing#scrollTo=VrX4VTl5pYNq). I now want to create a progressive distiallation trainer.


I created the training class "ProgressiveDistiller" which takes the previously trained nets as a teacher and distills down the model to use fewer steps to achieve the same as first the teacher model.

Problems:
- I expect the trained student network to do denoise the action sequence (of the same quality) in half the teacher steps. Currently the trained student just predicts random action points (not a smooth path of the robot endeffector)
- I think i falsely defined the intermediate teacher target. It should step the next two DDIM noise predictions by the teacher and mse-loss with the student.
What code changes would be necessary?
