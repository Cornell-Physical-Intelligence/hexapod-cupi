# Derived from RSL-RL5.0.1 PPO.update, BSD-3-Clause.
# Copyright (c)2021-2026 ETH Zurich and NVIDIA CORPORATION. All rights reserved.
# Full license preserved in inputs/RSL_LICENSE.
import hashlib, inspect
from pathlib import Path
import torch
from torch import nn
from rsl_rl.algorithms import PPO
from caps import MeanRegularizer
from gradient_diagnostics import gradient_decomposition, assigned_gradient_norm
BASE_SHA='a2d35e7ad7b884c80b7434e7d2ce785a6da1e18d93c96f1179fbd3a208669f8c'
class CapsPPO(PPO):
    def __init__(self,*args,caps_options,gradient_diagnostics=True,**kwargs):
        if hashlib.sha256(Path(inspect.getfile(PPO)).read_bytes()).hexdigest()!=BASE_SHA:
            raise RuntimeError('Installed RSL PPO differs from reviewed source')
        super().__init__(*args,**kwargs)
        if self.rnd or self.symmetry or self.is_multi_gpu or self.actor.is_recurrent or self.critic.is_recurrent:
            raise ValueError('This bounded CAPS consumer excludes RND/symmetry/distributed/recurrent modes')
        self.regularizer=MeanRegularizer(caps_options,self.device)
        if type(gradient_diagnostics) is not bool:raise ValueError('Explicit boolean diagnostics required')
        self.gradient_diagnostics=gradient_diagnostics
        self.completed_updates=0;self.last_update_diagnostics=[]
        if {id(p) for p in self.actor.parameters()}&{id(p) for p in self.critic.parameters()}:
            raise ValueError('Gradient decomposition requires separate actor/critic parameters')
    def update(self):
        self.regularizer.stats=[];self.last_update_diagnostics=[]
        self.current_update=self.completed_updates+1
        result=self._caps_update()
        self.completed_updates+=1
        for key in ['temporal','spatial','weighted','valid_pair_fraction']:
            rows=self.regularizer.stats
            result['caps_'+key]=sum(r[key] for r in rows)/len(rows) if rows else 0.
        for mode in ('quiet','moving'):
            rows=self.regularizer.stats;count=sum(r[mode+'_pairs'] for r in rows)
            result['caps_'+mode+'_temporal_mean']=sum(r[mode+'_temporal_mean']*r[mode+'_pairs'] for r in rows)/max(count,1)
        self.regularizer.last_terms=None
        return result
    def _caps_update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for minibatch_index,batch in enumerate(generator,1):
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Snapshot scalar state without changing the existing adaptive rule.
            lr_before=float(self.learning_rate);kl_value=None
            sparse=self.gradient_diagnostics and self.current_update in (1,10,25,50) and minibatch_index in (1,self.num_learning_epochs*self.num_mini_batches)
            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            if self.gradient_diagnostics:
                if self.desired_kl is not None and self.schedule=='adaptive':kl_value=float(kl_mean)
                diagnostic={'update':self.current_update,'minibatch':minibatch_index,'kl_mean':kl_value,
                            'learning_rate_before':lr_before,'learning_rate_after':float(self.learning_rate),
                            'gradient':None}
            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
            loss = loss + self.regularizer(self.actor, batch.observations)

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            if self.rnd:
                # Extract the rnd_state
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            if sparse:
                terms=self.regularizer.last_terms
                if terms is None:
                    zero=surrogate_loss.new_zeros(())
                    terms={k:zero for k in ('quiet_temporal','moving_temporal','spatial')}
                diagnostic['gradient']=gradient_decomposition(
                    surrogate_loss-self.entropy_coef*entropy.mean(),terms,self.actor.parameters())
            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            if sparse:diagnostic['gradient']['combined_actor_before_clip']=assigned_gradient_norm(self.actor.parameters())
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            if sparse:diagnostic['gradient']['combined_actor_after_clip']=assigned_gradient_norm(self.actor.parameters())
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            if self.gradient_diagnostics:
                if self.regularizer.stats:
                    stats=self.regularizer.stats[-1]
                    diagnostic['pair_counts']={k:stats[k] for k in ('valid_pairs','quiet_pairs','moving_pairs')}
                else:diagnostic['pair_counts']=None
                self.last_update_diagnostics.append(diagnostic)
            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict
