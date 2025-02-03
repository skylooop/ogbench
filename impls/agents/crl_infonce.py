from typing import Any

import flax
import flax.struct
import jax
import jax.numpy as jnp
import ml_collections
import optax
from impls.utils.encoders import GCEncoder, encoder_modules
from impls.utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from impls.utils.networks import GCActor, GCBilinearValue, GCDiscreteActor, GCDiscreteBilinearCritic

from jaxtyping import *

class CRLInfoNCEAgent(flax.struct.PyTreeNode):
    """Contrastive RL (CRL) agent with InfoNCE loss."""

    rng: Key
    network: TrainState
    config: ml_collections.ConfigDict = nonpytree_field()
    
    @classmethod
    def create(cls,
               seed: int,
               ex_observations,
               ex_actions,
               config):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.key(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]
            
        assert config['critic_arch'] in ['bilinear', 'mlp']
        
    
def get_config():
    pass