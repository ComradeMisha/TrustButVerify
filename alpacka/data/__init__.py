"""Datatypes and functions for manipulating them."""

import collections
from enum import Enum

import numpy as np

from alpacka.data.ops import *


# Transition between two states, S and S'.
Transition = collections.namedtuple(
    'Transition',
    [
        # Observation obtained at S.
        'observation',
        # Action played based on the observation.
        'action',
        # Reward obtained after performing the action.
        'reward',
        # Whether the environment is "done" at S'.
        'done',
        # Observation obtained at S'.
        'next_observation',
        # Dict of any additional info supplied by the agent.
        'agent_info',
        # Dict of any additional info supplied by the env.
        'env_info',
    ]
)


# Basic Episode object, summarizing experience collected when solving an episode
# in the form of transitions. It's used for basic Agent -> Trainer
# communication. Agents and Trainers can use different (but shared) episode
# representations, as long as they have a 'return_' field, as this field is used
# by Runner for reporting metrics.
Episode = collections.namedtuple(
    'Episode',
    [
        # Transition object containing a batch of transitions.
        'transition_batch',
        # Undiscounted return (cumulative reward) for the entire episode.
        'return_',
        # Whether the episode was "solved".
        'solved',
        # Whether the episode was truncated by the TimeLimitWrapper.
        'truncated',
        # Trainable Env model info.
        'trainable_env_info',
    ]
)
# solved, truncated, trainable_env_info
Episode.__new__.__defaults__ = (None, None, None)

# Signature of a tensor. Contains shape and datatype - the static information
# needed to initialize a tensor, for example a numpy array.
TensorSignature = collections.namedtuple(
    'TensorSignature', ['shape', 'dtype']
)
TensorSignature.__new__.__defaults__ = (np.float32,)  # dtype
# Register TensorSignature as a leaf type, so we can for example do nested_map
# over a structure of TensorSignatures to initialize a pytree of arrays.
register_leaf_type(TensorSignature)


# Signature of a network: input -> output. Both input and output are pytrees of
# TensorSignatures.
NetworkSignature = collections.namedtuple(
    'NetworkSignature', ['input', 'output']
)


# Request of a network_fn: Function () -> Network and the current parameters.
# This class is deprecated - use Request class instead.
class NetworkRequest:
    pass

# Requests in form of raw np.ndarray are deprecated as well.
# Please use Request class instead.


class RequestType(Enum):
    AGENT_NETWORK = 1
    AGENT_PREDICTION = 2
    MODEL_PREDICTION = 3


class Request:
    def __init__(self, request_type, content=None):
        self.type = request_type
        self.content = content
