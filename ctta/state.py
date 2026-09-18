"""State/reset policy for source, episodic, domain and continual TTA."""

from __future__ import annotations

from copy import deepcopy

import torch


class AdaptationStateController:
    VALID_MODES = {'source', 'episodic', 'domain', 'continual'}

    def __init__(self, model, optimizer, args, mode: str):
        if mode not in self.VALID_MODES:
            raise ValueError(f'Unknown reset mode: {mode}')
        self.model = model
        self.optimizer = optimizer
        self.args = args
        self.mode = mode
        self.optimizer_state = deepcopy(optimizer.state_dict()) if optimizer is not None else None

    @property
    def core_model(self):
        return self.model.module if hasattr(self.model, 'module') else self.model

    @property
    def should_adapt(self) -> bool:
        return self.mode != 'source' and self.optimizer is not None and self.args.tta_steps > 0

    def reset(self):
        with torch.no_grad():
            self.core_model.reset(self.args)
        if self.optimizer is not None and self.optimizer_state is not None:
            self.optimizer.load_state_dict(deepcopy(self.optimizer_state))

    def before_stream(self):
        self.reset()

    def before_domain(self, domain_index: int):
        del domain_index
        if self.mode == 'domain':
            self.reset()

    def before_sample(self):
        if self.mode == 'episodic':
            self.reset()
