from typing import List, Optional
from overrides import overrides
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from archai.discrete_search.api import ArchaiModel

class MyModel(nn.Module):
    def __init__(self,nb_layers: int,kernel_size: int,hidden_dim:int):
        super().__init__()
        self.nb_layers=nb_layers
        self.kernel_size=kernel_size
        self.hidden_dim=hidden_dim

        layer_list = []

        for i in range(nb_layers):
            in_ch = (1 if i == 0 else hidden_dim)

            layer_list += [
                nn.Conv2d(in_ch, hidden_dim, kernel_size=kernel_size, padding=(kernel_size-1)//2),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(),
            ]

        layer_list += [
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Conv2d(hidden_dim, 10, kernel_size=1)
        ]

        self.model = nn.Sequential(*layer_list)
    def forward(self,x):
        return self.model(x).squeeze()
    def get_archid(self):
        return f'({self.nb_layers}, {self.kernel_size}, {self.hidden_dim})'

model_obj=MyModel(1,1,28)
model=ArchaiModel(arch=model_obj,archid=f'L={model_obj.nb_layers}, K={model_obj.kernel_size}, H={model_obj.hidden_dim}')

import json
from random import Random
from archai.discrete_search.api import DiscreteSearchSpace

import json
from random import Random
from archai.discrete_search.api import DiscreteSearchSpace


class CNNSearchSpace(DiscreteSearchSpace):
    def __init__(self, min_layers: int = 1, max_layers: int = 12,
                 kernel_list=(1, 3, 5, 7), hidden_list=(16, 32, 64, 128),
                 seed: int = 1):

        self.min_layers = min_layers
        self.max_layers = max_layers
        self.kernel_list = kernel_list
        self.hidden_list = hidden_list

        self.rng = Random(seed)

    def get_archid(self, model: MyModel) -> str:
        return f'L={model.nb_layers}, K={model.kernel_size}, H={model.hidden_dim}'

    @overrides
    def random_sample(self) -> ArchaiModel:
        # Randomly chooses architecture parameters
        nb_layers = self.rng.randint(1, self.max_layers)
        kernel_size = self.rng.choice(self.kernel_list)
        hidden_dim = self.rng.choice(self.hidden_list)

        model = MyModel(nb_layers, kernel_size, hidden_dim)

        # Wraps model into ArchaiModel
        return ArchaiModel(arch=model, archid=self.get_archid(model))

    @overrides
    def save_arch(self, model: ArchaiModel, file_path: str=r'C:\Users\Aryan\Downloads\archai-main\archai-main\ff_test.json'):
        with open(file_path, 'w') as fp:
            print(model)
            json.dump({
                'nb_layers': model.arch.nb_layers,
                'kernel_size': model.arch.kernel_size,
                'hidden_dim': model.arch.hidden_dim
            }, fp)

    @overrides
    def load_arch(self, file_path: str)->ArchaiModel:
        config = json.load(open(file_path))
        model = MyModel(**config)

        return ArchaiModel(arch=model, archid=self.get_archid(model))

    @overrides
    def save_model_weights(self, model: ArchaiModel, file_path: str=r'C:\Users\Aryan\Downloads\archai-main\archai-main\ff_test.json'):
        state_dict = model.arch.get_state_dict()
        torch.save(state_dict, file_path)

    @overrides
    def load_model_weights(self, model: ArchaiModel, file_path: str=r'C:\Users\Aryan\Downloads\archai-main\archai-main\ff_test.json'):
        model.arch.load_state_dict(torch.load(file_path))

from archai.discrete_search.api.search_space import EvolutionarySearchSpace, BayesOptSearchSpace
from random import random
class CNNSearchSpaceExt(CNNSearchSpace, EvolutionarySearchSpace, BayesOptSearchSpace):
    ''' We are subclassing CNNSearchSpace just to save up space'''

    @overrides
    def mutate(self, arch: ArchaiModel) -> ArchaiModel:
        print(arch.arch.nb_layers)
        self.config = {
            'nb_layers': arch.arch.nb_layers,
            'kernel_size': arch.arch.kernel_size,
            'hidden_dim': arch.arch.hidden_dim
        }
        print(self.config)
        if random() < 0.2:
            self.config['nb_layers'] = self.rng.randint(self.min_layers, self.max_layers)

        if random() < 0.2:
            self.config['kernel_size'] = self.rng.choice(self.kernel_list)

        if random() < 0.2:
            self.config['hidden_dim'] = self.rng.choice(self.hidden_list)

        mutated_model = MyModel(**self.config)

        return ArchaiModel(
            arch=mutated_model, archid=self.get_archid(mutated_model)
        )

    @overrides
    def crossover(self, arch_list: List[ArchaiModel]) -> ArchaiModel:
        new_config = {
            'nb_layers': self.rng.choice([m.arch.nb_layers for m in arch_list]),
            'kernel_size': self.rng.choice([m.arch.kernel_size for m in arch_list]),
            'hidden_dim': self.rng.choice([m.arch.hidden_dim for m in arch_list]),
        }

        crossover_model = MyModel(**new_config)

        return ArchaiModel(
            arch=crossover_model, archid=self.get_archid(crossover_model)
        )

    @overrides
    def encode(self,arch: ArchaiModel) -> np.ndarray:
        return np.array([arch.nb_layers,arch.kernel_size, arch.hidden_dim])
    

ss = CNNSearchSpaceExt(min_layers=1, max_layers=10, kernel_list=[3, 5, 7], hidden_list=[16, 32, 64])
m=ss.random_sample()
print(m.arch.nb_layers)

from archai.discrete_search.api import SearchObjectives
objectives = SearchObjectives()

from archai.discrete_search.evaluators.pt_profiler import TorchFlops
ss = CNNSearchSpaceExt(max_layers=10, kernel_list=[3, 5, 7], hidden_list=[16, 32, 64])
objectives = SearchObjectives()


arch=model
objectives.add_objective(
    'FLOPs', TorchFlops(torch.randn(1, 1, 28, 28)),
    higher_is_better=False,
    compute_intensive=False,
    # We may optionally add a constraint.
    # Architectures outside this range will be ignored by the search algorithm
    # constraint=(0.0, 1e9)
)

from archai.discrete_search.algos import EvolutionParetoSearch
algo = EvolutionParetoSearch(
    ss, objectives,
    output_dir='./out_evo',
    num_iters=5, num_crossovers=5,
    mutations_per_parent=5,
    max_unseen_population=10,
    save_pareto_model_weights=False,
    seed=42
)

search_results = algo.search()
print(search_results)
print(objectives._objs)
import os


search_results.plot_2d_pareto_evolution(('FLOPs', 'iteration_num'))

results_df = search_results.get_search_state_df()
print(results_df)
