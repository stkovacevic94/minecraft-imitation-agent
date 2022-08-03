import dataclasses

import numpy as np

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


@dataclasses.dataclass(frozen=True)
class Transitions(Dataset):
    obs: np.ndarray
    act: np.ndarray

    def __post_init__(self):
        """Performs input validation: check shapes & dtypes match docstring.
        Also make array values read-only.
        Raises:
            ValueError: if batch size (array length) is inconsistent
                between `obs`, `acts` and `infos`.
        """
        for val in vars(self).values():
            if isinstance(val, np.ndarray):
                val.setflags(write=False)

        if len(self.obs) != len(self.act):
            raise ValueError(
                "obs and acts must have same number of timesteps: "
                f"{len(self.obs)} != {len(self.act)}",
            )

    def __len__(self):
        """Returns number of transitions. Always positive."""
        return len(self.obs)

    def __getitem__(self, index) -> T_co:
        return self.obs[index], self.act[index]
