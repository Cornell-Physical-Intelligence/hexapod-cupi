"""Build policy observations from measured state."""
import torch

from locomotion.omniwalk.mdp.frames import navigation


def proprio(state, neutral):
    """One 42-value frame: body rates and gravity in navigation axes, joint offsets from neutral, joint rates."""
    return torch.cat((navigation(state["angular"]), navigation(state["gravity"]),
                      state["q"] - neutral, state["dq"]), dim=-1)


def observations(history, commands, previous_action, state):
    """Pack the 231-value actor input and the 234-value critic input."""
    obs = torch.cat((history.flatten(1), commands, previous_action), dim=-1)
    critic = torch.cat((obs, navigation(state["linear"])), dim=-1)
    return {"obs": obs, "critic": critic}
