from contextlib import contextmanager

import matplotlib.pyplot as plt

plt.style.use("seaborn")


class TracePlotter:

    def __init__(self, trajectory, label=None):
        self.trajectory = trajectory
        self._label = label

    @contextmanager
    def display(self, nrows=1, ncols=1, suptitle=None, show=False, fname=None):
        fig, axes = plt.subplots(nrows, ncols)

        if suptitle:
            fig.suptitle(suptitle)

        yield axes

        plt.tight_layout()

        if fname:
            fig.savefig(fname)

        if show:
            plt.show()

    def plot_cumulative_cost(self, ax):
        ax.set_title("Cumulative Costs", fontweight="bold")
        ax.set_xlabel("Timesteps")
        ax.plot(self.trajectory.timesteps, self.trajectory.cumulative_cost)

    def plot_states_history(self, ax, legend=False):
        timesteps = self.trajectory.timesteps
        for idx in range(self.trajectory.state_size):
            states = self.trajectory.states[:, idx]
            ax.plot(timesteps, states, label=f"x[{idx}]")

        ax.set_title("States", fontweight="bold")
        ax.set_xlabel("Timesteps")
        if legend:
            ax.legend()

    def plot_actions_history(self, ax, legend=False):
        timesteps = self.trajectory.timesteps[1:]
        for idx in range(self.trajectory.action_size):
            actions = self.trajectory.actions[:, idx]
            ax.plot(timesteps, actions, label=f"u[{idx}]")

        ax.set_title("Actions", fontweight="bold")
        ax.set_xlabel("Timesteps")
        if legend:
            ax.legend()

    def plot_cost_history(self, ax):
        ax.set_title("Costs", fontweight="bold")
        ax.set_xlabel("Timesteps")
        ax.plot(self.trajectory.timesteps, self.trajectory.costs)
