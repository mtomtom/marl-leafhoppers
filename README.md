# Leafhopper Simulation

This code simulates the behavior of leafhoppers in a 2D field representing host quality. The leafhoppers move, eat, rest, reproduce, and die based on their energy levels and age. The field regenerates over time, simulating plant growth.

## Functions and Classes

### `create_field(size=100)`
Creates a 2D matrix representing the host quality of the field with random values between 0 and 1.

- **Parameters:**
  - `size` (int): The size of the field (default is 100).
- **Returns:**
  - `np.ndarray`: A size x size matrix with random values between 0 and 1.

### `class Leafhopper`

Represents a leafhopper in the simulation.

#### `__init__(self, x, y, field, alpha=0.1, gamma=0.9, inherited_q_table=None)`

Initializes a leafhopper with its position, field, and Q-learning parameters.

- **Parameters:**
  - `x` (int): The x-coordinate of the leafhopper.
  - `y` (int): The y-coordinate of the leafhopper.
  - `field` (np.ndarray): The field in which the leafhopper exists.
  - `alpha` (float): Learning rate for Q-learning (default is 0.1).
  - `gamma` (float): Discount factor for Q-learning (default is 0.9).
  - `inherited_q_table` (dict): Q-table inherited from the parent (default is None).

#### `mutate_q_table(self, q_table)`

Mutates the Q-table by adding noise to inherited Q-values.

- **Parameters:**
  - `q_table` (dict): The Q-table to be mutated.
- **Returns:**
  - `dict`: The mutated Q-table.

#### `get_state(self)`

Gets the current state of the leafhopper.

- **Returns:**
  - `tuple`: The current state as a tuple of (x, y, energy).

#### `choose_action(self, state)`

Chooses an action based on the epsilon-greedy policy.

- **Parameters:**
  - `state` (tuple): The current state of the leafhopper.
- **Returns:**
  - `str`: The chosen action.

#### `move(self)`

Moves the leafhopper to the neighboring cell with the highest resources.

#### `eat(self)`

Leafhopper eats part of the resources in the cell, leaving some for regeneration.

#### `rest(self)`

Leafhopper rests and does nothing for one time step.

#### `take_action(self, action)`

Takes the specified action.

- **Parameters:**
  - `action` (str): The action to be taken.

#### `get_reward(self)`

Gets the reward based on the leafhopper's energy and age.

- **Returns:**
  - `int`: The reward value.

#### `update_q_table(self, state, action, reward, next_state)`

Updates the Q-table using the Q-learning algorithm.

- **Parameters:**
  - `state` (tuple): The current state.
  - `action` (str): The action taken.
  - `reward` (int): The reward received.
  - `next_state` (tuple): The next state.

#### `step(self)`

Performs one step in the environment.

- **Returns:**
  - `int`: The reward received.

#### `die(self)`

Checks if the leafhopper dies based on its age and energy.

- **Returns:**
  - `bool`: True if the leafhopper dies, False otherwise.

#### `reproduce(self)`

Reproduces a new leafhopper if the energy threshold is met.

- **Returns:**
  - `Leafhopper`: The new leafhopper if reproduction occurs, None otherwise.

### `regenerate_field(field, growth_rate=0.01)`

Regenerates the field by increasing the energy values over time.

- **Parameters:**
  - `field` (np.ndarray): The field to be regenerated.
  - `growth_rate` (float): The rate of regeneration (default is 0.01).

### `run_simulation(num_epochs=100, num_hoppers=50, field_size=100)`

Simulates the environment with leafhoppers over a specified number of epochs.

- **Parameters:**
  - `num_epochs` (int): The number of epochs to run the simulation (default is 100).
  - `num_hoppers` (int): The initial number of leafhoppers (default is 50).
  - `field_size` (int): The size of the field (default is 100).

## Example Usage

```python
# Run the simulation
run_simulation(num_epochs=200, num_hoppers=500, field_size=50)
```

This will run the simulation with 500 leafhoppers in a 50x50 field for 200 epochs.

## Dependencies

- `numpy`
- `matplotlib`

Ensure you have these libraries installed before running the simulation. You can install them using pip:

```bash
pip install numpy matplotlib
```

## Notes

- The field regenerates over time, simulating plant growth.
- Leafhoppers move, eat, rest, reproduce, and die based on their energy levels and age.
- The simulation uses Q-learning for leafhopper behavior adaptation.
