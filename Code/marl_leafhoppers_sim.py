import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
import matplotlib.lines as mlines
import matplotlib
matplotlib.use('TkAgg')

def create_field(xdim, ydim):
    field = np.random.rand(xdim,ydim)
    ## Set 80% of the field to 0
    # Flatten the field to a 1D array
    flat_field = field.flatten()

    # Calculate the number of elements to set to 0
    num_elements_to_zero = int(0.8 * flat_field.size)

    # Randomly select indices to set to 0
    indices_to_zero = np.random.choice(flat_field.size, num_elements_to_zero, replace=False)

    # Set the selected indices to 0
    flat_field[indices_to_zero] = 0

    # Reshape the array back to its original 2D shape
    field = flat_field.reshape(100, 100)

    # Print the resulting field to verify
    return field

# Create the field (2D matrix) representing host quality (between 0 and 1)
def create_field(size=100):
    return np.random.rand(size, size)  # Returns a size x size matrix with random values between 0 and 1

class Leafhopper:
    def __init__(self, x, y, field, alpha=0.1, gamma=0.9, epsilon=0.1, inherited_q_table=None):
        self.x = x
        self.y = y
        self.field = field
        self.energy = 0.5  # Start with some initial energy
        self.age = 0
        self.reproduction = 0
        self.mutation_rate = 0.1  # Mutation rate for offspring's behavior
        self.energy_threshold = 0.6  # Adjust reproduction threshold
        self.age_threshold = 20  # Increase age threshold
        self.energy_decay = 0.01  # Lose energy each step
        
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Initialize Q-table or inherit it from a parent
        if inherited_q_table is None:
            self.q_table = {}  # Initialize new Q-table
        else:
            self.q_table = self.mutate_q_table(inherited_q_table)  # Mutate the parent's Q-table
        
        self.actions = ["move", "eat", "rest"]  # Possible actions
    
    def mutate_q_table(self, q_table):
        mutated_q_table = {}
        for state, q_values in q_table.items():
            if isinstance(q_values, np.ndarray):
                mutated_q_values = q_values + np.random.normal(0, self.mutation_rate, size=q_values.shape)
            else:
                mutated_q_values = np.random.normal(0, self.mutation_rate, size=len(self.actions))
            mutated_q_table[state] = mutated_q_values
        return mutated_q_table
    
    def get_state(self):
        # State is a tuple of the current position and energy
        state = (self.x, self.y, self.energy)
        return state

    def choose_action(self, state):
    # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:  # Exploration
            self.current_action = np.random.choice(self.actions)
        else:  # Exploitation
            if state not in self.q_table:
                self.q_table[state] = np.zeros(len(self.actions))  # Initialize Q-values
            self.current_action = self.actions[np.argmax(self.q_table[state])]
        
        return self.current_action

    def move(self):
        # Define relative positions for the 8 possible neighbors
        relative_positions = [(i, j) for i in range(-1, 2) for j in range(-1, 2) if not (i == 0 and j == 0)]
        
        # Shuffle the list of relative positions to randomize the evaluation order
        random.shuffle(relative_positions)

        neighbours = []
        best_quality = -1
        best_pos = None
        
        # Iterate over the shuffled neighboring positions
        for i, j in relative_positions:
            new_x = self.x + i
            new_y = self.y + j
            if 0 <= new_x < self.field.shape[0] and 0 <= new_y < self.field.shape[1]:
                neighbours.append((new_x, new_y))
                if self.field[new_x, new_y] > best_quality:
                    best_quality = self.field[new_x, new_y]
                    best_pos = (new_x, new_y)

        # Randomly choose to move to a random neighbor or the best neighboring cell
        if np.random.rand() < 0.1:  # 10% chance to move randomly
            if neighbours:
                self.x, self.y = random.choice(neighbours)
        elif best_pos:
            self.x, self.y = best_pos


    
    def eat(self):
        # Leafhopper eats part of the resources in the cell, leaving some for regeneration
        self.energy += self.field[self.x, self.y] * 0.5  # Consume 50% of available resources
        self.field[self.x, self.y] *= 0.5  # Resources get reduced by 50%

    def rest(self):
        pass  # Rest: Do nothing for one time step
    
    def take_action(self, action):
        if action == "move":
            self.move()
        elif action == "eat":
            self.eat()
        elif action == "rest":
            self.rest()
    
    def get_reward(self):
        # Reward for energy accumulation and penalize for aging without action
        if self.energy >= self.energy_threshold:
            return 10  # Positive reward for survival and reproduction potential
        elif self.energy == 0:
            return -10  # Penalty for starvation
        else:
            return -1  # Small penalty to encourage activity

    def update_q_table(self, state, action, reward, next_state):
        # Convert action to an index
        action_idx = self.actions.index(action)
        
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))
        
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(len(self.actions))
        
        # Q-learning update
        best_next_action = np.argmax(self.q_table[next_state])
        self.q_table[state][action_idx] += self.alpha * (
            reward + self.gamma * self.q_table[next_state][best_next_action] - self.q_table[state][action_idx]
        )

    def step(self):
        # Perform one step in the environment
        state = self.get_state()
        action = self.choose_action(state)
        self.take_action(action)
        reward = self.get_reward()
        next_state = self.get_state()
        self.update_q_table(state, action, reward, next_state)
        self.energy -= self.energy_decay  # Lose energy per step
        self.age += 1
        return reward

    def die(self):
        # Leafhopper dies if it exceeds age threshold or runs out of energy
        return self.age >= self.age_threshold or self.energy <= 0

    def reproduce(self):
        if self.energy >= self.energy_threshold:
            self.energy /= 2  # Share energy with offspring
            return Leafhopper(self.x, self.y, self.field, inherited_q_table=self.q_table)

# Function to regenerate the field
def regenerate_field(field, growth_rate=0.01):
    field += growth_rate
    np.clip(field, 0, 1, out=field)  # Ensure values stay within [0, 1]

def run_simulation(num_epochs=100, num_hoppers=50, field_size=100):
    field = create_field(size=field_size)
    leafhoppers = [Leafhopper(random.randint(0, field_size-1), random.randint(0, field_size-1), field) for _ in range(num_hoppers)]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # Set fixed vmin and vmax for the color scale
    vmin, vmax = 0, 1  # Since the field values are between 0 and 1
    img = ax.imshow(field, cmap='YlGn_r', origin='lower', vmin=vmin, vmax=vmax)
    scatter = ax.scatter([hopper.x for hopper in leafhoppers], 
                         [hopper.y for hopper in leafhoppers], c='red', label="Leafhoppers")

    def update(epoch):
        regenerate_field(field)
        new_leafhoppers = []
        
        # Track leafhopper positions and their colors based on actions
        x_positions = []
        y_positions = []
        colors = []

        for hopper in leafhoppers[:]:
            if not hopper.die():
                hopper.step()  # Update the hopper's state (action is chosen in the process)
                
                # Store positions
                x_positions.append(hopper.x)
                y_positions.append(hopper.y)
                
                # Assign color based on current action
                if hopper.current_action == "move":
                    colors.append("blue")
                elif hopper.current_action == "eat":
                    colors.append("green")
                elif hopper.current_action == "rest":
                    colors.append("red")
                
                offspring = hopper.reproduce()
                if offspring:
                    new_leafhoppers.append(offspring)
            else:
                leafhoppers.remove(hopper)

        leafhoppers.extend(new_leafhoppers)
        
        # Update the field and scatter plot
        img.set_array(field)
        
        # Update scatter plot with new positions and colors
        scatter.set_offsets(np.column_stack((x_positions, y_positions)))
        scatter.set_color(colors)
        
        ax.set_title(f"Epoch {epoch}: {len(leafhoppers)} leafhoppers")
         # Create legend proxies
        move_proxy = mlines.Line2D([], [], color='blue', marker='o', linestyle='None', markersize=10, label='Move')
        eat_proxy = mlines.Line2D([], [], color='green', marker='o', linestyle='None', markersize=10, label='Eat')
        rest_proxy = mlines.Line2D([], [], color='red', marker='o', linestyle='None', markersize=10, label='Rest')
        
        # Add the legend to the plot
        ax.legend(handles=[move_proxy, eat_proxy, rest_proxy], loc="upper right", title="Leafhopper Actions")

    ani = FuncAnimation(fig, update, frames=num_epochs, repeat=False)
    plt.show()

if __name__ == "__main__":
    num_epochs = 2000  # Or allow command-line arguments for these values
    num_hoppers = 50
    field_size = 100
    run_simulation(num_epochs=num_epochs, num_hoppers=num_hoppers, field_size=field_size)
