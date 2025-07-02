
import numpy as np

class ForagingArena:
    def __init__(self, goal_locations, arena_size=(10, 10)):
        self.goal_locations = goal_locations
        self.arena_size = arena_size
        self.position = np.zeros(2)
        self.hunger_level = 1.0

    def reset(self, hunger_level=1.0):
        self.position = np.random.uniform(0, self.arena_size[0], size=2)
        self.hunger_level = hunger_level
        return np.concatenate([self.position, [self.hunger_level]])

    def step(self, action):
        direction = np.tanh(action[:2])  # Clip movement
        speed = np.abs(action[2]) * self.hunger_level  # modulate by hunger
        move = direction * speed
        self.position += move
        self.position = np.clip(self.position, 0, self.arena_size[0])

        reward = 0.0
        for goal in self.goal_locations:
            dist = np.linalg.norm(self.position - goal)
            if dist < 0.5:
                reward += 1.0

        obs = np.concatenate([self.position, [self.hunger_level]])
        done = reward > 0
        return obs, reward, done, {}
