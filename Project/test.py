import numpy as np
from pyamaze import maze, agent

# Function to convert pyamaze maze to numpy array
def maze_to_numpy(m, size):
    grid = np.ones((2 * size + 1, 2 * size + 1), dtype=int)  # Initialize all walls as 1
    for cell in m.maze_map:
        x, y = cell
        grid[2 * x - 1, 2 * y - 1] = 0  # Mark the cell as unoccupied
        if m.maze_map[cell]['E']:
            grid[2 * x - 1, 2 * y] = 0  # East
        if m.maze_map[cell]['W']:
            grid[2 * x - 1, 2 * y - 2] = 0  # West
        if m.maze_map[cell]['N']:
            grid[2 * x - 2, 2 * y - 1] = 0  # North
        if m.maze_map[cell]['S']:
            grid[2 * x, 2 * y - 1] = 0  # South
    return grid

# Create an 11x11 maze
maze_size = 5  # pyamaze size is halved for numpy grid
m = maze(maze_size, maze_size)
m.CreateMaze()

# Convert pyamaze maze to numpy array
maze_array = maze_to_numpy(m, maze_size)
print(maze_array)
