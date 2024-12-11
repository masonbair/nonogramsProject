import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from matplotlib.animation import FuncAnimation

#USE THIS FOR UPDATING THE SQUARES IN THE GRID
def update(frame):
    global grid
    grid = grid
    #This is how you can color the grid
    grid[2,2] = 1
    grid[2,1] = 1
    grid[0,2] = 1
    grid[0,0] = 1


    ax_grid.clear()
    ax_grid.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')

    #Used for displaying grid lines and making sure no x-axis and y-axis numbers are visible.
    ax_grid.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
    ax_grid.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
    ax_grid.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    
    add_clues()

def add_clues():
    # Rows
    for i, clue in enumerate(rows):
        ax_grid.text(-0.5, len(rows)-1-i, ' '.join(map(str, clue)), 
                     ha='right', va='center', fontsize=12)
    # Columns
    for i, clue in enumerate(cols):
        ax_grid.text(len(cols)-1-i, -0.5, '\n'.join(map(str, clue)), 
                     ha='left', va='bottom', fontsize=12)


#________________GRID SET UP_____________________________________
cmap = colors.ListedColormap(["white", "black", "gray"])
bounds = [-1.5, -0.5, 0.5, 1.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

rows = [(1,2), (3,), (4,1), (2,2), (1,), (4,1), (1,1), (1,), (1,), (1,1), (1,), (1,), (1,1), (1,), (1,)]
cols = [(2,), (1,1), (3,1), (1,1), (2,), (1,3), (1,1), (1,), (1,), (1,1), (1,), (1,), (1,1), (1,), (1,)]
grid = np.full((len(rows), len(cols)), -1)

fig, ax = plt.subplots(figsize=(8, 8))
gs = fig.add_gridspec(12, 12, wspace=0.2, hspace=0.2)
ax_grid = fig.add_subplot(gs[2:, 2:])

ax_grid.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
ax_grid.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
ax_grid.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
ax_grid.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
ax_grid.set_xticks([])
ax_grid.set_yticks([])
add_clues()
ax.axis("off")
###################################################################

ani = FuncAnimation(fig, update, frames=10, repeat=False)
plt.show()