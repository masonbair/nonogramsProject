import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import numpy as np
from matplotlib.animation import FuncAnimation
import time

# Nonogram solver
# A nonogram is a triple consisting of an m*n bit matrix,
# a row hint array of length m
# and a column hint array of length n.
# The bit matrix cells can take values 0 (white), 1 (black) and -1 (white, marked).
# Marked cells will be handled differently by the algorithm.
# In the end, if the algorithm succeeds, all cells will be black or marked.
# If it does not, our job is to solve the rest manually and make up more heuristics based on that.
def solve(field, rowhints, colhints):

    cmap = colors.ListedColormap(["grey", "white", "black"])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    def create_figure(iteration_num):
        """Creates a new figure for each iteration"""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(field, cmap=cmap, norm=norm, interpolation='nearest')
        
        # Add title showing iteration number
        plt.title(f'Iteration {iteration_num}', pad=20)
        
        # Add row hints
        for i, hint in enumerate(rowhints):
            ax.text(-0.5, i, ' '.join(map(str, hint)), ha='right', va='center')
            
        # Add column hints
        for i, hint in enumerate(colhints):
            ax.text(i, -0.5, '\n'.join(map(str, hint)), ha='center', va='bottom')
            
        # Add grid
        ax.set_xticks(np.arange(-.5, len(colhints), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(rowhints), 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax.set_xticks([])
        ax.set_yticks([])
        
        plt.tight_layout()
        plt.show(block=False)


    # We assume hints are correct, that is, a row's hint doesn't exceed the row's length in size.
    # Initialise some helpful constants
    ROW = 0
    COL = 1
    WIDTH = len(colhints)
    HEIGHT = len(rowhints)

    changed = True


    # The simple heuristic goes in front because it will only be used once.
    for i, hint in enumerate(rowhints):
        # Simple heuristic: if the hints describe the row exactly,
        # then make the hint into this full row and plug it in
        if sum(hint) + len(hint)-1 == WIDTH:
            full_row = []
            for nr in hint:
                full_row += [1] * nr + [-1]
            # Remove the last space
            full_row = full_row[:-1]
            field[i] = full_row
    for i, hint in enumerate(colhints):
        # Simple heuristic
        if sum(hint) + len(hint)-1 == HEIGHT:
            full_col = []
            for nr in hint:
                full_col += [1] * nr + [-1]
            full_col = full_col[:-1]
            # A column must be changed with a cycle
            for j, row in enumerate(field):
                row[i] = full_col[j]

    # Create a mapping between hints and row or column indices:
    # which squares are eligible to be coloured in relation to a particular hint?
    hint_ranges = {}
    row_col_index = 0
    for r, hint in enumerate(rowhints):
        for i in range(len(hint)):
            # Put the preceeding and succeeding hints into variables
            previous_hints = hint[:i]
            if i+1 < len(hint):
                next_hints = hint[i+1:]
            else:
                next_hints = []
            # Get the first evaluation of the range based on the above
            hint_range = range(*(sum(previous_hints) + len(previous_hints), WIDTH - sum(next_hints) - len(next_hints)))
            hint_ranges[(row_col_index, i)] = hint_range
        row_col_index += 1
    for r, hint in enumerate(colhints):
        for i in range(len(hint)):
            # Put the preceeding and succeeding hints into variables
            previous_hints = hint[:i]
            if i+1 < len(hint):
                next_hints = hint[i+1:]
            else:
                next_hints = []
            # Get the first evaluation of the range based on the above
            hint_range = range(*(sum(previous_hints) + len(previous_hints), HEIGHT - sum(next_hints) - len(next_hints)))
            hint_ranges[(row_col_index, i)] = hint_range
        row_col_index += 1

    # If we're starting or the field was changed
    iteration = 0
    while changed and iteration <= max(WIDTH, HEIGHT) + 3:
        changed = False
        # print("iteration", iteration, hint_ranges)
        iteration += 1
        # Try all the heuristics on all the rows and columns
        for i, hint in enumerate(rowhints):
            # If the hint is larger than half of the squares it can occupy, fill the middle ones
            for j, nr in enumerate(hint):
                other_lengths = sum(hint) + len(hint)-1 - nr
                leave_empty = WIDTH - other_lengths - nr # This is greater than 0 because the 0-case is handled by the simple heuristic.
                if nr > leave_empty:
                    hints_before = sum(hint[:j]) + len(hint[:j])
                    hints_after = sum(hint[j:]) + len(hint[j:])-1 - nr
                    for fill in range(hints_before+leave_empty, WIDTH-hints_after-leave_empty):
                        if field[i][fill] == 0:
                            changed = True
                            field[i][fill] = 1
                            # print(fill, i, "middle square in whole row")
            # Mark empty squares of complete rows
            if 0 in field[i] and check_solved(field[i], hint):
                # if not check_gaps(field[i], hint):
                #     raise ValueError(f"Row {i} does not match hints: {hint}")
                field[i] = [-1 if square < 1 else 1 for square in field[i]]
                # print("row", i, "is complete")
                changed = True
            # Close gaps too small to contain any hints
            if 0 in field[i]:
                # Look for sequences of unmarked squares not next to a black one
                gap_size = 0
                for j, space in enumerate(field[i]):
                    if space == 0:
                        gap_size += 1
                    elif space == 1:
                        gap_size = -99 # THIS IS A CONSTANT NOT SMALL ENOUGH IT MUST BE MADE SMALLEST
                    else:
                        if gap_size > 0 and gap_size < min(hint):
                            for gap_index in range(1, gap_size+1):
                                # print(i, j - gap_index, "small row gap")
                                
                                field[i][j - gap_index] = -1
                            changed = True
                        gap_size = 0
                # If there is a gap in the end, only check with the last hint
                if gap_size > 0 and gap_size < hint[-1]:
                    for gap_index in range(1, gap_size+1):
                        # print(i, WIDTH - gap_index, "small row gap in the end")
                        field[i][-gap_index] = -1
                    changed = True
            # If there are as many unmarked squares as the hints have left, colour them all
            if 0 in field[i] and sum(hint) == sum([1 if square > -1 else 0 for square in field[i]]):
                field[i] = [1 if square > -1 else -1 for square in field[i]]
                # print(i, "unmarked square left in hints in row")
                changed = True
        for i, hint in enumerate(colhints):
            # If the hint is larger than half of the squares it can occupy, fill the middle ones
            for j, nr in enumerate(hint):
                bonus = 0
                other_lengths = sum(hint) + len(hint)-1 - nr
                leave_empty = HEIGHT - other_lengths - nr # This is greater than 0 because the 0-case is handled by the simple heuristic.
                if nr > leave_empty:
                    hints_before = sum(hint[:j]) + len(hint[:j])
                    hints_after = sum(hint[j:]) + len(hint[j:])-1 - nr
                    for fill in range(hints_before+leave_empty+bonus, HEIGHT-hints_after-leave_empty):
                        if field[fill][i] == 0:
                            ### print("It does something")
                            changed = True
                            field[fill][i] = 1
                            # print(fill, i, "middle square in whole column")
            # Mark empty squares of complete columns
            column = [row[i] for row in field]
            if 0 in column and check_solved(column, hint):
                for j, row in enumerate(field):
                    if row[i] == 0:
                        ## print("marked")
                        # print(i, j, "column is complete")
                        row[i] = -1
                changed = True
            # Close gaps too small to contain any hints
            if 0 in column:
                # Look for sequences of unmarked squares not next to a black one
                gap_size = 0
                for j, space in enumerate(column):
                    if space == 0:
                        gap_size += 1
                    elif space == 1:
                        gap_size = -99 # THIS IS A CONSTANT NOT SMALL ENOUGH IT MUST BE MADE SMALLEST
                    else:
                        if gap_size > 0 and gap_size < min(hint):
                            for gap_index in range(1, gap_size+1):
                                # print(j - gap_index, i, "small column gap")
                                field[j - gap_index][i] = -1
                            changed = True
                        gap_size = 0
                # If there is a gap in the end, only check with the last hint
                if gap_size > 0 and gap_size < hint[-1]:
                    for gap_index in range(1, gap_size+1):
                        # print(HEIGHT - gap_index, i, "small column gap in the end")
                        field[-gap_index][i] = -1
                    changed = True
            # If there are as many unmarked squares as the hints have left, colour them all
            if 0 in column and sum(hint) == sum([1 if square > -1 else 0 for square in column]):
                for row in field:
                    if row[i] == 0:
                        # print(row, i, "unmarked square left in hints in column")
                        row[i] = 1
                changed = True

            # Create a mapping between hints and row or column indices:
            # which squares are eligible to be coloured in relation to a particular hint?
            row_col_index = 0
            for r, hint in enumerate(rowhints):
                # Evaluate the situation of black squares
                black_index = []
                for s, v in enumerate(field[r]):
                    if v == 1:
                        black_index += [s]
                # Check which hints each square could belong to and narrow the ranges accordingly
                for black_number in black_index:
                    black_overlaps = []
                    for hint_number in hint_ranges.keys():
                        if hint_number[0] == row_col_index and black_number in hint_ranges[(hint_number)]:
                            black_overlaps += [hint_number]
                    # Adjust the range if there is a single overlap
                    if len(black_overlaps) == 1:
                        previous = hint_ranges[black_overlaps[0]]
                        hint_ranges[black_overlaps[0]] = range(*(max(hint_ranges[black_overlaps[0]][0], black_number - hint[black_overlaps[0][1]] + 1), min(hint_ranges[black_overlaps[0]][-1] + 1, black_number + hint[black_overlaps[0][1]])))
                        if previous != hint_ranges[black_overlaps[0]]:
                            changed = True
                # Exclude gaps from the ranges
                for i, hint_value in enumerate(hint):
                    for j, index in enumerate(hint_ranges[r, i]):
                        if field[r][index] == -1:
                            try:
    #                             if r == 5 and i == 0:
    #                                 # print(j, index, hint_ranges[r, i], hint_ranges[r, i][0], index - hint_ranges[r, i][0], hint_value)
                                if index - hint_ranges[r, i][0] < hint_value:
                                    hint_ranges[r, i] = range(index + 1, hint_ranges[r, i][-1] + 1)
                                    changed = True
                                if hint_ranges[r, i][-1] - index < hint_value:
                                    hint_ranges[r, i] = range(hint_ranges[r, i][0], index)
                                    changed = True
                            except:
                              pass
                    # Narrow the range based on the previous hint's range
                    if i > 0:
                        hint_ranges[r, i] = range(max(hint_ranges[r, i-1][0] + hint[i-1], hint_ranges[r, i][0]), hint_ranges[r, i][-1] + 1)
                    # Narrow the range based on the next hint's range
                    if i < len(hint) - 1:
                        hint_ranges[r, i] = range(hint_ranges[r, i][0], min(hint_ranges[r, i][-1] + 1, hint_ranges[r, i+1][-1] - hint[i+1]))
                row_col_index += 1
            for r, hint in enumerate(colhints):
                # Evaluate the situation of black squares
                black_index = []
                for s, row in enumerate(field):
                    if row[r] == 1:
                        black_index += [s]
                # Check which hints each square could belong to and narrow the ranges accordingly
                for black_number in black_index:
                    black_overlaps = []
                    for hint_number in hint_ranges.keys():
                        if hint_number[0] == row_col_index and black_number in hint_ranges[(hint_number)]:
                            black_overlaps += [hint_number]
                    # Adjust the range if there is a single overlap
                    if len(black_overlaps) == 1:
                        previous = hint_ranges[black_overlaps[0]]
                        hint_ranges[black_overlaps[0]] = range(*(max(hint_ranges[black_overlaps[0]][0], black_number - hint[black_overlaps[0][1]] + 1), min(hint_ranges[black_overlaps[0]][-1] + 1, black_number + hint[black_overlaps[0][1]])))
                        if previous != hint_ranges[black_overlaps[0]]:
                            changed = True
                ## print(hint_ranges)
#                 for row in field:
#                     # print(row)
#                 # print()
                # Exclude gaps from the ranges
                for i, hint_value in enumerate(hint):
                    for j, index in enumerate(hint_ranges[r + HEIGHT, i]):
                        if field[index][r] == -1:
                            try:
                            ## print("282. rida:", r + HEIGHT, i, hint_ranges[r + HEIGHT, i], index)
                                if index - hint_ranges[r + HEIGHT, i][0] < hint_value:
                                    hint_ranges[r + HEIGHT, i] = range(index + 1, hint_ranges[r + HEIGHT, i][-1] + 1)
                                    changed = True
                                ## print(r + HEIGHT, i, hint_ranges[r + HEIGHT, i])
                                if hint_ranges[r + HEIGHT, i][-1] - index < hint_value:
                                    hint_ranges[r + HEIGHT, i] = range(hint_ranges[r + HEIGHT, i][0], index)
                                    changed = True
                            except:
                               pass
                    # Narrow the range based on the previous hint's range
                    if i > 0:
                        hint_ranges[r + HEIGHT, i] = range(max(hint_ranges[r + HEIGHT, i-1][0] + hint[i-1], hint_ranges[r + HEIGHT, i][0]), hint_ranges[r + HEIGHT, i][-1] + 1)
                    # Narrow the range based on the next hint's range
                    if i < len(hint) - 1:
                        hint_ranges[r + HEIGHT, i] = range(hint_ranges[r + HEIGHT, i][0], min(hint_ranges[r + HEIGHT, i][-1] + 1, hint_ranges[r + HEIGHT, i+1][-1] - hint[i+1]))
                row_col_index += 1
            ## print(hint_ranges)
#             for row in field:
#                 # print(row)

        # Apply all the heuristics on the small ranges for each hint
        for r, hint in enumerate(rowhints):
            positions_covered = []
            for h, hint_value in enumerate(hint):
                for index in hint_ranges[r, h]:
                    positions_covered += [index]
                if len(hint_ranges[r, h]) == hint_value:
                    for position in hint_ranges[r, h]:
                        if field[r][position] == 0:
                            field[r][position] = 1
                            # print(r, position, "full square in row range")
                            changed = True
                    if hint_ranges[r, h][0] > 0 and field[r][hint_ranges[r, h][0] - 1] == 0:
                        # print(r, hint_ranges[r, h][0] - 1, "gray before full row range")
                        field[r][hint_ranges[r, h][0] - 1] = -1
                        changed = True
                    if hint_ranges[r, h][-1] < WIDTH - 1 and field[r][hint_ranges[r, h][-1] + 1] == 0:
                        # print(r, hint_ranges[r, h][-1] + 1, "gray after full row range")
                        field[r][hint_ranges[r, h][-1] + 1] = -1
                        changed = True
                elif len(hint_ranges[r, h]) < 2 * hint_value:
                    leave_empty = len(hint_ranges[r, h]) - hint_value
                    for position in hint_ranges[r, h][leave_empty:-leave_empty]:
                        if field[r][position] == 0:
                            field[r][position] = 1
                            # print(r, position, "middle square in row range")
                            changed = True
            for position in range(WIDTH):
                if position not in positions_covered and field[r][position] == 0:
                    # print(r, position, "not in row ranges")
                    field[r][position] = -1
                    changed = True
        for r, hint in enumerate(colhints):
            positions_covered = []
            for h, hint_value in enumerate(hint):
                for index in hint_ranges[r + HEIGHT, h]:
                    positions_covered += [index]
                if len(hint_ranges[r + HEIGHT, h]) == hint_value:
                    for position in hint_ranges[r + HEIGHT, h]:
                        if field[position][r] == 0:
                            field[position][r] = 1
                            # print(position, r, "full square in column range")
                            changed = True
                    if hint_ranges[r + HEIGHT, h][0] > 0 and field[hint_ranges[r + HEIGHT, h][0] - 1][r] == 0:
                        # print(hint_ranges[r + HEIGHT, h][0] - 1, r, "gray before full column range")
                        field[hint_ranges[r + HEIGHT, h][0] - 1][r] = -1
                        changed = True
                    if hint_ranges[r + HEIGHT, h][-1] < HEIGHT - 1 and field[hint_ranges[r + HEIGHT, h][-1] + 1][r] == 0:
                        # print(hint_ranges[r + HEIGHT, h][-1] + 1, r, "gray after full column range")
                        field[hint_ranges[r + HEIGHT, h][-1] + 1][r] = -1
                        changed = True
                elif len(hint_ranges[r + HEIGHT, h]) < 2 * hint_value:
                    leave_empty = len(hint_ranges[r + HEIGHT, h]) - hint_value
                    for position in hint_ranges[r + HEIGHT, h][leave_empty:-leave_empty]:
                        if field[position][r] == 0:
                            field[position][r] = 1
                            # print(position, r, "middle square in column range")
                            changed = True
            for position in range(HEIGHT):
                if position not in positions_covered and field[position][r] == 0:
                    # print(position, r, "not in column ranges")
                    field[position][r] = -1
                    changed = True
            

        if changed:
            create_figure(iteration)

    # Once the nonogram is solved or the algorithm is out of ideas, it stops

    #  Creates the final figure
    plt.figure(figsize=(10, 10))
    plt.title('Final Solution', pad=20)
    plt.imshow(field, cmap=cmap, norm=norm, interpolation='nearest')
    
    # Add hints and grid to final figure
    ax = plt.gca()
    for i, hint in enumerate(rowhints):
        ax.text(-0.5, i, ' '.join(map(str, hint)), ha='right', va='center')
    for i, hint in enumerate(colhints):
        ax.text(i, -0.5, '\n'.join(map(str, hint)), ha='center', va='bottom')
    ax.set_xticks(np.arange(-.5, len(colhints), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(rowhints), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

    return field

def check_solved(row, rowhints):
# Check if a row has its hints exhausted.
# Also works with columns.
    hints = rowhints[:] # So we can modify them while counting.
    for i in range(len(row)):
        # current_hint_number = hints[0]
        if row[i] == 1:
            # A sequence of black squares can't be longer than the current hint.
            if hints[0] < 1:
                return False
            else:
                hints[0] -= 1
        elif len(hints) > 0:
            # An exhausted hint is removed.
            # If it is not exhausted by the time a gap is reached, there is more work to do.
            if hints[0] == 0:
                hints.pop(0)
            elif i > 0 and row[i-1] == 1:
                return False
    return sum(hints) == 0

#USE THIS FOR UPDATING THE SQUARES IN THE GRID
def update(frame):
    global grid
    grid = grid
    #This is how you can color the grid
    for i, _ in enumerate(rowhints):
        for j, _ in enumerate(colhints):
            grid[i][j] = solved[i][j]

    ax_grid.clear()
    ax_grid.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')

    #Used for displaying grid lines and making sure no x-axis and y-axis numbers are visible.
    ax_grid.set_xticks(np.arange(-.5, len(colhints), 1), minor=True)
    ax_grid.set_yticks(np.arange(-.5, len(rowhints), 1), minor=True)
    ax_grid.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])
    
    add_clues()

def add_clues():
    # Rows
    for i, clue in enumerate(rowhints):
        ax_grid.text(-0.5, i, ' '.join(map(str, clue)), 
                     ha='right', va='center', fontsize=12)
    # Columns
    for i, clue in enumerate(colhints):
        ax_grid.text(i, -0.5, '\n'.join(map(str, clue)), 
                     ha='left', va='bottom', fontsize=12)


areas = []
times = []

# start_time = time.time()
# field = [[0 for _ in range(7)] for _ in range(3)]
# rowhints = [[1, 1], [7], [1, 1]]
# colhints = [[1], [3], [1], [1], [1], [3], [1]]
# solved = solve(field, rowhints, colhints)
# end_time = time.time()
# areas.append(21)
# times.append(end_time-start_time)
# print("First Nonogram of area 21 solved in: ", end_time-start_time)
# # for row in solved:
# #     # print(row)

# print()
# start_time = time.time()
# field = [[0 for _ in range(7)] for _ in range(3)]
# rowhints = [[5], [1, 4], [4, 1]]
# colhints = [[2], [1, 1], [1, 1], [3], [2], [3], [1]]
# solved = solve(field, rowhints, colhints)
# end_time = time.time()
# areas.append(21)
# times.append(end_time-start_time)
# print("Second Nonogram of area 21 solved in: ", end_time-start_time)
# # for row in solved:
# #     # print(row)

# print()
# start_time = time.time()
# field = [[0 for _ in range(4)] for _ in range(5)]
# rowhints = [[1,1],[2,1],[1], [1,2], [2]]
# colhints = [[2,1], [1], [1,3], [1,2]]
# solved = solve(field, rowhints, colhints)
# end_time = time.time()
# areas.append(20)
# times.append(end_time-start_time)
# print("Third Nonogram of area 20 solved in: ", end_time-start_time)
# # for row in solved:
# #     # print(row)

# #Bird shaped
# print()
# start_time = time.time()
# field = [[0 for _ in range(6)] for _ in range(6)]
# rowhints = [[2,1],[1,3],[1,2], [3], [4], [1]]
# colhints = [[1], [5], [2], [5], [2,1], [2]]
# solved = solve(field, rowhints, colhints)
# end_time = time.time()
# areas.append(36)
# times.append(end_time-start_time)
# print("Fourth Nonogram of area 36 solved in: ", end_time-start_time)
# # for row in solved:
# #     # print(row)

# # #Duck shaped
# print()
# start_time = time.time()
# field = [[0 for _ in range(8)] for _ in range(9)]
# rowhints = [[3],[2,1],[3,2], [2,2], [6], [1,5], [6], [1], [2]]
# colhints = [[1,2], [3,1], [1,5], [7,1], [5], [3], [4], [3]]
# solved = solve(field, rowhints, colhints)
# end_time = time.time()
# areas.append(72)
# times.append(end_time-start_time)
# print("Fifth Nonogram of area 72 solved in: ", end_time-start_time)
# # for row in solved:
# #     # print(row)

# # # Seahorse
# print()
# start_time = time.time()
# field = [[0 for _ in range(6)] for _ in range(7)]
# rowhints = [[4],[2,1],[3], [3], [4], [1,2], [3]]
# colhints = [[1], [1,1], [2,2,1], [1,5], [6], [1,1,1]]
# solved = solve(field, rowhints, colhints)
# end_time = time.time()
# areas.append(42)
# times.append(end_time-start_time)
# print("Sixth Nonogram of area 42 solved in: ", end_time-start_time)
# # for row in solved:
# #     # print(row)

# # #Dolphin
# print()
# start_time = time.time()
# field = [[0 for _ in range(13)] for _ in range(12)]
# rowhints = [[7,5],[6,5],[3,3], [2,2], [2,1,1], [1], [6,2], [5,1,3], [11,1],[11,1],[9,2],[10,2]]
# colhints = [[5,6], [5,6], [3,6], [2,1,6], [2,6], [2,1,4], [1,5], [4], [2,7],[2,4,1],[3,3],[4,2],[5,4]]
# solved = solve(field, rowhints, colhints)
# end_time = time.time()
# areas.append(156)
# times.append(end_time-start_time)
# print("Seventh Nonogram of area 157 solved in: ", end_time-start_time)
# # for row in solved:
# #     # print(row)

# print()
# start_time = time.time()
# field = [[0 for _ in range(24)] for _ in range(20)]
# rowhints = [
#     [4, 1],
#     [8, 2,6],
#     [21],
#     [22],
#     [22],
#     [6, 4, 8],
#     [1, 2, 6],
#     [2, 1, 4],
#     [1, 2],
#     [2,1],
#     [1, 2],
#     [2,1],
#     [4,2],
#     [6,4],
#     [3,3,6],
#     [2, 4, 3,3],
#     [8, 2, 4],
#     [6,8],
#     [4, 6],
#     [4]
# ]

# colhints = [
#     [1],
#     [3],
#     [4], 
#     [5],
#     [6,3],
#     [6,5],
#     [6,3,3],
#     [6,3,3],
#     [5,5,4],
#     [4,3,7],
#     [6,5],
#     [6,3],
#     [5],
#     [5,3],
#     [3,3,5],
#     [4,3,3,3],
#     [5,5,3],
#     [5,3,4],
#     [6,7],
#     [6,5],
#     [7,3],
#     [6],
#     [4],
#     [3]
# ]

# solved = solve(field, rowhints, colhints)
# end_time = time.time()
# areas.append(480)
# times.append(end_time-start_time)
# print("Eigth Nonogram of area 480 solved in: ", end_time-start_time)

# # Parrot
# print()
# start_time = time.time()
# field = [[0 for _ in range(35)] for _ in range(30)]
# rowhints = [[8], [1, 13], [1, 17], [3, 20], [31], [21, 7], [18, 7], [1, 17, 3, 3, 2], [21, 2, 2, 2, 2], [17, 2, 3, 1], [23, 5, 1], [3, 20, 7], [19, 8], [21, 6], [7, 11, 3, 4], [9, 19], [10, 13, 2], [24, 2], [24, 1], [25], [25], [25], [11, 13], [11, 5, 4], [11, 5, 2, 3], [11, 8, 4], [11, 13], [11, 12], [10, 13], [10, 13]]
# colhints = [[3, 11], [2, 2, 12], [3, 1, 1, 13], [2, 1, 1, 13], [3, 1, 2, 14], [5, 1, 14], [8, 16], [26], [26], [26], [24], [19, 2], [27], [12, 15], [14, 13], [14, 13], [15, 13], [22, 5], [23, 6], [23, 6], [23, 4], [24, 5], [6, 21], [5, 1, 20], [5, 2, 17], [6, 2, 4, 3], [6, 2, 4], [9, 5], [7, 3, 3], [6, 5, 2], [4, 6, 1], [3, 8, 2], [2, 8], [2, 6], [8]]
# solved = solve(field, rowhints, colhints)
# end_time = time.time()
# areas.append(1050)
# times.append(end_time-start_time)
# print("Eigth Nonogram of area 1050 solved in: ", end_time-start_time)
# for row in solved:
#     print(row)

# # Owl
# print()
# start_time = time.time()
# field = [[0 for _ in range(40)] for _ in range(40)]

# rowhints = [[1],[2,1],[2,1],[2,2],[3,3],[4,1,1,3],[1,5,1,1,5],[2,8,9,8,1],[39],[37],[35],[2,4,17,7],[2,3,15,3,2],[2,2,13,2,2],[5,2,9,2,5],[7,7,7],[7,5,5,6,7],[7,7,3,8,6],[1,5,2,2,2,1,3,2,2,7],[1,5,5,2,1,1,5,6],[1,5,4,4,4,3,5,1],[1,5,2,1,1,3,5,1],[2,2,4,3,1,1,4,4,2,2],[2,1,13,13,2,2],[2,12,2,2,10,1,2],[1,11,2,1,8,1,1],[2,9,1,1,7,2],[2,1,5,2,2,5,1,2],[4,3,1,1,3,4],[5,2,2,1,3,1],[8,2,2,1,7],[7,5,2,2,5,7],[13,1,3,1,13],[2,10,3,1,3,12],[13,4,1,4,10],[12,6,6,11],[2,7,7,7,7,2],[1,17,9,7,1],[1,4,2,9,9,2,4,1],[1,4,6,11,6,5]]
# colhints = [[3,2,3],[5,2,3,12],[4,2,16,3],[7,2,10,3],[7,7,12],[7,11,11],[5,10,2,10],[5,13,8],[1,5,14,1,8],[7,3,5,9],[6,1,2,6,7,1],[3,2,1,4,6,5,3],[4,2,2,2,6,4,4],[5,1,5,6,2,5],[6,5,5,5],[6,2,5,7],[8,3,8,7],[9,7,3,8],[10,2,3,3,5],[11,7,3,3],[12,3,1],[11,6,3,3],[10,3,2,3,5],[9,2,4,3,8],[8,12,7],[6,2,5,7],[6,4,6,2,5],[5,1,5,6,2,5],[4,2,2,2,6,3,4],[3,2,6,6,5,3],[6,1,3,6,6,1],[7,3,5,9],[1,5,5,5,2,8],[5,13,8],[16,10],[7,11,11],[7,8,12],[7,4,10,3],[4,2,11,7,1],[4,2,3,6,4]]
# solved = solve(field, rowhints, colhints)
# end_time = time.time()
# areas.append(1600)
# times.append(end_time-start_time)
# print("Eigth Nonogram of area 1600 solved in: ", end_time-start_time)
# # for row in solved:
#     # print(row)

# #BIG TESTER
# print()
# start_time = time.time()
# field = [[0 for _ in range(45)] for _ in range(45)]
# rowhints = [[5, 6], [5, 6], [6, 6], [6, 6, 1], [4, 6, 5, 1], [6, 7, 4, 2], [7, 8, 3, 2], [8, 9, 3, 3], [9, 9, 3, 3], [10, 9, 2, 3], [12, 9, 6], [6, 5, 10, 4], [1, 6, 8, 4], [4, 8, 7, 5], [8, 24], [13, 22], [7, 7, 18], [8, 7, 9, 11], [9, 6, 10, 7], [10, 6, 7, 3, 5], [1, 8, 7, 7, 2, 1, 4], [1, 3, 9, 6, 6, 2], [2, 3, 7, 10, 3, 3], [3, 4, 7, 12, 5], [5, 4, 6, 21], [5, 4, 6, 21], [6, 1, 4, 6, 4, 16], [6, 1, 6, 9, 1, 14], [5, 2, 6, 7, 1, 1, 8, 2], [5, 3, 8, 5, 1, 1, 2, 6, 1], [5, 4, 8, 3, 2, 1, 6, 1], [5, 5, 9, 2, 2, 7], [6, 7, 11, 1, 3, 3], [6, 8, 7, 3, 2, 2], [7, 8, 5, 3, 1, 1], [7, 8, 3, 3, 1, 1], [8, 7, 3, 3], [9, 7, 3, 4], [10, 6, 3, 5], [11, 4, 4, 1], [12, 4, 7], [14, 3, 5], [15, 3, 2], [17, 2, 2], [19, 1]]
# colhints = [[9, 6, 16], [8, 5, 20], [8, 6, 23], [8, 6, 23], [1, 7, 7, 22], [3, 7, 8, 4, 13], [4, 5, 10, 11], [5, 4, 3, 7, 9], [6, 3, 4, 3, 3, 7, 8], [6, 3, 5, 2, 3, 6, 7], [6, 2, 4, 5, 7, 6], [6, 3, 5, 2, 5, 7, 5], [1, 5, 3, 9, 5, 7, 4], [2, 5, 3, 9, 4, 7, 4], [3, 5, 3, 9, 4, 7, 3], [4, 5, 3, 9, 5, 7, 3], [5, 5, 4, 9, 4, 8, 2], [5, 5, 3, 1, 8, 4, 7, 2], [5, 5, 3, 2, 2, 5, 4, 6, 1], [4, 4, 3, 3, 2, 4, 4, 5], [4, 4, 2, 4, 2, 4, 3, 4], [3, 4, 7, 3, 3, 4], [3, 3, 12, 3, 3], [2, 20, 3], [2, 20, 3], [26], [6, 7, 1, 16], [6, 5, 2, 5, 5], [5, 5, 2, 10, 4], [3, 5, 3, 4, 4, 3], [3, 4, 1, 4, 3, 4], [2, 4, 1, 5, 4, 4], [7, 3, 6, 4, 3], [7, 2, 4, 9], [7, 2, 6, 2, 4], [8, 8, 2, 2], [16, 3], [17, 1], [5, 14], [11], [9], [8], [3, 4], [3, 5], [7]]
# solved = solve(field, rowhints, colhints)
# end_time = time.time()
# areas.append(2025)
# times.append(end_time-start_time)
# print("Eigth Nonogram of area 2025 solved in: ", end_time-start_time)
# for row in solved:
    # print(row)

# # Heart Shaped
# print()
# field = [[0 for _ in range(9)] for _ in range(10)]
# rowhints = [[1,1],[3,3],[1,1,1], [1,1], [1,1], [1,1], [1,1], [1,1], [3],[1]]
# colhints = [[4], [1,1], [2,1], [1,1], [1,2], [1,1], [2,1], [1,1], [4]]
# solved = solve(field, rowhints, colhints)
# for row in solved:
#     print(row)

#BIGGG BOYYYY
print()
field = [[0 for _ in range(100)] for _ in range(90)]
rowhints = [[3, 3, 1], [4, 4, 1], [5, 1, 4, 1], [6, 3, 5, 4, 1], [7, 8, 6, 5, 4, 1], [9, 11, 9, 2, 4, 1], [10, 15, 17, 1, 5, 1], [3, 26, 17, 5, 1], [4, 18, 8, 20, 4, 1], [4, 18, 7, 24, 4, 1], [5, 17, 7, 19, 4, 1], [6, 15, 7, 23, 3, 1], [2, 3, 15, 24, 6, 5], [2, 3, 13, 2, 4, 9, 3, 4, 5], [2, 4, 13, 2, 3, 8, 1, 1, 4, 4], [2, 3, 39, 3, 4, 4], [2, 4, 15, 1, 12, 2, 3, 4], [3, 4, 12, 8, 2, 3, 4], [3, 4, 9, 4, 6, 5, 4], [3, 3, 8, 10, 4, 1, 4, 4], [3, 3, 6, 9, 3, 3, 1, 5, 3, 1], [4, 4, 6, 10, 3, 2, 6, 3, 1], [5, 3, 7, 3, 10, 2, 15, 3, 1], [5, 3, 9, 6, 4, 4, 5, 4, 1], [6, 2, 11, 16, 14, 2, 2, 1], [6, 2, 12, 6, 3, 4, 6, 7, 1], [7, 2, 20, 1, 8, 10, 3, 1, 1, 1], [2, 4, 2, 6, 12, 2, 7, 5, 7, 4, 1], [2, 4, 2, 7, 9, 3, 16, 1, 1, 1, 2], [2, 4, 2, 6, 9, 3, 12, 2, 5, 3], [2, 4, 2, 7, 4, 4, 5, 33, 6], [3, 4, 6, 4, 2, 9, 42], [3, 4, 5, 8, 5, 4, 7, 1], [3, 4, 5, 10, 6, 8, 1], [4, 4, 4, 5, 5, 10, 7, 1], [4, 5, 4, 4, 3, 2,  13, 5, 3, 1], [4, 3, 4, 7, 6, 12, 6, 3, 1], [4, 3, 3, 10, 3, 5, 6, 8, 4, 1], [5, 2, 3, 12, 4, 6, 8, 4, 1], [5, 2, 17, 5, 6, 10, 4, 1], [5, 3, 21, 6, 10, 4, 1], [5, 5, 7, 9, 18, 4, 1], [6, 2, 3, 6, 2, 3, 13, 3, 1], [6, 1, 2, 7, 3, 2, 8, 4, 1], [7, 2, 2, 2, 4, 2, 4, 5, 4, 1], [7, 1, 4, 2, 5, 2, 4, 5, 4, 1], [7, 1, 3, 2, 5, 1, 5, 3, 4, 1], [3, 3, 2, 6, 7, 7, 3, 4, 1], [3, 4, 2, 4, 3, 5, 4, 3, 3, 5, 1], [3, 3, 1, 2, 3, 3, 3, 3, 7, 2, 4, 2], [3, 4, 2, 2, 3, 3, 4, 3, 2, 3, 3, 5, 3], [2, 3, 1, 2, 3, 2, 4, 2, 10, 5], [2, 3, 1, 2, 3, 2, 3, 8, 9, 7], [2, 3, 1, 2, 3, 2, 2, 10, 10, 4, 3], [2, 2, 1, 2, 4, 2, 13, 6, 2, 3, 3], [2, 3, 2, 2, 6, 2, 3, 5, 3, 3, 4], [2, 2, 1, 2, 7, 2, 1, 2, 8, 2, 1], [2, 1, 2, 3, 8, 2, 2, 1, 5, 2, 1], [1, 2, 2, 9, 3, 3, 3, 2, 1], [1, 6, 6, 10, 14, 3, 2, 3, 1], [1, 3, 6, 18, 16, 4, 8, 1], [4, 5, 10, 21, 3, 3, 8, 1], [4, 1, 2, 3, 22, 2, 4, 11, 1], [3, 12, 4, 13, 6, 2, 4, 4, 7, 1], [3, 13, 4, 14, 6, 2, 5, 5, 7, 1], [4, 16, 19, 4, 1, 7, 6, 8, 1], [7, 1, 5, 3, 15, 8, 12, 10], [6, 1, 9, 15, 7, 13, 9], [6, 1, 10, 16, 7, 5, 8, 9], [6, 1, 10, 15, 3, 2, 4, 9, 6, 1], [6, 1, 4, 3, 17, 3, 2, 4, 8, 3, 1], [1, 5, 1, 5, 3, 18, 2, 7, 9, 3, 1], [1, 5, 2, 5, 3, 20, 2, 4, 6, 3, 4, 1], [1, 7, 7, 3, 20, 2, 4, 5, 3, 3, 1], [1, 4, 1, 6, 25, 4, 2, 6, 5, 2], [1, 3, 1, 3, 3, 21, 3, 2, 6, 3, 3], [1, 3, 1, 4, 21, 3, 2, 5, 3, 2, 2], [1, 4, 1, 4, 22, 4, 1, 2, 3, 3, 2, 2], [1, 7, 4, 26, 2, 1, 3, 3, 3, 3], [1, 10, 4, 25, 4, 2, 16], [1, 6, 3, 4, 60], [10, 22, 4, 59], [1, 4, 3, 4, 21, 6, 16], [1, 4, 3, 5, 17, 8, 5, 9], [1, 9, 5, 12, 15, 7], [1, 9, 5, 11, 14, 9], [1, 8, 4, 11, 17, 7], [1, 7, 4, 1, 9, 21, 5], [1, 7, 7, 1, 8, 4, 19, 5], [15, 9, 7, 5, 24]]
colhints = [[90], [58, 10, 1, 1], [12, 10, 20, 15, 1, 1], [6, 5, 6, 13, 4, 13, 1, 1], [6, 5, 7, 11, 2, 13, 1, 1], [6, 6, 7, 9, 1, 17, 1], [6, 5, 6, 10, 1, 4, 3, 7, 1], [5, 5, 7, 8, 1, 3, 1, 8, 1], [6, 4, 7, 7, 2, 3, 14, 1], [5, 6, 6, 3, 2, 3, 12], [5, 6, 6, 2, 3, 3, 6], [7, 5, 4, 2, 3, 1, 6], [10, 1, 3, 1, 2, 7, 3, 1, 6], [11, 3, 2, 6, 9, 11], [13, 4, 1, 4, 1, 10, 11], [15, 3, 5, 1, 3, 3, 9], [17, 1, 3, 4, 9, 3, 1], [20, 2, 18, 4, 1], [22, 2, 2, 4, 3, 2, 3, 5, 1], [23, 2, 2, 2, 2, 2, 5, 1], [24, 2, 5, 2, 10, 1], [25, 2, 6, 2, 8, 1], [16, 10, 1, 2, 3, 7, 5, 1], [4, 10, 10, 2, 2, 15, 3, 1], [4, 3, 4, 9, 2, 1, 15, 3, 1], [3, 2, 3, 10, 2, 2, 18, 1], [4, 4, 1, 10, 2, 2, 7, 7, 1], [3, 3, 2, 3, 8, 2, 2, 8, 7, 1], [4, 1, 3, 4, 7, 2, 2, 10, 6, 1], [4, 2, 3, 3, 6, 3, 2, 10, 6, 1], [4, 1, 4, 4, 7, 4, 2, 10, 4, 1], [4, 1, 3, 3, 5, 7, 2, 10, 4, 1], [4, 1, 3, 3, 13, 1, 9, 3, 4, 1], [3, 1, 4, 3, 9, 2, 15, 3, 1], [3, 1, 4, 3, 13, 2, 14, 4, 1], [3, 1, 2, 2, 19, 2, 15, 4], [3, 1, 2, 2, 19, 1, 16, 4], [4, 1, 1, 20, 3, 17, 4], [1, 4, 1, 1, 11, 6, 2, 1, 18, 4, 1], [1, 1, 3, 2, 6, 1, 2, 6, 2, 3, 2, 20, 4, 2], [1, 1, 2, 2, 1, 4, 1, 9, 2, 2, 2, 22, 4, 3], [1, 2, 3, 2, 5, 8, 4, 2, 2, 2, 5, 18, 7], [1, 2, 7, 4, 7, 5, 1, 2, 1, 9, 25], [1, 12, 4, 4, 2, 7, 2, 2, 2, 8, 17, 6], [1, 13, 6, 1, 2, 5, 2, 1, 2, 5, 16, 2, 2], [1, 13, 2, 1, 2, 4, 5, 3, 6, 16, 2], [16, 4, 2, 2, 3, 6, 3, 6, 16, 1, 1], [16, 3, 2, 2, 2, 6, 3, 6, 15, 1], [17, 2, 2, 2, 7, 1, 3, 2, 3, 14, 1], [16, 2, 2, 1, 6, 2, 3, 2, 2, 17], [2, 8, 2, 2, 1, 2, 2, 5, 6, 2, 2, 18], [1, 10, 4, 1, 1, 4, 6, 3, 2, 18], [11, 1, 3, 2, 6, 10, 4, 17], [11, 1, 3, 2, 5, 5, 4, 16], [10, 3, 1, 3, 1, 4, 3, 5, 16], [1, 7, 2, 3, 1, 1, 5, 2, 5, 15], [8, 3, 1, 2, 4, 2, 3, 2, 14], [1, 6, 6, 1, 2, 5, 5, 2, 12], [7, 6, 4, 6, 4, 2, 10], [7, 3, 1, 3, 8, 2, 2, 8], [12, 1, 14, 2, 11], [10, 1, 7, 6, 2, 10], [7, 1, 7, 6, 12], [2, 1, 2, 5, 6, 3, 5], [3, 5, 5, 4, 2], [3, 4, 5, 2, 2, 2], [2, 4, 4, 3, 2, 2], [2, 4, 4, 3, 2, 3], [2, 4, 3, 3, 6], [1, 4, 4, 3, 3, 4, 1], [1, 4, 5, 2, 3, 4, 2], [1, 5, 5, 2, 2, 4, 10], [1, 2, 2, 6, 2, 3, 10], [1, 1, 2, 7, 3, 3, 10], [1, 2, 2, 4, 4, 2, 6, 8], [4, 2, 3, 2, 2, 6, 9], [3, 2, 4, 2, 2, 10, 2, 7], [2, 2, 4, 2, 2, 12, 2, 7], [2, 2, 3, 4, 3, 10, 2, 7], [2, 2, 4, 4, 2, 13, 2, 6], [2, 2, 4, 3, 2, 17, 6], [1, 1, 2, 4, 3, 3, 3, 9, 3, 6], [2, 2, 2, 4, 5, 2, 2, 3, 4, 2, 6], [2, 2, 2, 4, 2, 2, 2, 3, 3, 4, 2, 7], [2, 1, 2, 4, 3, 2, 3, 4, 3, 13], [2, 1, 2, 4, 3, 4, 3, 3, 7, 4], [2, 1, 2, 4, 2, 3, 3, 3, 5, 4], [3, 6, 4, 3, 2, 3, 3, 5, 4], [2, 2, 3, 4, 3, 2, 3, 4, 4, 3], [3, 2, 2, 4, 3, 2, 5, 3, 7, 3], [3, 2, 2, 3, 4, 2, 8, 2, 6, 3], [2, 3, 6, 6, 2, 10, 3, 6, 1, 3], [3, 1, 5, 18, 3, 13, 5, 1, 2], [3, 2, 1, 1, 1, 16, 3, 13, 9, 2], [5, 5, 1, 5, 11, 3, 3, 9, 10, 1], [7, 3, 10, 5, 3, 3, 7, 14], [15, 4, 2, 3, 6, 2, 11], [12, 4, 7, 4, 1, 12], [8, 4, 7, 3, 16], [75, 14]]
solved = solve(field, rowhints, colhints)
for row in solved:
    print(row)

# areas = np.array(areas)
# times = np.array(times)
