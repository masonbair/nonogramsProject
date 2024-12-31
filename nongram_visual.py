import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import numpy as np
from matplotlib.animation import FuncAnimation

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


# Nonogram solver
# A nonogram is a triple consisting of an m*n bit matrix,
# a row hint array of length m
# and a column hint array of length n.
# The bit matrix cells can take values 0 (white), 1 (black) and -1 (white, marked).
# Marked cells will be handled differently by the algorithm.
# In the end, if the algorithm succeeds, all cells will be black or marked.
# If it does not, our job is to solve the rest manually and make up more heuristics based on that.
def solve(field, rowhints, colhints):
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
    
    # If we're starting or the field was changed 
    while changed:
        print(field)
        changed = False
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
            # Mark empty squares of complete rows
            if 0 in field[i] and check_solved(field[i], hint):
                # if not check_gaps(field[i], hint):
                #     raise ValueError(f"Row {i} does not match hints: {hint}")
                field[i] = [-1 if square < 1 else 1 for square in field[i]]
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
                                field[i][j - gap_index] = -1
                            changed = True
                        gap_size = 0
                # If there is a gap in the end, only check with the last hint
                if gap_size > 0 and gap_size < hint[-1]:
                    for gap_index in range(1, gap_size+1):
                        field[i][-gap_index] = -1
                    changed = True
            # If there are as many unmarked squares as the hints have left, colour them all
            if 0 in field[i] and sum(hint) == sum([1 if square > -1 else 0 for square in field[i]]):
                field[i] = [1 if square > -1 else -1 for square in field[i]]
                changed = True
        for i, hint in enumerate(colhints):
            # If the hint is larger than half of the squares it can occupy, fill the middle ones
            for j, nr in enumerate(hint):
                bonus = 0
                # May prove useful, not sure atm
                # if j > 0:
                #     bonus = 1

                other_lengths = sum(hint) + len(hint)-1 - nr
                leave_empty = HEIGHT - other_lengths - nr # This is greater than 0 because the 0-case is handled by the simple heuristic.
                if nr > leave_empty:
                    hints_before = sum(hint[:j]) + len(hint[:j])
                    hints_after = sum(hint[j:]) + len(hint[j:])-1 - nr
                    for fill in range(hints_before+leave_empty+bonus, HEIGHT-hints_after-leave_empty):
                        if field[fill][i] == 0:
                            print("It does something")
                            changed = True
                            field[fill][i] = 1
            # Mark empty squares of complete columns
            column = [row[i] for row in field]
            if 0 in column and check_solved(column, hint):
                if(len(hint) >= 2):
                    print('hello')
                else:
                    for row in field:
                        if row[i] < 1:
                            print("marked")
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
                                field[j - gap_index][i] = -1
                            changed = True
                        gap_size = 0
                # If there is a gap in the end, only check with the last hint
                if gap_size > 0 and gap_size < hint[-1]:
                    for gap_index in range(1, gap_size+1):
                        field[-gap_index][i] = -1
                    changed = True
            # If there are as many unmarked squares as the hints have left, colour them all
            if 0 in column and sum(hint) == sum([1 if square > -1 else 0 for square in column]):
                for row in field:
                    if row[i] > -1:
                        row[i] = 1
                changed = True
        rows_needing_changed = []
        cols_needing_changed = []
        print("Checking row correctness")
        for i, row in enumerate(field):
            if not check_line_gaps(row, rowhints[i], "row"):
                # changed = True
                if i not in rows_needing_changed:
                    rows_needing_changed.append(i)
        print("checking column correctness")
        for i in range(len(field[0])):
            column = [row[i] for row in field]
            if not check_line_gaps(column, colhints[i], "column"):
                # changed = True
                if i not in cols_needing_changed:
                    cols_needing_changed.append(i)
        print(f"Bad rows found: {rows_needing_changed}")
        print(f"Bad columns found: {cols_needing_changed}")

        field, changed = try_fix_combinations(field, rows_needing_changed, cols_needing_changed, rowhints, colhints)

    # Once the nonogram is solved or the algorithm is out of ideas, it stops
    return field



##SHEEESHHHHHH hahahaha this function is crazy
# The main purpose of the function that I created is to detect if the hint order matches up with the place it appear
# for example a hint of [1,2] must be in the order of 1011 or 01011 NOT 1101 or 01101 etc...
# This function is a bit more complex then I would have liked it too be, but it seems to work?
def check_line_gaps(line, hints, line_type="row"):
    print(f"{line_type} working with: {line}")
    line_matches_hints = False  # Just some pointless boolean tbh...

    current_hint = 0  # This keeps track of the current hint number in the hint line. EX [1,1] could have a current_hint of 0 or 1

    line_index = 0  # This is to keep track of the last 1 checked. This helps to tell the program where to start next when checking the next hint number
    # For example [0,1,1,0,1] when I run the program line_index starts at 0, then becomes 2, then 4 and the program stops.

    # This while loop is technically infinite, but is necessary to make sure all hints are checked.
    while not line_matches_hints:
        count_ones = 0
        # All the print statements are for debugging, I will keep them just incase!
        print(f"index: {line_index}")
        print(f"{line_type} len: {len(line)}")
        print(f"current_hint_number: {current_hint}")
        print(f"hint length: {hints}")
        print(f"{line_type}: {line}")

        #Checks if the current_hint is sufficient. This means that if the current_hint is bigger then the hints length, we need to make sure the line is good
        if current_hint >= len(hints):
            for index, expel in enumerate(line[line_index:], start=line_index):
                if expel == 1:
                    print("The current hint exceeds the current values: ie hint = 2 and the nonogram has 1,0")
                    return False
            return True

        # more checking if the line is good -----------------------------
        if line_index >= len(line) and current_hint < len(hints):
            print("The hint and line index do not match")
            return False
        if line_index >= len(line) and current_hint >= len(hints):
            line_matches_hints = True
            print("Hints and index match!")
            return True
        # --------------------------------------------------------------

        # Using a while loop to keep track of the value more 
        while line_index < len(line):
            line_n = line[line_index]
            print(line_index)
            print(line_n)
            # This checks for the first iteration. If 1 is not the first available value. This then moves the index to the first found 1 value
            if line_index == 0 and line_n != 1:
                times_a_zero_or_neg1_appears = 0
                while line[times_a_zero_or_neg1_appears] == 0 or line[times_a_zero_or_neg1_appears] == -1:
                    print("Zero found")
                    times_a_zero_or_neg1_appears += 1
                    if times_a_zero_or_neg1_appears >= len(line)-1:
                        return len(hints) == 0
                line_index += times_a_zero_or_neg1_appears 
                print(f"Current index after counting initial zeros: {line_index}")
                line_n = line[line_index]  # Setting current value to a one!
                print(f"{line_type}_n value: {line_n}")
            # Easy. if the line number is 1 then count it
            if line_n == 1:
                print("counting")
                count_ones += 1
            else:  # Check for the number of 0's after the last 1. This is to account for any gap that might potentially
                # exist
                print("Zero time")
                # This for loop is for when there are large gaps of 0's or -1's in the line. This helps find the next 1 for the next iteration
                for j, count_zeros in enumerate(line[line_index:], start=line_index):
                    print(f"j value: {j}, count_zero value: {count_zeros}")
                    if count_zeros == 1:
                        line_index = j 
                        break
                print(f"{line_index} : Index")
                print(f"count ones: {count_ones}")
                break
            line_index += 1
        # This is another check statement. It checks to make sure the number of counted ones is the same as the hint says it is!
        if count_ones != hints[current_hint]:
            print(f"The counted ones dont match the hint: counted_ones: {count_ones}, hint ones:{hints[current_hint]} ")
            return False
        current_hint += 1

def try_fix_combinations(field, rows_needing_changed, cols_needing_changed, rowhints, colhints):
    if len(rows_needing_changed) < 1 or len(cols_needing_changed) < 1:
        return field, False
        
    # Create a copy of the field to work with
    working_field = [row[:] for row in field]
    
    # Try each combination of row and column
    for row_idx in rows_needing_changed:
        for col_idx in cols_needing_changed:
            original_value = working_field[row_idx][col_idx]

            working_field[row_idx][col_idx] = 1
            
            #Checking the row and column to see if it works
            row_valid = check_line_gaps(working_field[row_idx], rowhints[row_idx], "row")
            column = [row[col_idx] for row in working_field]
            col_valid = check_line_gaps(column, colhints[col_idx], "column")
            
            #IF both are true then it is a valid change
            if row_valid and col_valid:
                field[row_idx][col_idx] = 1
                return field, True
                
            #Some times a value needs to be set backwards
            working_field[row_idx][col_idx] = -1
            
            row_valid = check_line_gaps(working_field[row_idx], rowhints[row_idx], "row")
            column = [row[col_idx] for row in working_field]
            col_valid = check_line_gaps(column, colhints[col_idx], "column")
            
            # Checking for valid solution
            if row_valid and col_valid:
                field[row_idx][col_idx] = -1
                return field, True
                
            # If neither works restore the original value and continue
            working_field[row_idx][col_idx] = original_value
    
    return field, False



def check_solved(row, rowhints):
# Check if a row has its hints exhausted.
# Also works with columns.
    hints = rowhints[:] # So we can modify them while counting.
    print(hints)
    for i in range(len(row)):
        # current_hint_number = hints[0]

        if row[i] == 1:
            # A sequence of black squares can't be longer than the current hint.
            if hints[0] < 1:
                print("Seqeunces check")
                return False
            else:
                hints[0] -= 1
        else:
            # An exhausted hint is removed.
            # If it is not exhausted by the time a gap is reached, there is more work to do.
            if len(hints) == 0:
                return False
            if hints[0] == 0:
                hints.pop(0)
            elif i > 0 and row[i-1] == 1:
                return False
    return sum(hints) == 0

field = [[0 for _ in range(7)] for _ in range(3)]
rowhints = [[1, 1], [7], [1, 1]]
colhints = [[1], [3], [1], [1], [1], [3], [1]]
solved = solve(field, rowhints, colhints)
for row in solved:
    print(row)

print()
field = [[0 for _ in range(7)] for _ in range(3)]
rowhints = [[5], [1, 4], [4, 1]]
colhints = [[2], [1, 1], [1, 1], [3], [2], [3], [1]]
solved = solve(field, rowhints, colhints)
for row in solved:
    print(row)

print()
field = [[0 for _ in range(4)] for _ in range(5)]
rowhints = [[1,1],[2,1],[1], [1,2], [2]]
colhints = [[2,1], [1], [1,3], [1,2]]
solved = solve(field, rowhints, colhints)
for row in solved:
    print(row)

#________________GRID SET UP_____________________________________
cmap = colors.ListedColormap(["grey", "white", "black"])
bounds = [-1.5, -0.5, 0.5, 1.5]
norm = colors.BoundaryNorm(bounds, cmap.N)

grid = np.full((len(rowhints), len(colhints)), -1)

fig, ax = plt.subplots(figsize=(8, 8))
gs = fig.add_gridspec(12, 12, wspace=0.2, hspace=0.2)
ax_grid = fig.add_subplot(gs[2:, 2:])

ax_grid.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
ax_grid.set_xticks(np.arange(-.5, len(colhints), 1), minor=True)
ax_grid.set_yticks(np.arange(-.5, len(rowhints), 1), minor=True)
ax_grid.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
ax_grid.set_xticks([])
ax_grid.set_yticks([])
add_clues()
ax.axis("off")
###################################################################

anim = FuncAnimation(fig, update, frames=10, repeat=False)
plt.show()
