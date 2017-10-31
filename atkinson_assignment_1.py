# (Problem 1)
# Flatten function, that takes a list which could contain any number of nested lists
# and returns a single list on the same level.

def flatten(lst):
	if lst == []:
		return []
	rest = lst[1:]
	if type(lst[0]) != type([]):
		return [lst[0]] + flatten(rest)
	return flatten(lst[0]) + flatten(rest)


# (Problem 2)
# Powerset function, that takes a list and creates all possible subsets of the list.

def powerset(lst):
	if lst == []:
		return [[]]
	else:
		rest = powerset(lst[1:])
		return map(lambda x: [lst[0]] + x, rest) + rest
		
# (Problem 3)
# Permutation function, that produces all permutations of a list.
		
def all_perms(lst):
	if len(lst)== 1:
		return [lst]
	else:
		rest = all_perms(lst[1:])
		perms = []
		for sp in rest:
			for i in range(len(sp)+1):
				np = list(sp)
				np.insert(i, lst[0])
				perms.append(np)
		return perms
		
# (Problem 4)
# Number spiral function, that prints a clockwise spiral of numbers based off the input.

def num_spiral(n, end_corner):
		# Direction vectors, movement represented with 0s and 1s.
		# These specific vectors are meant to signify which corner (n^2)-1 will result in.
		top_left = [[0,1],[1,0 ],[0,-1],[-1,0]]
		top_right =[[0,-1],[-1,0],[0,1],[1,0]]
		bottom_left =[[1,0],[0,1],[-1,0],[0,-1]]
		bottom_right = [[-1,0],[0,-1],[1,0],[0,1]]
		matrix = matrix[][] 
		curr_num = (n^2) - 1
		curr_coord = [curr_row, curr_col]
		
		# Using a while loop to satisy one of four posibilities of direction.
		while curr_num >= 0:
			row, col = curr_coord
			matrix[row][col] = curr_num
			curr_num = curr_num - 1 #Used to decrease the number incrimentally as it winds down to 0 in the center.
			if time_to_change_direction(curr_coord, curr_direction, matrix):  
				curr_direction = (curr_direction + 1) % 4  # This checks whether there is a corner and turns if there is.
				dy, dx = direction_vectors[curr_direction]  # Get the next row/column directional changes
				curr_coord = (row + dy, col + dx)  # Move the current coordinate in the appropriate direction
				
		# I tried my best to conceputalize the code on paper but it wasn't making any logical sense,
		# so I am submitting this, I genuinely believe I am burnt out and won't be able to solve the rest,
		# I am sorry if that is dissapointing.  I deeply appreciate the extension you allowed me to have sir.
			
			
