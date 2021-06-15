if __name__ == "__main__":
	print("modules.Node called")

class Node:
	result = False
	
	def __init__(self):
		self.result = True
	
	def process(self):
		# do something
		return self.result
