import logging


class UndoStack:

	def __init__(self, canvas):
		# if current is a valid index, actions[current] is the last active action
		self.actions = []
		self.current = -1
		self.canvas = canvas

	@property
	def action_name(self):
		action_type = type(self.action).__name__.replace('Action', '')
		return f"{action_type}[{len(self.action.traces)}]"

	@property
	def action(self):
		return self.actions[self.current]

	def set_state(self, ind):
		logging.info(f"Going to state {ind}")

	def add(self, action):
		self.discard_future()
		self.actions.append(action)
		self.current = len(self.actions) - 1
		self.action.do()
		self.update()
		self.canvas.parent.props.stack_widget.add(self.action_name)

	def undo(self):
		if self.current > -1:
			logging.info(f"Undoing action {self.current}: {self.action_name}")
			self.action.undo()
			self.current -= 1
			self.update()
		else:
			logging.info(f"Nothing to undo")

	def redo(self):
		if self.current + 1 < len(self.actions):
			self.current += 1
			logging.info(f"Redoing action {self.current}: {self.action_name}")
			self.action.do()
			self.update()
		else:
			logging.info(f"Nothing to redo")

	def update(self):
		self.canvas.parent.props.stack_widget.select_index(self.current)
		self.canvas.master_speed.update()
		self.canvas.master_reg_speed.update()

	def discard_future(self):
		"""Ensure that no actions from the future are purged from the stack"""
		if self.current > -1:
			# is the action the last action
			if self.action != self.actions[-1]:
				logging.info(f"Discarding future actions")
				# then delete all actions up to the last action
				self.actions[:] = self.actions[:self.current+1]


class Action:

	def __init__(self, traces, *args):
		self.traces = traces
		self.args = args

	def undo(self):
		pass

	def do(self):
		pass


class AddAction(Action):
	def undo(self):
		for trace in self.traces:
			trace.remove()

	def do(self):
		for trace in self.traces:
			trace.initialize()


class DeleteAction(Action):
	def undo(self):
		for trace in self.traces:
			trace.initialize()

	def do(self):
		for trace in self.traces:
			trace.remove()


class MoveAction(Action):
	def undo(self):
		for trace in self.traces:
			trace.set_offset(*reversed(self.args))

	def do(self):
		for trace in self.traces:
			trace.set_offset(*self.args)