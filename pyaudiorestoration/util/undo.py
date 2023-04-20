import logging

from PyQt5.QtWidgets import QUndoCommand, QUndoStack


class UndoStack(QUndoStack):

	def __init__(self, main_widget, canvas):
		super(UndoStack, self).__init__(main_widget)
		self.canvas = canvas
		self.indexChanged.connect(self.update)

	def push(self, cmd: QUndoCommand) -> None:
		if cmd.traces:
			super(UndoStack, self).push(cmd)

	def undo(self) -> None:
		if self.canUndo():
			super(UndoStack, self).undo()

	def redo(self) -> None:
		if self.canRedo():
			super(UndoStack, self).redo()

	def update(self):
		self.canvas.update_lines()
		# self.index() is the next that will be pushed or None if currently the end
		ind = self.index() - 1
		if -1 < ind:
			# select the traces used by the last active command in the stack
			self.canvas.deselect_all()
			command = self.command(ind)
			for trace in command.traces:
				trace.select()


class Action(QUndoCommand):

	def __init__(self, traces, *args, **kwargs):
		super(Action, self).__init__()
		self.traces = traces
		self.args = args
		self.kwargs = kwargs
		action_type = type(self).__name__.replace('Action', '')
		self.setText(f"{action_type}[{len(self.traces)}]")


class AddAction(Action):
	def undo(self):
		for trace in self.traces:
			trace.remove()

	def redo(self):
		for trace in self.traces:
			trace.initialize()


class DeleteAction(Action):
	def undo(self):
		for trace in self.traces:
			trace.initialize()

	def redo(self):
		for trace in self.traces:
			trace.remove()


class MergeAction(Action):
	def undo(self):
		for trace in self.traces:
			trace.remove()
		for trace in self.args[0]:
			trace.initialize()

	def redo(self):
		for trace in self.traces:
			trace.initialize()
		for trace in self.args[0]:
			trace.remove()


class MoveAction(Action):
	def undo(self):
		for trace in self.traces:
			trace.set_offset(*reversed(self.args))

	def redo(self):
		for trace in self.traces:
			trace.set_offset(*self.args)


class DeltaAction(Action):
	def undo(self):
		for trace, delta in zip(self.traces, self.args[0]):
			trace.set_offset(-delta)

	def redo(self):
		for trace, delta in zip(self.traces, self.args[0]):
			trace.set_offset(delta)
