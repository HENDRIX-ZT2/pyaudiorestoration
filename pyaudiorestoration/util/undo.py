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
		self.canvas.master_speed.update()
		self.canvas.master_reg_speed.update()


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
		# logging.info(f"Adding {len(self.traces)} traces")
		for trace in self.traces:
			trace.initialize()


class DeleteAction(Action):
	def undo(self):
		for trace in self.traces:
			trace.initialize()

	def redo(self):
		for trace in self.traces:
			trace.remove()


class MoveAction(Action):
	def undo(self):
		for trace in self.traces:
			trace.set_offset(*reversed(self.args))

	def redo(self):
		for trace in self.traces:
			trace.set_offset(*self.args)
