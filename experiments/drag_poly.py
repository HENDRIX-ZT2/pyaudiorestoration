import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

#------------------------------------------------



#================================================
fig, ax = plt.subplots()

ax.set_title("Double click left button to create draggable point\nDouble click right to remove a point", loc="left")
ax.set_xlim(0, 4000)
ax.set_ylim(0, 3000)
ax.set_aspect('equal')
line_object = ax.plot([], [], alpha=0.5, c='r', lw=2, picker=True)
line_object[0].set_pickradius(0.0)

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('pick_event', on_pick)
fig.canvas.mpl_connect('motion_notify_event', on_motion)

plt.grid(True)
plt.show()
