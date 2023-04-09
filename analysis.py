#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:08:23 2022

@author: tjards
"""
import matplotlib.pyplot as plt

#%%

fig, ax = plt.subplots()
ax.plot(t_all[4::],metrics_order_all[4::,1])

ax.set(xlabel='time (s)', ylabel='mean distance from target [m]',
       title='Distance from Target')
ax.grid()

#fig.savefig("test.png")
plt.show()



#%% Produce plots
# --------------

start = 10

#%% Convergence to target 
#-------------------------
fig, ax = plt.subplots()
ax.plot(t_all[start::],metrics_order_all[start::,1],'-b')
ax.plot(t_all[start::],metrics_order_all[start::,5],':b')
ax.plot(t_all[start::],metrics_order_all[start::,6],':b')
ax.fill_between(t_all[start::], metrics_order_all[start::,5], metrics_order_all[start::,6], color = 'blue', alpha = 0.1)
#note: can include region to note shade using "where = Y2 < Y1
ax.set(xlabel='Time [s]', ylabel='Mean Distance to Target [m]',
       title='Convergence to Target')
#ax.plot([70, 70], [100, 250], '--b', lw=1)
#ax.hlines(y=5, xmin=Ti, xmax=Tf, linewidth=1, color='r', linestyle='--')
ax.set_xlim([0, Tf])
ax.grid()

#fig.savefig("test.png")
plt.show()


#%% Energy
# ------------
fig, ax = plt.subplots()

# set forst axis
ax.plot(t_all[start::],metrics_order_all[start::,7],'-g')
#ax.plot(t_all[4::],metrics_order_all[4::,7]+metrics_order_all[4::,8],':g')
#ax.plot(t_all[4::],metrics_order_all[4::,7]-metrics_order_all[4::,8],':g')
ax.fill_between(t_all[start::], metrics_order_all[start::,7], color = 'green', alpha = 0.1)

#note: can include region to note shade using "where = Y2 < Y1
ax.set(xlabel='Time [s]', title='Energy Consumption')
ax.set_ylabel('Total Acceleration [m^2]', color = 'g')
ax.tick_params(axis='y',colors ='green')
ax.set_xlim([0, Tf])
ax.set_ylim([0, 10])
#ax.plot([70, 70], [100, 250], '--b', lw=1)
#ax.hlines(y=5, xmin=Ti, xmax=Tf, linewidth=1, color='r', linestyle='--')
total_e = np.sqrt(np.sum(cmds_all**2))
# ax.text(3, 2, 'Total Energy: ' + str(round(total_e,1)), style='italic',
#         bbox={'facecolor': 'green', 'alpha': 0.1, 'pad': 1})


# set second axis
ax2 = ax.twinx()
#ax2.set_xlim([0, Tf])
#ax2.set_ylim([0, 1])
ax2.plot(t_all[start::],1-metrics_order_all[start::,0], color='tab:blue', linestyle = '--')
#ax2.fill_between(t_all[4::], 1-metrics_order_all[4::,0], color = 'tab:blue', alpha = 0.1)
ax2.set(title='Energy Consumption')
ax2.set_ylabel('Disorder of the Swarm', color='tab:blue')
#ax2.invert_yaxis()
ax2.tick_params(axis='y',colors ='tab:blue')
ax2.text(Tf-Tf*0.3, 0.1, 'Total Energy: ' + str(round(total_e,1)), style='italic',
        bbox={'facecolor': 'green', 'alpha': 0.1, 'pad': 1})

ax.grid()
#fig.savefig("test.png")
plt.show()

#%% Spacing
# ---------

fig, ax = plt.subplots()

# set forst axis
ax.plot(t_all[start::],metrics_order_all[start::,9],'-g')
ax.plot(t_all[start::],metrics_order_all[start::,11],'--g')
ax.fill_between(t_all[start::], metrics_order_all[start::,9], metrics_order_all[start::,11], color = 'green', alpha = 0.1)

#note: can include region to note shade using "where = Y2 < Y1
ax.set(xlabel='Time [s]', title='Spacing between Agents [m]')
ax.set_ylabel('Mean Distance [m]', color = 'g')
ax.tick_params(axis='y',colors ='green')
#ax.set_xlim([0, Tf])
#ax.set_ylim([0, 40])
total_e = np.sqrt(np.sum(cmds_all**2))

# set second axis
ax2 = ax.twinx()
#ax2.set_xlim([0, Tf])
#ax2.set_ylim([0, 100])
ax2.plot(t_all[start::],metrics_order_all[start::,10], color='tab:blue', linestyle = '-')
ax2.set_ylabel('Number of Connections', color='tab:blue')
ax2.tick_params(axis='y',colors ='tab:blue')
#ax2.invert_yaxis()

ax.legend(['Within Range', 'Oustide Range'], loc = 'upper left')
ax.grid()
#fig.savefig("test.png")
plt.show()

