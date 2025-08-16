#!/usr/bin/env python3

from typing import Optional
from isaacgym import gymapi
import torch as th
import numpy as np

from ham.env.env.iface import EnvIface
from ham.env.env.wrap.base import WrapperEnv
from ham.util.torch_util import dcn
# import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from ham.util.vis.mplots import SimpleStreamingPlot

class ForceStreamingPlot:
    def __init__(self, ax: Optional[plt.Axes] = None,
                 lims=None):
    
        plt.ion()
        self.fig, self.axes = plt.subplot_mosaic(
            [['panda_hand','panda_leftfinger','panda_rightfinger']]
        )
        lns = {}
        for k, axis in self.axes.items():
            lns[k], = axis.plot([], [])
            axis.set_title(k)
        self.lns = lns
        self.lims = lims

    def update(self, *args, **kwds):
        y = kwds.pop("y")
        x = kwds.pop("x")
        for i ,(k, axis) in enumerate(self.axes.items()):
            self.lns[k].set_data(y[i], x[i])
            xy = self.lns[k].get_xydata()
            xx = xy[..., 0]
            yy = xy[..., 1]
            xmin = np.quantile(xx, 0.05)
            xmax = np.quantile(xx, 0.95)
            ymin = np.quantile(yy, 0.05)
            ymax = np.quantile(yy, 0.95)
            axis.set_xlim(xmin, xmax)

            if self.lims is not None:
                axis.set_ylim(self.lims[0], self.lims[1])
            else:
                axis.set_ylim(ymin, ymax)

            self.lns[k].figure.canvas.flush_events()
        plt.pause(0.001)

class DrawForceSensor(WrapperEnv):


    def __init__(self, env: EnvIface):
        super().__init__(env)
        self.__plot = ForceStreamingPlot()
        self.__forces = []
        # actor_id = self.gym.find_actor_index(self.envs[0],
        #                                             'robot',
        #                                             gymapi.IndexDomain.DOMAIN_SIM)
        body_ids={}
        actor_handle = self.gym.find_actor_handle(self.envs[0], 'robot')
        for k in ['panda_hand', 'panda_leftfinger', 'panda_rightfinger']:
            body_ids[k]=self.gym.find_actor_rigid_body_index(
                self.envs[0], actor_handle, k,
                gymapi.IndexDomain.DOMAIN_SIM)
        self.__body_ids = body_ids

    def step(self, actions: th.Tensor):
        # == step normally...
        out = super().step(actions)
        force_sensor = self.tensors['net_contact']
    
        
        force = []
        for v in self.__body_ids.values():
            print(v)
            force.append(th.norm(force_sensor[v], dim=-1))
        self.__forces.append(dcn(th.stack(force)))
        # self.__forces.append(dcn(th.norm(force_sensor[0,:3],
        #                                  dim=-1)))
        
        self.__forces = self.__forces[-100:]
        if True:
            f = np.stack(self.__forces, -1)
            if len(self.__forces) > 0:
                l = np.arange(f.shape[-1])
                self.__plot.update(y=np.repeat(l[None], 3, 0), x=f)
        else:
            self.__plot.update(np.arange(len(self.__forces)),
                               self.__forces)
        
        return out
