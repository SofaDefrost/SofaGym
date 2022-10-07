# -*- coding: utf-8 -*-
"""Interface to display the simulation rendering.
"""

__authors__ = ("PSC", "emenager")
__contact__ = ("pierre.schegg@robocath.com", "etienne.menager@ens-rennes.fr")
__version__ = "1.0.0"
__copyright__ = "(c) 2020, Robocath, Inria"
__date__ = "Oct 7 2020"

import numpy as np
import pygame
from PIL import Image
import queue
import importlib
import sys

import glfw
import Sofa
import Sofa.SofaGL
from OpenGL.GL import *
from OpenGL.GLU import *

import imageio
import datetime

from sofagym.env.common.rpc_server import get_position
from sofagym.env.common.simulate import init_simulation, step_simulation


class Viewer:
    """Class to manage the display of the current state of the gym environment.

    Methods:
    -------
        __init__: Initialization of all arguments.
        render: Display image in a window.
        get_image: Take the current image on the window.
        set_agent_display: Set a display callback provided by an agent.
        close: Close the viewer.
        save_image: save an image.

    Arguments:
    ---------
        env: <class AbstractEnv>
            The environment to be displayed.
        surface_size: tuple of int
            Size of the display.
        save_path: string
            Path to save images of the scene.
        screen_size: tuple of int
            Resize window.
        screen:
            Pygame window.
        agent_display:
            Callback provided by the agent to display on surfaces.
        sim_surface:
            Surface for the simulation.
        agent_surface:
            Surface for the agent display.
        frame:
            The number of the image to be saved.

    """

    def __init__(self, env, surface_size, zFar=100, save_path=None, create_video=None, fps=10):
        """
        Classic initialization of a class in python.

        Parameters:
        ----------
        env: <class AbstractEnv>
            The environment to be displayed.
        surface_size: tuple of int
            Size of the display.
        save_path: string or None, default = None
            Path to save images of the scene.

        Returns:
        ---------
            None.

        """
        self.env = env
        self.save_path = save_path
        self.create_video = create_video

        self.surface_size = surface_size
        self.screen_size = (surface_size[0], int(1.5 * surface_size[1]))

        pygame.display.init()
        pygame.font.init()

        self.screen = pygame.display.set_mode(self.screen_size, pygame.DOUBLEBUF)
        self.sim_surface = pygame.Surface((surface_size[0], surface_size[1]))
        self.agent_surface = pygame.Surface((surface_size[0], surface_size[1]//2))

        self.agent_display = None
        self.frame = 0

        if not glfw.init():
            sys.exit(1)

        glfw.window_hint(glfw.VISIBLE, False)
        self.window = glfw.create_window(self.surface_size[0], self.surface_size[1], "hidden window", None, None)

        if not self.window:
            print("ERROR glfw is dead")
            glfw.terminate()
            sys.exit(2)

        self.root = init_simulation(self.env.config, mode = 'visu')
        scene = self.env.config['scene']
        self._setPos = importlib.import_module("sofagym.env."+scene+"."+scene+"Toolbox").setPos

        self.zFar = self.root.camera.zFar.value + zFar

        if self.create_video is not None:
            self.writer = imageio.get_writer(self.create_video + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') +
                                             ".mp4", format='mp4', mode='I', fps=fps)

    def render(self, pos=None):
        """See the current state of the environment.

        Take a picture from the server and display it on the window.
        Display a black window if there are no images available.

        Parameters:
        ----------
            None.

        Returns:
        -------
            The picture on the window.
        """

        # Recovering an image and handling error cases
        try:
            if pos is None:
                pos = get_position(self.env.past_actions)['position']
            self.frame += 1
            if pos == []:
                image = np.zeros((self.surface_size[0], self.surface_size[1], 3))
            else:
                num_im = 0
                for p in pos:
                    self._setPos(self.root, p)
                    Sofa.Simulation.animate(self.root, 0.0001)

                    glfw.make_context_current(self.window)
                    glViewport(0, 0, self.surface_size[0], self.surface_size[1])

                    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
                    glEnable(GL_LIGHTING)
                    glEnable(GL_DEPTH_TEST)

                    if self.root:
                        Sofa.SofaGL.glewInit()
                        Sofa.Simulation.initVisual(self.root)
                        glMatrixMode(GL_PROJECTION)
                        glLoadIdentity()
                        gluPerspective(45, (self.surface_size[0] / self.surface_size[1]), 0.1, self.zFar)

                        glMatrixMode(GL_MODELVIEW)
                        glLoadIdentity()

                        cameraMVM = self.root.camera.getOpenGLModelViewMatrix()

                        glMultMatrixd(cameraMVM)
                        Sofa.SofaGL.draw(self.root)
                    else:
                        print("===============> ERROR")

                    try:
                        x, y, width, height = glGetIntegerv(GL_VIEWPORT)
                    except:
                        width, height = self.surface_size[0], self.surface_size[1]
                    buff = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)

                    image_array = np.fromstring(buff, np.uint8)
                    if image_array.shape != (0,):
                        image = image_array.reshape(self.surface_size[1], self.surface_size[0], 3)
                    else:
                        image = np.zeros((self.surface_size[1], self.surface_size[0], 3))

                    image = np.flipud(image)
                    image = np.moveaxis(image, 0, 1)
                    if self.env.config['render'] == 2:
                        # time.sleep(0.1)

                        # Update the window
                        self.sim_surface = pygame.surfarray.make_surface(image)
                        self.screen.blit(self.sim_surface, (0, 0))

                        if self.agent_display:
                            self.agent_display(self.agent_surface, None)
                        self.screen.blit(self.agent_surface, (0, self.surface_size[1]))

                        # Display the modifications
                        pygame.display.update()

                        if self.save_path is not None:
                            self.save_image(image, str(self.frame)+"_"+str(num_im))
                        num_im += 1

                        if self.create_video is not None:
                            displayed_image = self.get_image()
                            self.writer.append_data(displayed_image)

        except queue.Empty:
            print("No image available")
            image = np.zeros((self.surface_size[0], self.surface_size[1], 3))

        # Update the window
        self.sim_surface = pygame.surfarray.make_surface(image)
        self.screen.blit(self.sim_surface, (0, 0))

        if self.agent_display:
            self.agent_display(self.agent_surface, None)
        self.screen.blit(self.agent_surface, (0, self.surface_size[1]))

        # Display the modifications
        pygame.display.update()

        if self.save_path is not None and self.env.config['render'] == 1:
            self.save_image(image, str(self.frame))

        if self.create_video is not None:
            displayed_image = self.get_image()
            self.writer.append_data(displayed_image)

        return self.get_image()

    def save_image(self, image, name):
        """Save the image.

        Parameters:
        ----------
            image: array
                The image we want to save.
            name: string
                The name of the picture.

        Returns:
        -------
            None.
        """
        img = Image.fromarray(image, 'RGB')
        print(">> Saving new image at location " + self.save_path + "_" + name + ".png")
        img.save(self.save_path + "/img_" + name + ".png")

    def get_image(self):
        """Take the current image on the window.

        Parameters:
        ----------
            None.


        Returns:
        -------
            The rendered image as a rbg array.

        """
        data = pygame.surfarray.array3d(self.screen)
        return np.moveaxis(data, 0, 1)

    def set_agent_display(self, agent_display):
        """Set a display callback provided by an agent.

        Agent can render it behaviour on a dedicated agent surface, or even on
        the simulation surface.

        Parameters:
        -----------
            agent_display:
                A callback provided by the agent to display on surfaces.

        Returns:
        -------
            None.
        """
        self.agent_display = agent_display

    def close(self):
        """Close the viewer.

        Parameters:
        ----------
            None.

        Returns:
        -------
            None.

        """
        glfw.terminate()
        pygame.display.quit()
        pygame.quit()


class LegacyViewer:
    def __init__(self, env, surface_size, startCmd=None):
        self.env = env
        self.surface_size = surface_size

        pygame.display.init()
        pygame.font.init()
        self.screen_size = (surface_size[0], int(1.5 * surface_size[1]))  # Screen
        self.screen = pygame.display.set_mode(self.screen_size, pygame.DOUBLEBUF)
        self.agent_display = None
        self.sim_surface = pygame.Surface((surface_size[0], surface_size[1]))
        self.agent_surface = pygame.Surface((surface_size[0], surface_size[1]//2))

        # Initialize the library
        if not glfw.init():
            print("Error during init gl")
            return
        # Set window hint NOT visible
        # glfw.window_hint(glfw.VISIBLE, False)
        # Create a windowed mode window and its OpenGL context
        self.window = glfw.create_window(self.surface_size[0], self.surface_size[1], "hidden window", None, None)
        if not self.window:
            print("ERROR glfw is dead")
            glfw.terminate()
            return
        self.startCmd = startCmd
        self.env.config.update({"render": 0})
        self.root = None
        self.reset()

        self.writer = imageio.get_writer(self.env.config['save_path']
                                         + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                                         + ".mp4", format='mp4', mode='I', fps=10)

    def reset(self):
        self.root = init_simulation(self.env.config)

    def step(self, action):
        _ = step_simulation(self.root, self.env.config, action, self.startCmd, None, viewer=self)

    def render(self):
        image = self.render_sim()
        self.sim_surface = pygame.surfarray.make_surface(image)
        self.screen.blit(self.sim_surface, (0, 0))

        # if self.agent_display:
        #     self.agent_display(self.agent_surface, None)
        # else:
        #     print("NO AGENT DISPLAY TO RENDER")
        # self.screen.blit(self.agent_surface, (0, self.surface_size[1]))

        pygame.display.update()
        displayed_image = self.get_image()
        self.writer.append_data(displayed_image)

        return self.get_image()

    def render_simulation(self, root):
        image = self.render_sim(root=root)
        self.sim_surface = pygame.surfarray.make_surface(image)
        self.screen.blit(self.sim_surface, (0, 0))
        pygame.display.update()
        displayed_image = self.get_image()
        self.writer.append_data(displayed_image)

    def render_agent(self):
        print(self.agent_display)
        if self.agent_display:
            self.agent_display(self.agent_surface, None)
        self.screen.blit(self.agent_surface, (0, self.surface_size[1]))
        pygame.display.update()

    def get_image(self):
        """
        :return: the rendered image as a rbg array
        """
        data = pygame.surfarray.array3d(self.screen)
        return np.moveaxis(data, 0, 1)

    def render_sim(self, root=None):
        if root is None:
            root = self.root
        # Make the window's context current
        glfw.make_context_current(self.window)
        glViewport(0, 0, self.surface_size[0], self.surface_size[1])
        # glEnable(GL_DEPTH_TEST)
        # glClearColor(0.5, 0.5, 0.5,1.0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_LIGHTING)
        glEnable(GL_DEPTH_TEST)

        if root:
            Sofa.SofaGL.glewInit()
            Sofa.Simulation.initVisual(root)

            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            gluPerspective(45, (self.surface_size[0] / self.surface_size[1]), 0.1, 5000.0)

            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            cameraMVM = root.camera.getOpenGLModelViewMatrix()

            glMultMatrixd(cameraMVM)
            Sofa.SofaGL.draw(root)

        else:
            print("===============> ERROR")

        x, y, width, height = glGetIntegerv(GL_VIEWPORT)
        buff = glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE)
        image_array = np.fromstring(buff, np.uint8)
        if image_array.shape != (0,):
            image = image_array.reshape(self.surface_size[1], self.surface_size[0], 3)
        else:
            image = np.zeros((self.surface_size[1], self.surface_size[0], 3))
        np.flipud(image)

        return np.moveaxis(image, 0, 1)

    def set_agent_display(self, agent_display):
        """
            Set a display callback provided by an agent, so that they can render their behaviour on a dedicated
            agent surface, or even on the simulation surface.
        :param agent_display: a callback provided by the agent to display on surfaces
        """
        self.agent_display = agent_display

    def close(self):
        return
