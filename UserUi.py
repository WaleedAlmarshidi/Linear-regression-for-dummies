import logging
import time
from kivy.uix.widget import Widget
from numpy.core.fromnumeric import size
from scipy.stats.stats import mode, pearsonr
from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.lang import Builder
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot
from kivy.animation import Animation
from kivy.uix.screenmanager import ScreenManager, Screen

screen_manager = ScreenManager()
WidgetTree = Builder.load_file("UserUi.kv")

class UserUi(App):
    def build(self):
        return WidgetTree

class LogsLabel(Label):
    pass

class Graph(FigureCanvasKivyAgg):
    pass

class InstructionPage(Screen):
    pass

class MainPage(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def AddToModelBuildingLogs(self, msg):
        label = LogsLabel(text = str(msg))
        self.ids.ModelBuildingSection.add_widget(label)

    def AddToModelSelectionLogs(self, msg):
        label = LogsLabel(text = str(msg))
        self.ids.ModelFilteringSection.add_widget(label)

    def AddToErrorTestsLogs(self, msg):
        label = LogsLabel(text = str(msg))
        self.ids.ErrorTestsSection.add_widget(label)

    def AddGraphToModelBuildingSection(self, graph):
        self.ids.ModelBuildingSection.add_widget(FigureCanvasKivyAgg(graph))
    
    def AddGraphToModelFilteringSection(self, graph):
        self.ids.ModelFilteringSection.add_widget(FigureCanvasKivyAgg(graph))
    
    def AddGraphToErrorTestsSection(self, graph):
        # list(graph.get_size_inches()*graph.dpi)
        # graph.set_size_inches(0.5, 0.5, forward=True)
        self.ids.ErrorTestsSection.add_widget(Graph(graph)) 

    def Run(self, widgettree):
        global WidgetTree
        WidgetTree = widgettree
        app = UserUi()
        app.run()


