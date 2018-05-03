import os

from bokeh import plotting


class AverageMeter(object):

    def __init__(self):
        self.sum = 0
        self.count = 0
        self.average = None

    def update(self, value, number=1):
        self.sum += value * number
        self.count += number
        self.average = self.sum / self.count


class PlotHelper(object):

    def __init__(self, filename):
        self.filename = filename

        self.loss_g_list = []
        self.loss_d_list = []
        self.indices = []

    def append(self, loss_g, loss_d, index):
        self.loss_g_list.append(loss_g)
        self.loss_d_list.append(loss_d)
        self.indices.append(index)

        self.plot()

    def plot(self):
        figure = plotting.figure(sizing_mode='stretch_both')
        figure.line(
            self.indices,
            self.loss_g_list,
            line_color='green',
            alpha=0.5,
            line_width=5,
            legend='loss g')
        figure.line(
            self.indices,
            self.loss_d_list,
            line_color='blue',
            alpha=0.5,
            line_width=5,
            legend='loss d')

        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        plotting.output_file(self.filename)
        plotting.save(figure)
