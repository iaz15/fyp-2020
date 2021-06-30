import numpy as np

from bokeh.layouts import column, row
from bokeh.models import CustomJS, Slider
from bokeh.plotting import ColumnDataSource, figure, output_file, show

def bokeh_sliders_1():
    x = np.linspace(0, 10, 500)
    y = np.sin(x)

    source = ColumnDataSource(data=dict(x=x, y=y))

    plot = figure(y_range=(-10, 10), plot_width=400, plot_height=400)

    plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

    amp_slider = Slider(start=0.1, end=10, value=1, step=.1, title="Amplitude")
    freq_slider = Slider(start=0.1, end=10, value=1, step=.1, title="Frequency")
    phase_slider = Slider(start=0, end=6.4, value=0, step=.1, title="Phase")
    offset_slider = Slider(start=-5, end=5, value=0, step=.1, title="Offset")

    callback = CustomJS(args=dict(source=source, amp=amp_slider, freq=freq_slider, phase=phase_slider, offset=offset_slider),
                        code="""
        const data = source.data;
        const A = amp.value;
        const k = freq.value;
        const phi = phase.value;
        const B = offset.value;
        const x = data['x']
        const y = data['y']
        for (var i = 0; i < x.length; i++) {
            y[i] = B + A*Math.sin(k*x[i]+phi);
        }
        source.change.emit();
    """)

    amp_slider.js_on_change('value', callback)
    freq_slider.js_on_change('value', callback)
    phase_slider.js_on_change('value', callback)
    offset_slider.js_on_change('value', callback)

    layout = row(
        plot,
        column(amp_slider, freq_slider, phase_slider, offset_slider),
    )

    output_file("slider.html", title="slider.py example")

    show(layout)

def bokeh_sliders_3():
    from bokeh.layouts import column
    from bokeh.models import CustomJS, ColumnDataSource, Slider
    from bokeh.plotting import Figure, output_notebook, show

    output_notebook()

    x = [x*0.005 for x in range(0, 200)]
    y = x
    x1 = [x1*0.005 for x1 in range(0, 200)]
    y1 = x1

    source = ColumnDataSource(data=dict(x=x, y=y))
    source1 = ColumnDataSource(data=dict(x1=x1,y1=y1))

    plot = Figure(plot_width=400, plot_height=400)
    plot.line('x', 'y', source=source, line_width=3, line_alpha=0.6)

    plot1 = Figure(plot_width=400, plot_height=400)
    plot1.line('x1', 'y1', source=source1, line_width=3, line_alpha=0.6)

    callback = CustomJS(args=dict(source=source, source1=source1), code="""
        var data = source.data;
        var data1 = source1.data;
        var f1 =cb_obj.value
        var f = cb_obj.value
        x = data['x']
        y = data['y']
        x1 = data['x1']
        y1 = data['y1']

        for (i = 0; i < x.length; i++) {
            y[i] = Math.pow(x[i], f)
        }
        for (i = 0; i < x1.length; i++) {
            y1[i] = Math.pow(x1[i], f1)
        }
        source.change.emit();
        source1.change.emit();
    """)

    slider = Slider(start=0.1, end=4, value=1, step=.1, title="power")
    slider.js_on_change('value', callback)

    layout = column(slider, plot,plot1)

    show(layout)


if __name__ == "__main__":
    # bokeh_sliders_1()
    bokeh_sliders_3()