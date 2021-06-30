import plotly.graph_objects as go
import numpy as np

# https://pauliacomi.com/2020/06/07/plotly-v-bokeh.html
# It should be noted however that the Bokeh backend, Tornado, 
# operates over WebSockets. This means that communication between 
# server and client is done on a continuously connected “pipe”, 
# meaning it’s faster, asynchronous and with less overhead, allowing 
# Bokeh apps to be more feature-rich in terms of interactivity. 
# On the other hand, the Plotly server backend, Flask, is a WSGI 
# microframework, which is configured out of the box to be synchronous.
#  Plotly dashboards can’t easily save intermediary calculations for example.
def run_sample_plotly_sliders():
    # Create figure
    fig = go.Figure()

    # Add traces, one for each slider step
    for step in np.arange(0, 5, 0.1):
        fig.add_trace( 
            go.Scatter(
                visible=False,
                line=dict(color="#00CED1", width=6),
                name="𝜈 = " + str(step),
                x=np.arange(0, 10, 0.01),
                y=np.sin(step * np.arange(0, 10, 0.01))))

    # Make 10th trace visible
    fig.data[10].visible = True

    # Create and add slider
    steps = []
    for i in range(len(fig.data)):
        step = dict(
            method="update",
            args=[{"visible": [False] * len(fig.data)},
                {"title": "Slider switched to step: " + str(i)}],  # layout attribute
        )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps.append(step)

    sliders = [dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 50},
        steps=steps
    ),
    dict(
        active=10,
        currentvalue={"prefix": "Frequency: "},
        pad={"t": 150},
        steps=steps
    )]

    fig.update_layout(
        sliders=sliders
    )
    print(steps)
    fig.show()

def run_slider_multiple_plots():
    import plotly
    import plotly.graph_objs as go
    import numpy as np
    
    def fun(par1, par2, par3): 
        # some function that has three paramaters
        return par1 ** (par2/12) / (par3 ** 12)
    
    d_range = range(0, 101000, 1000)  # slider variable
    rate_range = [r/100 for r in range(1025, 1825, 1)]
    
    par3 = 12
    data = [dict(visible = False,
                name = par3,
                x = rate_range,
                y = [fun(200000 - d, r, par3) for r in rate_range])
            for d in d_range]
    data[0]['visible'] = True
    
    steps = []
    for i,d in zip(range(len(data)), d_range):
        step = dict(method = 'restyle',
                    args = ['visible', [False] * len(data)],
                    label = d)
        step['args'][1][i] = True # Toggle i'th trace to "visible"
        steps.append(step)
    
    print(steps)
    sliders = [dict(active = 0,
                    currentvalue = {"prefix": "d: "}, #label
                    pad = {"t": 50}, #position below chart
                    steps = steps)]  
    
    layout = go.Layout(sliders=sliders)
    
    plotly.offline.plot({'data': data, 'layout': layout}, auto_open=True)

if __name__ == "__main__":
    # run_sample_plotly_sliders()
    run_slider_multiple_plots()