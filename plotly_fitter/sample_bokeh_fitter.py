from bokeh.layouts import column, row
from bokeh.models import CustomJS, Slider
from bokeh.plotting import ColumnDataSource, figure, output_file, show

# Set up the plots and their data source
source = ColumnDataSource(data=dict(T=[], S=[], I=[], R=[]))

SIR_plot = figure(plot_width=400, plot_height=400)
SIR_plot.line('T', 'S', source=source, legend_label="S", line_width=3, line_alpha=0.6, color='blue')
SIR_plot.line('T', 'I', source=source, legend_label="I", line_width=3, line_alpha=0.6, color='orange')
SIR_plot.line('T', 'R', source=source, legend_label="R", line_width=3, line_alpha=0.6, color='green')

I_plot = figure(plot_width=400, plot_height=400)
I_plot.line('T', 'I', source=source, line_width=3, line_alpha=0.6, color='orange')

# declare the interactive interface elements
trans_rate = Slider(start=0.01, end=0.4, value=0.3, step=.01, title="transmission rate ")
recov_rate = Slider(start=0.01, end=0.4, value=0.1, step=.01, title="recovery rate")

I_init = Slider(start=0.01, end=0.1, value=0.05, step=.002, title="initial infected [proportion] ")
max_time = Slider(start=10, end=200, value=50, step=1, title="time range [days] ")

callback = CustomJS(args=dict(source=source, I_init=I_init, max_time=max_time, 
                              trans_rate=trans_rate, recov_rate=recov_rate), 
                    code="""\
    let i = I_init.value;
    let s = 1-i;
    let r = 0;
    const bet = trans_rate.value;
    const gam = recov_rate.value;
    let tf = max_time.value;
    const dt = 0.1;
    const tlst = source.data.T = [0];
    const slst = source.data.S = [s];
    const ilst = source.data.I = [i];
    const rlst = source.data.R = [r];

    function odefunc(t,sir) {
        let tr = bet*sir[0]*sir[1];
        let rc = gam*sir[1];
        return [-tr, tr-rc, rc];
    }
    let sir = [s,i,r];
    for (let t = 0; t < tf; t+=dt) {
        sir = RK4Step(t,sir,dt);
        tlst.push(t+dt);
        slst.push(sir[0]);
        ilst.push(sir[1]);
        rlst.push(sir[2]);
    }
    source.change.emit();

    function axpy(a,x,y) { 
        // returns a*x+y for arrays x,y of the same length
        var k = y.length >>> 0;
        var res = new Array(k);
        while(k-->0) { res[k] = y[k] + a*x[k]; }
        return res;
    }

    function RK4Step(t,y,h) {
        var k0 = odefunc(t      ,               y );
        var k1 = odefunc(t+0.5*h, axpy(0.5*h,k0,y));
        var k2 = odefunc(t+0.5*h, axpy(0.5*h,k1,y));
        var k3 = odefunc(t+    h, axpy(    h,k2,y));
        // ynext = y+h/6*(k0+2*k1+2*k2+k3);
        return axpy(h/6,axpy(1,k0,axpy(2,k1,axpy(2,k2,k3))),y);
    }

""")
trans_rate.js_on_change('value', callback)
recov_rate.js_on_change('value', callback)

I_init.js_on_change('value', callback)
max_time.js_on_change('value', callback)

# generate the layout

parameters_panel = column(trans_rate, recov_rate)
initials_panel = column(I_init,max_time)

plots = row(SIR_plot, I_plot)
inputs = row(parameters_panel, initials_panel)

simulation = column(plots, inputs)

show(simulation)