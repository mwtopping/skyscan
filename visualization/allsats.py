import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import psycopg2




conn = psycopg2.connect(
    host="localhost",
    database="skyscan",
    user="michael",
    password=""
)

cursor = conn.cursor()

getSatnums_query = """
SELECT satnum from transients
GROUP BY satnum
"""

getElements_query = """
SELECT id, created_at, updated_at, expstart, exptime, ra1, ra2, dec1, dec2, satnum, imgdata from transients
	WHERE satnum = %s;
"""

ras = []
decs = []

traces = []

times = []
cursor.execute(getSatnums_query, )
rows = cursor.fetchall()
colors = px.colors.qualitative.Plotly
ncolors = len(colors)
for ii, row in enumerate(rows):
    num = row[0]
    cursor.execute(getElements_query, (num,))
    results = cursor.fetchall()
    for result in results:
        id, created_at, updated_at, expstart, exptime, ra1, ra2, dec1, dec2, satnum, imgdata = result
        print(id, expstart)
        ras.append(ra1)
        ras.append(ra2)
        decs.append(dec1)
        decs.append(dec2)
        color = colors[ii%ncolors]
        traces.append(go.Scattergl(x=[ra1, ra2], y=[dec1, dec2], mode='lines', showlegend=False, 
                            line={"color":color,"width":2}))


fig = go.Figure()
fig.add_traces(traces)
fig.update_layout(dragmode='pan', xaxis={"fixedrange":False}, yaxis={"fixedrange":False})
fig.show()
#fig = px.scatter(x=ras, y=decs)
#fig.write_html("./test.html")
