package main

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	grob "github.com/MetalBlueberry/go-plotly/generated/v2.31.1/graph_objects"
	"github.com/MetalBlueberry/go-plotly/pkg/types"
	"html/template"
	"log"
	"net/http"
)

var Colors = []string{"#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000", "#000000", "#888888"}

func (c *apiConfig) handlerDisplayAll(w http.ResponseWriter, r *http.Request) {

	transients, err := c.dbQueries.GetAllTransients(context.Background())
	if err != nil {
		log.Println("Error getting transients")
		log.Println(err)
		return
	}

	data := make([]types.Trace, 0, len(transients))

	for _, sat := range transients {
		if sat.Satnum.Valid == true {
			color := Colors[int(sat.Satnum.Int32)%len(Colors)]
			data = append(data, &grob.Scattergeo{
				Lon:        types.DataArray([]float64{sat.Ra1, sat.Ra2}),
				Lat:        types.DataArray([]float64{sat.Dec1, sat.Dec2}),
				Mode:       grob.ScattergeoModeLines,
				Showlegend: types.False,
				Line:       &grob.ScattergeoLine{Color: types.Color(color)},
			})
		}
	}

	layout := &grob.Layout{
		Width:  types.N(800),
		Height: types.N(800),
		Geo: &grob.LayoutGeo{
			Projection: &grob.LayoutGeoProjection{
				Type: grob.LayoutGeoProjectionTypeMollweide,
			},
			Showland:       types.False,
			Showocean:      types.False,
			Showlakes:      types.False,
			Showcountries:  types.False,
			Showcoastlines: types.False,
			Lataxis: &grob.LayoutGeoLataxis{
				Dtick:     types.N(15),
				Showgrid:  types.True,
				Gridwidth: types.N(1),
			},
			Lonaxis: &grob.LayoutGeoLonaxis{
				Dtick:     types.N(15),
				Showgrid:  types.True,
				Gridwidth: types.N(1),
			},
		},

		Dragmode:  grob.LayoutDragmodePan,
		Hovermode: grob.LayoutHovermodeFalse,
		Xaxis:     &grob.LayoutXaxis{Title: &grob.LayoutXaxisTitle{Text: types.StringType("RA")}},
		Yaxis:     &grob.LayoutYaxis{Title: &grob.LayoutYaxisTitle{Text: types.StringType("Dec")}},
	}
	plotconfig := &grob.Config{
		ScrollZoom: types.True,
	}

	fig := &grob.Fig{
		Data:   data,
		Layout: layout,
		Config: plotconfig,
	}

	buf := figToBuffer(fig)

	w.Write(buf.Bytes())
}

func figToBuffer(fig types.Fig) *bytes.Buffer {
	figBytes, err := json.Marshal(fig)
	if err != nil {
		panic(err)
	}
	var singleFileHTML = `
	<head>
		<script src="{{ .Version.Cdn }}"></script>
	</head>
	
	<body>
		<center>
		<div id="plot"></div>
	<script>
		data = JSON.parse(atob('{{ .B64Content }}'))
		Plotly.newPlot('plot', data);
	</script>
		</center>
	</body>
	`
	tmpl, err := template.New("plotly").Parse(singleFileHTML)
	if err != nil {
		panic(err)
	}
	buf := &bytes.Buffer{}
	data := struct {
		Version    types.Version
		B64Content string
	}{
		Version: fig.Info(),
		// Encode to avoid problems with special characters
		B64Content: base64.StdEncoding.EncodeToString(figBytes),
	}

	err = tmpl.Execute(buf, data)
	if err != nil {
		panic(err)
	}
	return buf
}
