package main

import (
	"context"
	"fmt"
	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/event"
	"github.com/go-echarts/go-echarts/v2/opts"
	"log"
	"net/http"
)

var colors = []string{"#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000", "#000000", "#888888"}

func (c *apiConfig) handlerDisplayAll(w http.ResponseWriter, r *http.Request) {

	line := charts.NewLine()
	JFunc := ` (params) => window.open(params.name, '_self')`
	line.SetGlobalOptions(
		charts.WithTooltipOpts(opts.Tooltip{
			Show:      opts.Bool(true),
			Trigger:   "item",
			TriggerOn: "click",
		}),
		charts.WithEventListeners(event.Listener{
			EventName: "click",
			Handler:   opts.FuncOpts(JFunc),
		}),
		charts.WithInitializationOpts(
			opts.Initialization{Width: "1024px", Height: "768px", ChartID: "clickable_chart"}),
		charts.WithDataZoomOpts(opts.DataZoom{
			Type:       "inside",
			Start:      0,
			End:        100,
			XAxisIndex: []int{0},
		}),
		charts.WithDataZoomOpts(opts.DataZoom{
			Type:       "inside",
			Start:      0,
			End:        100,
			YAxisIndex: []int{0},
		}),
		charts.WithLegendOpts(opts.Legend{
			Show: opts.Bool(false),
		}),
		charts.WithXAxisOpts(opts.XAxis{
			Type:  "value",
			Scale: opts.Bool(true),
			Name:  "Right Ascension"}),
		charts.WithYAxisOpts(opts.YAxis{
			Type:  "value",
			Scale: opts.Bool(true),
			Name:  "Declination"}),
	)

	satelliteIDs, err := c.dbQueries.GetUniqueTransients(context.Background())
	if err != nil {
		log.Println("Error getting unique transient IDs")
		log.Println(err)
		return
	}

	for ii, satID := range satelliteIDs {
		colorindex := ii % 7

		col := colors[colorindex]
		if satID.Valid == true {
			transients, err := c.dbQueries.GetTransientsOfSatellite(context.Background(),
				satID)
			if err != nil {
				log.Println("Error retrieving transient from database")
				return
			}

			// we have all the transients
			data := []opts.ScatterData{}
			//			linedata := []opts.LineData{}
			for _, s := range transients {
				data = append(data, opts.ScatterData{Value: []interface{}{s.Ra1, s.Dec1}, SymbolSize: 5})
				data = append(data, opts.ScatterData{Value: []interface{}{s.Ra2, s.Dec2}, SymbolSize: 5})
				linedata := []opts.LineData{}
				uniqueName := fmt.Sprintf("Sat%d_Seg%d", satID.Int32, ii)
				linedata = append(linedata, opts.LineData{Value: []interface{}{s.Ra1, s.Dec1}, Name: "http://google.com"})
				linedata = append(linedata, opts.LineData{Value: []interface{}{s.Ra2, s.Dec2}, Name: "http://google.com"})
				line.AddSeries(fmt.Sprintf(uniqueName, satID.Int32), linedata, charts.WithLineChartOpts(opts.LineChart{
					Smooth: opts.Bool(true),
					Symbol: "none",
				}),
					charts.WithLineStyleOpts(opts.LineStyle{
						Color: col,
						Width: 4,
					}),
				)

				centerdata := []opts.LineData{}
				centerdata = append(centerdata, opts.LineData{
					Value: []interface{}{0.5 * (s.Ra1 + s.Ra2), 0.5 * (s.Dec1 + s.Dec2)},
					Name:  fmt.Sprintf("/transients/%v", s.ID)})

				line.AddSeries(fmt.Sprintf(uniqueName, satID.Int32), centerdata, charts.WithLineChartOpts(opts.LineChart{
					Smooth: opts.Bool(true),
				}),
					charts.WithItemStyleOpts(opts.ItemStyle{
						Color: col,
					}),
				)
			}

		}
	}

	lineres := line.RenderSnippet()

	tophtml := fmt.Sprintf(`<html>
		<head>
		<script src="https://go-echarts.github.io/go-echarts-assets/assets/echarts.min.js"></script>
		</head>
		<body>
		<center>`)

	w.Write([]byte(tophtml))

	w.Write([]byte(lineres.Element))
	w.Write([]byte(lineres.Script))

	bothtml := fmt.Sprintf(`
		</center>
		</body>
		</html>`)

	w.Write([]byte(bothtml))

}
