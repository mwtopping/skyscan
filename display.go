package main

import (
	"context"
	"database/sql"
	"fmt"
	"github.com/go-echarts/go-echarts/v2/charts"
	"github.com/go-echarts/go-echarts/v2/opts"
	"github.com/google/uuid"
	"html/template"
	"log"
	"net/http"
	"net/url"
	"skyscan/internal/database"
	"strconv"
)

func (c *apiConfig) handlerDisplay(w http.ResponseWriter, r *http.Request) {

	URL := r.URL.String()
	parsedURL, err := url.Parse(URL)
	if err != nil {
		log.Println("Error parsing URL")
		return
	}
	queryParams := parsedURL.Query()
	N := 10
	if queryParams.Has("N") {
		nStr := queryParams.Get("N")
		nInt, err := strconv.Atoi(nStr)
		if err != nil {
			log.Println("Error parsing url query")
		} else {
			N = nInt
		}
	}

	transients, err := c.dbQueries.GetSomeTransients(context.Background(), int32(N))
	if err != nil {
		log.Println("Error retrieving transients")
		log.Println(err)
		return
	}

	tmpl, err := template.ParseFiles("./templates/mytemplate.html")
	if err != nil {
		log.Println("Error reading template")
		log.Println(err)
		return
	}

	//                    <td>0</td>
	//                    <td><a href="/transients/{{.ID}}">{{.ID}}</a></td>
	//                    <td>{{.Expstart}}</td>
	//                    <td>{{.Satnum.Int32}}</td>
	//                    <td>temp</td>
	//                    <td>({{.Ra1}},{{.Dec1}})</td>
	//                    <td>({{.Ra2}},{{.Dec2}})</td>
	//                    <td><img src="data:image/png;base64,{{.Imgdata}}" alt="image"></td>

	type TransientInfo struct {
		N        int
		ID       string
		Expstart string
		Satnum   int32
		Ra1      string
		Ra2      string
		Dec1     string
		Dec2     string
		Imgdata  string
		Name     string
	}

	allTransients := make([]TransientInfo, 0)
	for ii, val := range transients {
		id := int32(0)
		satellite_name := ""
		if val.Satnum.Valid == true {
			id = val.Satnum.Int32
			satellite, err := c.dbQueries.GetSatellite(context.Background(), id)
			if err != nil {
				satellite_name = ""
				fmt.Println("Error retrieving satellite info")
				fmt.Println(err)
			} else {
				fmt.Println(satellite)
				satellite_name = satellite.Name
			}
		}
		tr := TransientInfo{Name: satellite_name,
			N:        ii,
			ID:       val.ID.String(),
			Expstart: val.Expstart.Format("2006-01-02 15:04:05"),
			Satnum:   id,
			Ra1:      fmt.Sprintf("%.2f", val.Ra1),
			Ra2:      fmt.Sprintf("%.2f", val.Ra2),
			Dec1:     fmt.Sprintf("%.2f", val.Dec1),
			Dec2:     fmt.Sprintf("%.2f", val.Dec2),
			Imgdata:  val.Imgdata}

		allTransients = append(allTransients, tr)
	}

	type PageData struct {
		Transients []TransientInfo
	}

	data := PageData{Transients: allTransients}

	tmpl.Execute(w, data)

	//	top_html := `<html>
	//	<head>
	//    <style>
	//        table {
	//            border-collapse: collapse;
	//            width: 100%;
	//            margin: 20px 0;
	//        }
	//
	//        th, td {
	//            border: 1px solid #ddd;
	//            padding: 8px;
	//            text-align: left;
	//        }
	//
	//        th {
	//            background-color: #f2f2f2;
	//            font-weight: bold;
	//        }
	//
	//        tr:nth-child(even) {
	//            background-color: #f9f9f9;
	//        }
	//
	//        tr:hover {
	//            background-color: #f1f1f1;
	//        }
	//    </style>
	//	</head>
	//	<body>`
	//	w.Write([]byte(top_html))
	//	table_header := `<table>
	//					<thead>
	//					<tr>
	//					<th> N </th>
	//					<th> ID </th>
	//					<th> EXP_Start (UTC) </th>
	//	<th> SATNUM </th>
	//	<th> SATNAME </th>
	//	<th> (RA,DEC)_1 </th>
	//	<th> (RA,DEC)_2 </th>
	//	<th> Postage Stamp </th>
	//					</tr>
	//					</thead>`
	//	w.Write([]byte(table_header))
	//
	//	w.Write([]byte("<tbody>"))
	//	for ii, val := range transients {
	//		satellite_name := ""
	//		if val.Satnum.Valid == true {
	//			id := val.Satnum.Int32
	//			satellite, err := c.dbQueries.GetSatellite(context.Background(), id)
	//			if err != nil {
	//				satellite_name = ""
	//				fmt.Println("Error retrieving satellite info")
	//				fmt.Println(err)
	//			} else {
	//				fmt.Println(satellite)
	//				satellite_name = satellite.Name
	//			}
	//		}
	//
	//		writeTableRow(w, val, ii, satellite_name)
	//		//fmt.Println(val)
	//		//obj_info := fmt.Sprintf("<p>ID: %v, Detected at: %v Movement between: (RA,DEC) = (%.2f,%.2f)<->(%.2f,%.2f)</p>",
	//		//	val.ID, val.Expstart,
	//		//	val.Ra1, val.Dec1,
	//		//	val.Ra2, val.Dec2)
	//
	//		//w.Write([]byte(obj_info))
	//	}
	//
	//	w.Write([]byte("</tbody>"))
	//	bottom_html := `</body></html>`
	//	w.Write([]byte(bottom_html))

}

func writeTableRow(w http.ResponseWriter, transient database.Transient, ii int, satellite_name string) {
	w.Write([]byte("<tr>\n"))

	w.Write([]byte(fmt.Sprintf("<td>%v</td>", ii)))
	w.Write([]byte(fmt.Sprintf(`<td><a href="/transients/%v">%v</a></td>`, transient.ID, transient.ID)))
	w.Write([]byte(fmt.Sprintf("<td>%v</td>", transient.Expstart.Format("2006-01-02 15:04:05"))))
	if transient.Satnum.Valid == true {
		w.Write([]byte(fmt.Sprintf(`<td><a href="/satellites/%v">%v</a></td>`, transient.Satnum.Int32, transient.Satnum.Int32)))
	} else {
		w.Write([]byte(fmt.Sprintf("<td></td>")))
	}
	w.Write([]byte(fmt.Sprintf("<td>%s</td>", satellite_name)))
	w.Write([]byte(fmt.Sprintf("<td>(%.2f,%.2f)</td>", transient.Ra1, transient.Dec1)))
	w.Write([]byte(fmt.Sprintf("<td>(%.2f,%.2f)</td>", transient.Ra2, transient.Dec2)))
	w.Write([]byte(fmt.Sprintf(`<td><img src="data:image/png;base64,%v" alt="image"></td>`, transient.Imgdata)))

	w.Write([]byte("</tr>\n"))
}

func (c *apiConfig) handlerDisplayOne(w http.ResponseWriter, r *http.Request) {

	transientID, err := uuid.Parse(r.PathValue("transientID"))
	if err != nil {
		log.Println("Error parsing transient ID from url")
		return
	}

	transient, err := c.dbQueries.GetTransient(context.Background(), transientID)
	if err != nil {
		log.Println("Error retrieving transient from database")
		return
	}

	html := fmt.Sprintf(`<html>
		<head>
		</head>
		<body>
		<center>
		<h1>%v</h1>
		<h3>Observed in %vs exposure at %v</h3>
		<img src="data:image/png;base64,%v" alt="image" style="image-rendering: crisp-edges; min-width: 330px;" >
		</center>
		</body>
		</html>`,
		transient.ID,
		transient.Exptime,
		transient.Expstart.Format("2006-01-02 15:04:05"),
		transient.Imgdata)

	w.Write([]byte(html))
}

// Display all detections of a given satellite
func (c *apiConfig) handlerDisplaySatellite(w http.ResponseWriter, r *http.Request) {

	satelliteID, err := strconv.Atoi(r.PathValue("satelliteID"))
	if err != nil {
		log.Println("Error parsing transient ID from url")
		return
	}

	satellites, err := c.dbQueries.GetTransientsOfSatellite(context.Background(),
		sql.NullInt32{Int32: int32(satelliteID), Valid: true})
	if err != nil {
		log.Println("Error retrieving transient from database")
		return
	}

	scatter := charts.NewScatter()
	scatter.SetGlobalOptions(
		charts.WithXAxisOpts(opts.XAxis{
			Type:  "value",
			Scale: opts.Bool(true),
			Name:  "Right Ascension"}),
		charts.WithYAxisOpts(opts.YAxis{
			Type:  "value",
			Scale: opts.Bool(true),
			Name:  "Declination"}),
		charts.WithInitializationOpts(
			opts.Initialization{Width: "30%"}),
	)

	data := []opts.ScatterData{}
	for _, s := range satellites {
		data = append(data, opts.ScatterData{Value: []interface{}{s.Ra1, s.Dec1}})
		data = append(data, opts.ScatterData{Value: []interface{}{s.Ra2, s.Dec2}})
	}

	scatter.AddSeries(r.PathValue("satelliteID"), data)

	res := scatter.RenderSnippet()

	tophtml := fmt.Sprintf(`<html>
		<head>
		<script src="https://go-echarts.github.io/go-echarts-assets/assets/echarts.min.js"></script>
		<style>
		.wrapper {
		  display: grid;
		  grid-template-columns: 0.3fr 0.3fr 0.3fr;
		  max-width: fit-content;
		  margin-left: auto;
		  margin-right: auto;
		}
		.graph {
		  margin-left: auto;
		  margin-right: auto;
	  
		}
		</style>
		</head>
		<body>
		<center>
		`)

	w.Write([]byte(tophtml))

	w.Write([]byte(`<div class="graph">`))
	w.Write([]byte(res.Element))
	w.Write([]byte(res.Script))
	w.Write([]byte("</div>"))

	w.Write([]byte(`<div class="wrapper">`))
	for _, satellite := range satellites {
		html := fmt.Sprintf(`
			<div>
		<p style="font-size: 14px;">%v</p>
		<p style="font-size: 14px;">Observed in %vs exposure at %v</p>
		<img src="data:image/png;base64,%v" alt="image" style="image-rendering: crisp-edges" width="80%";" >
			</div>`,
			satellite.ID,
			satellite.Exptime,
			satellite.Expstart.Format("2006-01-02 15:04:05"),
			satellite.Imgdata)
		w.Write([]byte(html))
	}
	bothtml := fmt.Sprintf(`
		</div>
		</center>
		</body>
		</html>`)
	fmt.Println(res)

	w.Write([]byte(bothtml))

}
