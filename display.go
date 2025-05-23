package main

import (
	"context"
	"fmt"
	"github.com/google/uuid"
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
	top_html := `<html>
	<head>
    <style>
        table {
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }

        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }

        th {
            background-color: #f2f2f2;
            font-weight: bold;
        }

        tr:nth-child(even) {
            background-color: #f9f9f9;
        }

        tr:hover {
            background-color: #f1f1f1;
        }
    </style>
	</head>
	<body>`
	w.Write([]byte(top_html))
	table_header := `<table>
					<thead>
					<tr>
					<th> N </th>
					<th> ID </th>
					<th> EXP_Start (UTC) </th>
	<th> SATNUM </th>
	<th> SATNAME </th>
	<th> (RA,DEC)_1 </th>
	<th> (RA,DEC)_2 </th>
	<th> Postage Stamp </th>
					</tr>
					</thead>`
	w.Write([]byte(table_header))

	w.Write([]byte("<tbody>"))
	for ii, val := range transients {
		satellite_name := ""
		if val.Satnum.Valid == true {
			id := val.Satnum.Int32
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

		writeTableRow(w, val, ii, satellite_name)
		//fmt.Println(val)
		//obj_info := fmt.Sprintf("<p>ID: %v, Detected at: %v Movement between: (RA,DEC) = (%.2f,%.2f)<->(%.2f,%.2f)</p>",
		//	val.ID, val.Expstart,
		//	val.Ra1, val.Dec1,
		//	val.Ra2, val.Dec2)

		//w.Write([]byte(obj_info))
	}

	w.Write([]byte("</tbody>"))
	bottom_html := `</body></html>`
	w.Write([]byte(bottom_html))

}

func writeTableRow(w http.ResponseWriter, transient database.Transient, ii int, satellite_name string) {
	w.Write([]byte("<tr>\n"))

	w.Write([]byte(fmt.Sprintf("<td>%v</td>", ii)))
	w.Write([]byte(fmt.Sprintf("<td>%v</td>", transient.ID)))
	w.Write([]byte(fmt.Sprintf("<td>%v</td>", transient.Expstart.Format("2006-01-02 15:04:05"))))
	if transient.Satnum.Valid == true {
		w.Write([]byte(fmt.Sprintf("<td>%v</td>", transient.Satnum.Int32)))
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
