package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"net/url"
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
	for _, val := range transients {
		fmt.Println(val)
		obj_info := fmt.Sprintf("ID: %v, Detected at: %v Movement between: (RA,DEC) = (%.2f,%.2f)<->(%.2f,%.2f)\n",
			val.ID, val.Expstart,
			val.Ra1, val.Dec1,
			val.Ra2, val.Dec2)

		w.Write([]byte(obj_info))
	}

}
