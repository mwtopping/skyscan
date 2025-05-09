package main

import (
	"encoding/json"
	"log"
	"net/http"
)

func jsonRespondError(w http.ResponseWriter, errCode int, errMsg string) {
	type errorret struct {
		Error string `json:"error"`
	}

	error := errorret{Error: errMsg}

	ret, _ := json.Marshal(error)
	w.WriteHeader(errCode)
	w.Write(ret)

}

func jsonRespondPayload(w http.ResponseWriter, resCode int, payload interface{}) {
	w.Header().Add("Content-Type", "application/json")
	w.WriteHeader(resCode)
	ret, err := json.Marshal(payload)
	if err != nil {
		log.Println("Error marshalling payload")
	}
	w.Write(ret)
}
