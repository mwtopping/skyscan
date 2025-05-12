package main

import (
	"encoding/json"
	"fmt"
	"net/http"
)

func (c *apiConfig) handlerRecieveEntry(w http.ResponseWriter, r *http.Request) {

	type properties struct {
		RA    float32 `json:"RA"`
		DEC   float32 `json:"DEC"`
		X_pix float32 `json:"x_pix"`
		Y_pix float32 `json:"y_pix"`
	}

	decoder := json.NewDecoder(r.Body)
	myProperties := properties{}
	err := decoder.Decode(&myProperties)
	if err != nil {
		jsonRespondError(w, 500, "Something went wrong recieving json")
		fmt.Println(err)
		return
	}

	fmt.Printf("Recieved detection at coordinates RA:%v RA:%v\n", myProperties.RA, myProperties.DEC)

	jsonRespondPayload(w, 200, "ok")

}
