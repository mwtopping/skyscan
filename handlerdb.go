package main

import (
	"context"
	"fmt"
	"net/http"
)

func (c *apiConfig) handlerReset(w http.ResponseWriter, r *http.Request) {

	err := c.dbQueries.Reset(context.Background())
	if err != nil {
		fmt.Println("Error resetting database")
		fmt.Println(err)
		jsonRespondError(w, 500, "Error resetting database")
		return
	}

	jsonRespondPayload(w, 200, "ok")
}
