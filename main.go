package main

import (
	"fmt"
	"log"
	"net/http"
)

func main() {
	fmt.Println("Starting SkyScan")

	const port = ":8080"

	cfg := apiConfig{Port: port}

	mux := http.NewServeMux()

	mux.HandleFunc("/", cfg.handlerRecieveEntry)

	server := &http.Server{Handler: mux, Addr: cfg.Port}

	log.Printf("Server listening on port%v\n", cfg.Port)
	log.Fatal(server.ListenAndServe())

}

//func handle(w http.ResponseWriter, r *http.Request) {
//	fmt.Println("Got request!")
//}
